from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np

from simulator.models import PipelineEvent


FAULT_TYPES: Tuple[str, ...] = (
    "latency_spike",
    "dropped_connection",
    "schema_drift",
    "partition_skew",
    "throughput_collapse",
    "error_burst",
)


@dataclass
class FaultSpec:
    """
    Pure value object so FaultSchedule can be serialised to YAML without carrying
    stateful RNG or callback references.

    magnitude is intentionally dimensionless — its interpretation is fault-type-specific:
    a multiplier for latency_spike and partition_skew, a probability for
    dropped_connection and error_burst, a reduction fraction for schema_drift, and a
    divisor for throughput_collapse.
    """

    fault_type: str          # must be one of FAULT_TYPES
    target_stage_id: str
    start_offset_s: float    # seconds after simulation_start when the fault activates
    duration_s: float        # how long the fault remains active
    magnitude: float         # severity — interpretation is fault-type-specific
    seed: int                # seeds the per-spec RNG for reproducible probabilistic faults


@dataclass
class FaultSchedule:
    """
    Co-locates simulation_start with the spec list — every active-window check needs
    both, and keeping them together prevents accidentally supplying a mismatched start
    time that silently activates faults at wrong offsets.

    fault_specs is ordered by start_offset_s by convention for readability; the
    injector does not depend on list order.
    """

    simulation_start: datetime
    fault_specs: List[FaultSpec]

    def active_spec_indices_at(self, event_time: datetime, stage_id: str) -> List[int]:
        """
        Returns spec indices rather than FaultSpec objects so FaultInjector can look
        up pre-seeded per-spec RNGs without a second linear search. Keying RNGs by
        spec identity would require FaultSpec to be hashable, conflicting with keeping
        it mutable for YAML round-trips.
        """
        offset_s = (event_time - self.simulation_start).total_seconds()
        return [
            i
            for i, spec in enumerate(self.fault_specs)
            if (
                spec.target_stage_id == stage_id
                and spec.start_offset_s <= offset_s <= spec.start_offset_s + spec.duration_s
            )
        ]


class FaultInjector:
    """
    Pre-seeds one RNG per FaultSpec at construction time. Seeding inside inject()
    would reset RNG state on every call, making every event draw the same value —
    a 50% drop rate would become 0% or 100% depending on that one draw.

    When multiple faults target the same stage simultaneously, they are applied in
    spec-list order via chained dataclasses.replace() calls — composition is
    intentional to produce realistic combined failure modes.
    """

    def __init__(self, schedule: FaultSchedule) -> None:
        self._schedule = schedule
        # Keyed by list index because FaultSpec is mutable (not hashable).
        # Index is stable since fault_specs is never mutated after construction.
        self._spec_rngs: List[np.random.Generator] = [
            np.random.default_rng(spec.seed) for spec in schedule.fault_specs
        ]

    def inject(self, event: PipelineEvent) -> PipelineEvent:
        """
        Returns the original event object unchanged when no fault is active —
        the common path pays only the cost of one active-window lookup rather than
        a dataclasses.replace() allocation.
        """
        active_indices = self._schedule.active_spec_indices_at(
            event.event_time, event.stage_id
        )
        if not active_indices:
            return event

        mutated = event
        for idx in active_indices:
            spec = self._schedule.fault_specs[idx]
            mutated = self._apply_fault(mutated, spec, self._spec_rngs[idx])
        return mutated

    def _apply_fault(
        self,
        event: PipelineEvent,
        spec: FaultSpec,
        rng: np.random.Generator,
    ) -> PipelineEvent:
        """
        Sets fault_label on every event within the fault window, including events that
        probabilistic faults decide not to mutate. The label represents ground truth
        about the fault window, not the per-event mutation outcome — a detector that
        misses a fault-window event is a false negative even if that event kept its
        normal status.
        """
        if spec.fault_type == "latency_spike":
            # magnitude is the latency multiplier. ±20% uniform jitter prevents
            # a perfectly scaled distribution that would make detection trivially easy
            # for any threshold-based detector.
            noise = float(rng.uniform(0.8, 1.2))
            return dataclasses.replace(
                event,
                latency_ms=event.latency_ms * spec.magnitude * noise,
                fault_label="latency_spike",
            )

        if spec.fault_type == "dropped_connection":
            # magnitude is drop probability (0.0–1.0). Dropped events get zero rows
            # and payload; latency triples to model the TCP timeout cost.
            if rng.random() < spec.magnitude:
                return dataclasses.replace(
                    event,
                    status="error",
                    row_count=0,
                    payload_bytes=0,
                    latency_ms=event.latency_ms * 3.0,
                    fault_label="dropped_connection",
                )
            return dataclasses.replace(event, fault_label="dropped_connection")

        if spec.fault_type == "schema_drift":
            # magnitude is the fraction of rows that fail schema validation (0.0–1.0).
            # All events in the window get status="schema_error" — the error is
            # structural, not probabilistic per-event.
            surviving_fraction = max(0.0, 1.0 - spec.magnitude)
            return dataclasses.replace(
                event,
                row_count=int(event.row_count * surviving_fraction),
                status="schema_error",
                fault_label="schema_drift",
            )

        if spec.fault_type == "partition_skew":
            # magnitude is the skew factor. Payload and latency scale together —
            # scaling them independently would produce unrealistic combinations that
            # confuse latency-payload correlation in the detection layer.
            return dataclasses.replace(
                event,
                payload_bytes=int(event.payload_bytes * spec.magnitude),
                latency_ms=event.latency_ms * spec.magnitude,
                fault_label="partition_skew",
            )

        if spec.fault_type == "throughput_collapse":
            # magnitude is the collapse divisor. Floored at 1 rather than 0 to
            # distinguish from dropped_connection — completely zero throughput is
            # indistinguishable from a connection failure in downstream metrics.
            return dataclasses.replace(
                event,
                row_count=max(1, int(event.row_count / spec.magnitude)),
                fault_label="throughput_collapse",
            )

        if spec.fault_type == "error_burst":
            # magnitude is the error rate (0.0–1.0). Row and payload data are preserved —
            # this models a processing failure, not a network failure. The causal
            # signature differs: throughput stays high while error rate spikes.
            if rng.random() < spec.magnitude:
                return dataclasses.replace(
                    event,
                    status="error",
                    fault_label="error_burst",
                )
            return dataclasses.replace(event, fault_label="error_burst")

        raise ValueError(
            f"Unknown fault_type '{spec.fault_type}' — must be one of {FAULT_TYPES}"
        )
