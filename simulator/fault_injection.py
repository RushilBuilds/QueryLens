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
    I'm keeping FaultSpec as a pure value object so FaultSchedule can be serialised
    to YAML for ScenarioConfig without carrying stateful RNG or callback references.
    The seed here belongs to the FaultInjector's per-spec RNG — separating it from
    the workload seed means fault behaviour is independently reproducible regardless
    of how many workload events were generated before the fault window opens.

    magnitude is intentionally dimensionless. Its interpretation is fault-type-specific:
    a multiplier for latency_spike and partition_skew, a probability for
    dropped_connection and error_burst, a reduction fraction for schema_drift, and a
    divisor for throughput_collapse. This avoids a separate parameter class per fault
    type while keeping the contract explicit in each branch of _apply_fault.
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
    I'm co-locating simulation_start with the spec list rather than passing it
    separately to inject() because every active-window check requires both pieces of
    information. Keeping them together prevents callers from accidentally supplying a
    mismatched start time and silently activating faults at wrong offsets.

    fault_specs is ordered by start_offset_s by convention, not enforcement — the
    injector does not depend on list order. Ordering is a readability aid so a YAML
    scenario config reads chronologically.
    """

    simulation_start: datetime
    fault_specs: List[FaultSpec]

    def active_spec_indices_at(self, event_time: datetime, stage_id: str) -> List[int]:
        """
        I'm returning spec indices rather than FaultSpec objects so FaultInjector can
        use the same indices to look up pre-seeded per-spec RNGs without a second
        linear search. The alternative — keying RNGs by spec identity — would require
        FaultSpec to be hashable, which conflicts with keeping it mutable for YAML
        round-trips.
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
    I'm pre-seeding one RNG per FaultSpec at construction time rather than seeding
    inside inject(). Seeding inside inject() would reset the RNG state on every call,
    making every event draw the same random value — a 50% drop rate would become
    either 0% or 100% depending on whether that one draw lands above or below the
    threshold. Pre-seeded stateful RNGs advance independently per event, producing
    the intended probabilistic distribution across the full fault window.

    When multiple faults target the same stage at the same time, they are applied in
    spec-list order via chained dataclasses.replace() calls. Composition is
    intentional: a latency_spike and a dropped_connection active simultaneously
    should produce high-latency error events, which is more realistic than either
    fault overwriting the other.
    """

    def __init__(self, schedule: FaultSchedule) -> None:
        self._schedule = schedule
        # I'm keying RNGs by list index because FaultSpec is mutable (not hashable).
        # Index is stable since fault_specs is never mutated after construction.
        self._spec_rngs: List[np.random.Generator] = [
            np.random.default_rng(spec.seed) for spec in schedule.fault_specs
        ]

    def inject(self, event: PipelineEvent) -> PipelineEvent:
        """
        I'm returning the original event object unchanged when no fault is active so
        the common path — most events, most of the time, are fault-free — pays only
        the cost of one active-window lookup rather than a dataclasses.replace()
        allocation. This matters in the simulator hot loop where inject() is called
        for every event across every stage.
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
        I'm setting fault_label on every event within the fault window, including
        events that probabilistic faults decide not to mutate (e.g., an error_burst
        event whose draw fell below the error threshold). The label represents ground
        truth about the fault window, not the per-event mutation outcome. A detector
        that misses an event during an active fault window is a false negative even
        if that particular event kept its normal status.
        """
        if spec.fault_type == "latency_spike":
            # magnitude is the latency multiplier. I'm adding ±20% uniform jitter so
            # the injected distribution is not a perfectly scaled version of the
            # baseline — real spikes have variance, and a perfectly clean multiple
            # would make detection trivially easy for any threshold-based detector.
            noise = float(rng.uniform(0.8, 1.2))
            return dataclasses.replace(
                event,
                latency_ms=event.latency_ms * spec.magnitude * noise,
                fault_label="latency_spike",
            )

        if spec.fault_type == "dropped_connection":
            # magnitude is drop probability (0.0–1.0). Dropped events get zero rows
            # and payload because no data was transferred; latency triples to model
            # the TCP timeout cost incurred before the error is surfaced to the client.
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
            # All events in the window get status="schema_error" because the stage is
            # processing a malformed schema regardless of whether individual rows
            # survive — the error is structural, not probabilistic.
            surviving_fraction = max(0.0, 1.0 - spec.magnitude)
            return dataclasses.replace(
                event,
                row_count=int(event.row_count * surviving_fraction),
                status="schema_error",
                fault_label="schema_drift",
            )

        if spec.fault_type == "partition_skew":
            # magnitude is the skew factor applied to the hot partition. Payload and
            # latency scale together because a heavier partition takes proportionally
            # longer to process. Scaling them independently would produce unrealistic
            # combinations (huge payload, normal latency) that confuse latency-payload
            # correlation in the detection layer.
            return dataclasses.replace(
                event,
                payload_bytes=int(event.payload_bytes * spec.magnitude),
                latency_ms=event.latency_ms * spec.magnitude,
                fault_label="partition_skew",
            )

        if spec.fault_type == "throughput_collapse":
            # magnitude is the collapse divisor; row_count drops to 1/magnitude of
            # normal. Floored at 1 rather than 0 to distinguish this fault from
            # dropped_connection — a completely zero throughput is indistinguishable
            # from a connection failure in downstream metrics, and preserving one row
            # keeps the causal signatures distinct.
            return dataclasses.replace(
                event,
                row_count=max(1, int(event.row_count / spec.magnitude)),
                fault_label="throughput_collapse",
            )

        if spec.fault_type == "error_burst":
            # magnitude is the error rate (0.0–1.0). Unlike dropped_connection, row
            # and payload data are preserved — this models a processing failure (e.g.,
            # a downstream write rejection or transform exception) rather than a
            # network failure. The causal signature differs: throughput metrics stay
            # high while error rate spikes, as opposed to throughput collapsing with
            # the error rate.
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
