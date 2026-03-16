from datetime import datetime
from typing import List

import numpy as np
import pytest

from simulator.fault_injection import (
    FAULT_TYPES,
    FaultInjector,
    FaultSchedule,
    FaultSpec,
)
from simulator.models import PipelineEvent
from simulator.workload import PoissonEventGenerator, WorkloadProfile

# I'm fixing the simulation start to a known timestamp so every active-window
# calculation is deterministic. Tests that need events outside the fault window
# can set start_offset_s to a value larger than the total event span.
SIM_START = datetime(2024, 1, 1, 0, 0, 0)
N_EVENTS = 1_000


def _make_baseline_events(
    stage_id: str = "stage_a",
    n: int = N_EVENTS,
) -> List[PipelineEvent]:
    """
    I'm generating baseline events via PoissonEventGenerator rather than constructing
    PipelineEvent objects by hand so the latency, payload, and row_count distributions
    match what the simulator will actually produce. Hand-crafted constants would make
    the statistical assertions fragile if WorkloadProfile defaults change.
    """
    profile = WorkloadProfile(
        arrival_rate_lambda=10.0,
        payload_mean_bytes=4096.0,
        payload_std_bytes=1024.0,
        max_concurrency=4,
    )
    rng = np.random.default_rng(seed=42)
    gen = PoissonEventGenerator(profile=profile, stage_id=stage_id, rng=rng)
    return list(gen.generate(n_events=n, start_time=SIM_START))


def _full_window_spec(fault_type: str, magnitude: float, seed: int = 0) -> FaultSpec:
    """
    I'm factoring this out because every statistical test needs a fault that covers
    all N_EVENTS. Repeating the start_offset_s=0 / duration_s=1e9 pattern across
    every test would obscure what each test is actually asserting.
    """
    return FaultSpec(
        fault_type=fault_type,
        target_stage_id="stage_a",
        start_offset_s=0.0,
        duration_s=1e9,
        magnitude=magnitude,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Fault-label and identity contract
# ---------------------------------------------------------------------------


def test_inject_returns_original_object_outside_fault_window() -> None:
    """
    I'm asserting object identity (not just equality) because the inject() docstring
    documents that the original object is returned unchanged when no fault is active.
    This is a performance contract, not just a correctness one — allocating a copy for
    every fault-free event would add measurable overhead in the simulator hot loop.
    """
    events = _make_baseline_events()
    spec = FaultSpec(
        fault_type="latency_spike",
        target_stage_id="stage_a",
        start_offset_s=1e9,  # far in the future — no event falls in this window
        duration_s=100.0,
        magnitude=5.0,
        seed=0,
    )
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    for event in events:
        result = injector.inject(event)
        assert result is event, (
            "inject() must return the original object when no fault window is active, "
            "not a copy — returning a copy would allocate unnecessarily in the hot loop"
        )


def test_inject_sets_fault_label_on_all_events_in_window() -> None:
    """
    I'm verifying that fault_label is set on every event in the window, including
    probabilistic faults where the individual event was not mutated. The label is
    ground truth about the fault window, not the per-event mutation — a detector that
    misses a fault-window event is a false negative regardless of whether that event
    happened to keep its normal status.
    """
    events = _make_baseline_events()
    spec = _full_window_spec("error_burst", magnitude=0.5, seed=7)
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    for event in events:
        result = injector.inject(event)
        assert result.fault_label == "error_burst", (
            f"Event at {result.event_time} inside error_burst window has "
            f"fault_label={result.fault_label!r} — expected 'error_burst'"
        )


def test_inject_does_not_mutate_events_for_wrong_stage() -> None:
    """
    I'm testing that a fault targeting stage_b does not touch events from stage_a.
    This is a cross-stage isolation check — without it a misconfigured target_stage_id
    would silently corrupt unrelated stages and produce phantom anomalies.
    """
    events = _make_baseline_events(stage_id="stage_a")
    spec = FaultSpec(
        fault_type="latency_spike",
        target_stage_id="stage_b",  # different stage
        start_offset_s=0.0,
        duration_s=1e9,
        magnitude=10.0,
        seed=0,
    )
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    for event in events:
        result = injector.inject(event)
        assert result is event, (
            "inject() must not touch events whose stage_id does not match "
            "the fault spec's target_stage_id"
        )


def test_unknown_fault_type_raises_value_error() -> None:
    """
    I'm testing that an unrecognised fault_type raises ValueError rather than silently
    producing a no-op or partially mutated event. Silent failures here would mean a
    typo in a scenario YAML produces a scenario with no faults, passing all accuracy
    benchmarks against ground-truth labels that were never actually injected.
    """
    events = _make_baseline_events()
    spec = _full_window_spec("not_a_real_fault", magnitude=1.0)
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    with pytest.raises(ValueError, match="Unknown fault_type"):
        injector.inject(events[0])


# ---------------------------------------------------------------------------
# Statistical signatures — one test per fault type
# ---------------------------------------------------------------------------


def test_latency_spike_raises_p99() -> None:
    """
    I'm asserting p99 rather than mean because the latency_spike fault type is
    designed to move the tail of the latency distribution, which is what CUSUM and
    EWMA detectors will observe. At magnitude=5.0 with ±20% jitter, the minimum
    applied multiplier is 4.0, so injected p99 must be at least 3× baseline p99.
    Using 3× instead of 4× gives headroom for the tail's natural variance while still
    requiring a meaningful shift that no noise-driven fluctuation would produce.
    """
    events = _make_baseline_events()
    baseline_p99 = float(np.percentile([e.latency_ms for e in events], 99))

    spec = _full_window_spec("latency_spike", magnitude=5.0, seed=1)
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    injected = [injector.inject(e) for e in events]
    injected_p99 = float(np.percentile([e.latency_ms for e in injected], 99))

    assert injected_p99 >= 3.0 * baseline_p99, (
        f"latency_spike p99 {injected_p99:.1f}ms is less than 3× baseline "
        f"p99 {baseline_p99:.1f}ms at magnitude=5.0"
    )


def test_dropped_connection_produces_expected_error_rate() -> None:
    """
    I'm testing both the error rate and the zero-payload invariant because
    dropped_connection has two observable signatures: the rate of error events and the
    fact that dropped events carry no data. Testing only the rate would miss an
    implementation bug where status="error" is set but row_count and payload_bytes
    are preserved, which produces a false throughput reading in downstream metrics.
    """
    events = _make_baseline_events()
    drop_probability = 0.9
    spec = _full_window_spec("dropped_connection", magnitude=drop_probability, seed=2)
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    injected = [injector.inject(e) for e in events]
    error_events = [e for e in injected if e.status == "error"]
    observed_error_rate = len(error_events) / len(injected)

    # At n=1000 and p=0.9 the std dev is ~0.009; a ±10% absolute tolerance gives
    # roughly 10 sigma of headroom before a correct implementation fails.
    assert observed_error_rate >= 0.80, (
        f"dropped_connection error rate {observed_error_rate:.2%} is below 0.80 "
        f"at magnitude={drop_probability}"
    )
    for e in error_events:
        assert e.row_count == 0 and e.payload_bytes == 0, (
            f"Dropped event must have row_count=0 and payload_bytes=0, "
            f"got row_count={e.row_count}, payload_bytes={e.payload_bytes}"
        )


def test_schema_drift_reduces_mean_row_count() -> None:
    """
    I'm computing the reduction relative to the baseline mean rather than asserting
    an absolute value because the baseline row_count is drawn from a Poisson(1000)
    distribution — its mean is ~1000 but varies with the seed. Using a relative
    threshold makes the test seed-independent. At magnitude=0.6, surviving_fraction
    is 0.4, so the injected mean must be ≤ 50% of baseline (allowing some tolerance
    for integer truncation and the seed-driven baseline variance).
    """
    events = _make_baseline_events()
    baseline_mean_rows = float(np.mean([e.row_count for e in events]))

    spec = _full_window_spec("schema_drift", magnitude=0.6, seed=3)
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    injected = [injector.inject(e) for e in events]
    injected_mean_rows = float(np.mean([e.row_count for e in injected]))

    assert injected_mean_rows <= 0.50 * baseline_mean_rows, (
        f"schema_drift mean row_count {injected_mean_rows:.1f} exceeds 50% of "
        f"baseline mean {baseline_mean_rows:.1f} at magnitude=0.6"
    )
    # All events must carry status="schema_error" — the schema is malformed at the
    # stage level, not randomly per-event.
    for e in injected:
        assert e.status == "schema_error", (
            f"schema_drift event has status={e.status!r} — expected 'schema_error'"
        )


def test_partition_skew_inflates_mean_payload() -> None:
    """
    I'm testing payload_bytes inflation because that is the defining observable of
    partition skew — one partition receives a disproportionate share of the data volume.
    At magnitude=4.0 the expected mean payload is 4× baseline; asserting ≥ 3.5×
    tolerates the integer truncation of payload_bytes without masking implementation
    bugs where magnitude is applied to the wrong field or applied as addition rather
    than multiplication.
    """
    events = _make_baseline_events()
    baseline_mean_payload = float(np.mean([e.payload_bytes for e in events]))

    spec = _full_window_spec("partition_skew", magnitude=4.0, seed=4)
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    injected = [injector.inject(e) for e in events]
    injected_mean_payload = float(np.mean([e.payload_bytes for e in injected]))

    assert injected_mean_payload >= 3.5 * baseline_mean_payload, (
        f"partition_skew mean payload {injected_mean_payload:.0f} bytes is less than "
        f"3.5× baseline mean {baseline_mean_payload:.0f} bytes at magnitude=4.0"
    )


def test_throughput_collapse_reduces_mean_row_count() -> None:
    """
    I'm asserting that mean row_count falls to ≤ 20% of baseline at magnitude=8.0
    (theoretical reduction is 1/8 = 12.5%). The 20% ceiling gives headroom for the
    floor-at-1 rule applied to events that would otherwise reach row_count=0, which
    raises the empirical mean slightly above the theoretical 1/magnitude fraction when
    the baseline row distribution has a long left tail.
    """
    events = _make_baseline_events()
    baseline_mean_rows = float(np.mean([e.row_count for e in events]))

    spec = _full_window_spec("throughput_collapse", magnitude=8.0, seed=5)
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    injected = [injector.inject(e) for e in events]
    injected_mean_rows = float(np.mean([e.row_count for e in injected]))

    assert injected_mean_rows <= 0.20 * baseline_mean_rows, (
        f"throughput_collapse mean row_count {injected_mean_rows:.1f} exceeds 20% of "
        f"baseline mean {baseline_mean_rows:.1f} at magnitude=8.0"
    )


def test_error_burst_produces_expected_error_rate_and_preserves_data() -> None:
    """
    I'm testing two invariants together because error_burst's signature is specifically
    the combination of a high error rate WITH preserved data volumes. Testing only the
    error rate would not distinguish error_burst from dropped_connection. The causal
    engine uses precisely this distinction to differentiate processing failures from
    network failures when scoring root-cause candidates.
    """
    events = _make_baseline_events()
    error_rate = 0.8
    spec = _full_window_spec("error_burst", magnitude=error_rate, seed=6)
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    injected = [injector.inject(e) for e in events]
    error_events = [e for e in injected if e.status == "error"]
    observed_error_rate = len(error_events) / len(injected)

    assert observed_error_rate >= 0.70, (
        f"error_burst error rate {observed_error_rate:.2%} is below 0.70 "
        f"at magnitude={error_rate}"
    )
    # Data preservation check — error_burst events must not zero out row or payload.
    for orig, result in zip(events, injected):
        if result.status == "error":
            assert result.row_count == orig.row_count, (
                "error_burst must preserve row_count — this is a processing failure, "
                "not a connection drop"
            )
            assert result.payload_bytes == orig.payload_bytes, (
                "error_burst must preserve payload_bytes — data was received, "
                "processing failed after receipt"
            )


# ---------------------------------------------------------------------------
# FaultSchedule active-window boundary behaviour
# ---------------------------------------------------------------------------


def test_event_at_window_boundary_is_included() -> None:
    """
    I'm testing the inclusive boundary condition (start_offset_s <= offset <= end)
    because an off-by-one on either boundary would silently exclude the first or last
    event from the fault window and bias ground-truth label counts downward in
    benchmark runs.
    """
    # Create one event whose offset lands exactly at the fault window start.
    event = PipelineEvent(
        stage_id="stage_a",
        event_time=SIM_START,  # offset = 0.0 exactly
        latency_ms=33.0,
        row_count=1000,
        payload_bytes=4096,
        status="success",
        fault_label=None,
    )
    spec = FaultSpec(
        fault_type="latency_spike",
        target_stage_id="stage_a",
        start_offset_s=0.0,
        duration_s=60.0,
        magnitude=5.0,
        seed=0,
    )
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[spec])
    injector = FaultInjector(schedule)

    result = injector.inject(event)
    assert result.fault_label == "latency_spike", (
        "Event at exactly start_offset_s=0.0 must be inside the fault window"
    )
    assert result.latency_ms > event.latency_ms, (
        "Latency must be elevated for an event at the fault window boundary"
    )
