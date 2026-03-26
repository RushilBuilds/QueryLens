import dataclasses
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from simulator.engine import ScenarioConfig, SimulationClock, SimulatorEngine
from simulator.fault_injection import FaultInjector, FaultSchedule, FaultSpec
from simulator.topology import PipelineStage, PipelineTopologyGraph
from simulator.workload import WorkloadProfile

SCENARIO_PATH = Path(__file__).parent.parent / "config" / "scenario_example.yaml"
SIM_START = datetime(2024, 1, 1, 0, 0, 0)


def _minimal_topology() -> PipelineTopologyGraph:
    """
    Two-stage topology rather than the full five-stage fixture: enough to verify
    cross-stage behaviour (separate RNG streams, fault targeting) without event
    volumes that slow assertions.
    """
    return PipelineTopologyGraph([
        PipelineStage(stage_id="src", stage_type="source", upstream_ids=[], propagation_delay_ms=0.0),
        PipelineStage(stage_id="snk", stage_type="sink", upstream_ids=["src"], propagation_delay_ms=20.0),
    ])


def _minimal_workload() -> WorkloadProfile:
    return WorkloadProfile(
        arrival_rate_lambda=5.0,
        payload_mean_bytes=2048.0,
        payload_std_bytes=512.0,
        max_concurrency=4,
    )


def _empty_injector(sim_start: datetime) -> FaultInjector:
    return FaultInjector(FaultSchedule(simulation_start=sim_start, fault_specs=[]))


# ---------------------------------------------------------------------------
# SimulationClock
# ---------------------------------------------------------------------------


def test_clock_current_time_starts_at_simulation_start() -> None:
    """
    Initial state tested separately from advance() so a tick_count=1 initialisation
    bug fails here rather than producing an off-by-one that only surfaces in
    timing-sensitive detection tests.
    """
    clock = SimulationClock(start_time=SIM_START, tick_interval_ms=1000.0)
    assert clock.current_time == SIM_START
    assert clock.tick_count == 0
    assert clock.elapsed_s == 0.0


def test_clock_advances_by_tick_interval() -> None:
    """
    Exact timedelta equality rather than float comparison guards against the
    floating-point drift that motivated the multiplicative timestamp design.
    The drift test below catches cumulative-addition regressions at scale.
    """
    clock = SimulationClock(start_time=SIM_START, tick_interval_ms=250.0)
    clock.advance()
    clock.advance()

    expected = SIM_START + timedelta(milliseconds=500.0)
    assert clock.current_time == expected
    assert clock.tick_count == 2
    assert clock.elapsed_s == pytest.approx(0.5)


def test_clock_no_drift_over_many_ticks() -> None:
    """
    100,000 ticks at 1ms each exposes floating-point drift from cumulative timedelta
    addition. The multiplicative implementation must land within 1 microsecond of
    the 100-second mark; cumulative addition typically drifts 10-50 microseconds.
    """
    clock = SimulationClock(start_time=SIM_START, tick_interval_ms=1.0)
    for _ in range(100_000):
        clock.advance()

    expected = SIM_START + timedelta(seconds=100)
    drift_us = abs((clock.current_time - expected).total_seconds()) * 1e6
    assert drift_us < 1.0, (
        f"Clock drifted {drift_us:.2f} microseconds after 100,000 ticks — "
        "current_time must be computed multiplicatively, not by cumulative addition"
    )


# ---------------------------------------------------------------------------
# SimulatorEngine
# ---------------------------------------------------------------------------


def test_engine_yields_events_for_all_stages() -> None:
    """
    Both stages must contribute events. A bug generating events only for the first
    stage in topology.all_stages would produce the right total count from a single
    stage and pass a count-only assertion.
    """
    topology = _minimal_topology()
    clock = SimulationClock(start_time=SIM_START, tick_interval_ms=1000.0)
    engine = SimulatorEngine(
        clock=clock,
        topology=topology,
        workload_profile=_minimal_workload(),
        fault_injector=_empty_injector(SIM_START),
        rng_seed=42,
    )

    events = list(engine.run(n_ticks=20))
    stage_ids = {e.stage_id for e in events}

    assert "src" in stage_ids, "Source stage must contribute events"
    assert "snk" in stage_ids, "Sink stage must contribute events"
    assert len(events) > 0


def test_engine_events_are_ordered_by_event_time() -> None:
    """
    Both the ingestion and detection layers assume chronological order. An unsorted
    stream would corrupt sliding window aggregator state in a way that is very
    difficult to diagnose.
    """
    topology = _minimal_topology()
    clock = SimulationClock(start_time=SIM_START, tick_interval_ms=1000.0)
    engine = SimulatorEngine(
        clock=clock,
        topology=topology,
        workload_profile=_minimal_workload(),
        fault_injector=_empty_injector(SIM_START),
        rng_seed=0,
    )

    events = list(engine.run(n_ticks=30))
    timestamps = [e.event_time for e in events]
    assert timestamps == sorted(timestamps), (
        "Engine must yield events in ascending event_time order"
    )


def test_engine_events_fall_within_simulation_window() -> None:
    """
    An off-by-one in the pre-generation filter could include events belonging to
    tick n+1, shifting fault-label alignment near the window edge and producing
    incorrect ground-truth counts in the detection benchmark.
    """
    n_ticks = 10
    tick_interval_ms = 1000.0
    topology = _minimal_topology()
    clock = SimulationClock(start_time=SIM_START, tick_interval_ms=tick_interval_ms)
    engine = SimulatorEngine(
        clock=clock,
        topology=topology,
        workload_profile=_minimal_workload(),
        fault_injector=_empty_injector(SIM_START),
        rng_seed=1,
    )

    events = list(engine.run(n_ticks=n_ticks))
    sim_end = SIM_START + timedelta(milliseconds=n_ticks * tick_interval_ms)

    for e in events:
        assert e.event_time >= SIM_START, "Event precedes simulation start"
        assert e.event_time < sim_end, (
            f"Event at {e.event_time} falls outside simulation window ending {sim_end}"
        )


def test_engine_applies_fault_labels_in_active_window() -> None:
    """
    Fault covering ticks 5-10 of a 20-tick simulation. Events inside the window
    must carry fault_label; events outside must not. Verifies SimulatorEngine
    correctly threads FaultInjector through the event stream.
    """
    topology = _minimal_topology()
    clock = SimulationClock(start_time=SIM_START, tick_interval_ms=1000.0)

    fault_spec = FaultSpec(
        fault_type="latency_spike",
        target_stage_id="src",
        start_offset_s=5.0,
        duration_s=5.0,   # active from t=5s to t=10s
        magnitude=3.0,
        seed=0,
    )
    schedule = FaultSchedule(simulation_start=SIM_START, fault_specs=[fault_spec])
    injector = FaultInjector(schedule)

    engine = SimulatorEngine(
        clock=clock,
        topology=topology,
        workload_profile=_minimal_workload(),
        fault_injector=injector,
        rng_seed=5,
    )

    events = list(engine.run(n_ticks=20))
    src_events = [e for e in events if e.stage_id == "src"]

    fault_window_start = SIM_START + timedelta(seconds=5)
    fault_window_end = SIM_START + timedelta(seconds=10)

    for e in src_events:
        in_window = fault_window_start <= e.event_time <= fault_window_end
        if in_window:
            assert e.fault_label == "latency_spike", (
                f"src event at {e.event_time} inside fault window has "
                f"fault_label={e.fault_label!r}"
            )
        else:
            assert e.fault_label is None, (
                f"src event at {e.event_time} outside fault window has "
                f"fault_label={e.fault_label!r} — should be None"
            )


# ---------------------------------------------------------------------------
# ScenarioConfig — YAML loading and reproducibility
# ---------------------------------------------------------------------------


def test_scenario_config_loads_from_yaml() -> None:
    """
    Tested against the actual scenario_example.yaml on disk, not an inline string:
    validates that the file is well-formed and that topology path resolution works
    from the config directory.
    """
    config = ScenarioConfig.load(SCENARIO_PATH)
    assert config.name == "latency_spike_source_postgres"


def test_identical_seeds_produce_identical_event_streams() -> None:
    """
    build_engine() called twice rather than run() twice: run() continues from
    the advanced clock and consumed RNG state, not restart from the beginning.

    dataclasses.asdict() used for comparison so new PipelineEvent fields are
    automatically included without a test update.
    """
    config = ScenarioConfig.load(SCENARIO_PATH)

    engine_a = config.build_engine()
    engine_b = config.build_engine()

    events_a = list(engine_a.run(n_ticks=50))
    events_b = list(engine_b.run(n_ticks=50))

    assert len(events_a) == len(events_b), (
        f"Identical seeds produced {len(events_a)} vs {len(events_b)} events"
    )
    for i, (a, b) in enumerate(zip(events_a, events_b)):
        assert dataclasses.asdict(a) == dataclasses.asdict(b), (
            f"Event {i} differs between identical-seed runs: {a} vs {b}"
        )


def test_different_seeds_produce_different_event_streams() -> None:
    """
    An implementation that ignores rng_seed and uses a fixed seed would pass
    the identical-seed test above while silently making all scenarios produce
    the same data. This test confirms the seed actually drives the RNG.
    """
    config_a = ScenarioConfig.load(SCENARIO_PATH)

    # Construct a second config with a different seed by directly instantiating
    # rather than modifying the YAML file.
    raw_config_a = config_a
    config_b = ScenarioConfig(
        name=config_a.name,
        rng_seed=config_a._rng_seed + 1,
        simulation_start=config_a._simulation_start,
        tick_interval_ms=config_a._tick_interval_ms,
        topology=config_a._topology,
        workload_profile=config_a._workload_profile,
        fault_schedule=config_a._fault_schedule,
    )

    events_a = list(config_a.build_engine().run(n_ticks=20))
    events_b = list(config_b.build_engine().run(n_ticks=20))

    # It would be astronomically unlikely for all events to match with different seeds.
    latencies_a = [e.latency_ms for e in events_a]
    latencies_b = [e.latency_ms for e in events_b]
    assert latencies_a != latencies_b, (
        "Different rng_seeds produced identical event streams — seed is not being used"
    )
