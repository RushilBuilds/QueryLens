from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import yaml

from simulator.fault_injection import FaultInjector, FaultSchedule, FaultSpec
from simulator.models import PipelineEvent
from simulator.topology import PipelineTopologyGraph, TopologyLoader
from simulator.workload import PoissonEventGenerator, WorkloadProfile


class SimulationClock:
    """
    I'm computing current_time as start_time + tick_count * tick_interval rather than
    accumulating it via repeated timedelta addition. Repeated floating-point addition
    drifts: after 100,000 ticks at 1ms each, naive accumulation can diverge by tens of
    microseconds depending on the platform's floating-point behaviour. Multiplying from
    a fixed origin keeps every tick's timestamp exactly reproducible across machines.

    The clock intentionally has no relationship to wall time. Any component that needs
    a timestamp must call clock.current_time — calling datetime.utcnow() anywhere in
    the simulator would break reproducibility silently.
    """

    def __init__(self, start_time: datetime, tick_interval_ms: float) -> None:
        self._start_time = start_time
        self._tick_interval_ms = tick_interval_ms
        self._tick_count: int = 0

    @property
    def current_time(self) -> datetime:
        return self._start_time + timedelta(
            milliseconds=self._tick_count * self._tick_interval_ms
        )

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def tick_interval_ms(self) -> float:
        return self._tick_interval_ms

    @property
    def elapsed_s(self) -> float:
        return self._tick_count * self._tick_interval_ms / 1000.0

    def advance(self) -> None:
        self._tick_count += 1


class SimulatorEngine:
    """
    I'm pre-generating all events for the full simulation window upfront rather than
    generating per-tick. The per-tick approach would require calling
    PoissonEventGenerator.generate() once per tick per stage, which restarts the
    Poisson inter-arrival chain at each tick boundary — that breaks the statistical
    properties of the Poisson process by introducing artificial gaps at tick edges.
    Pre-generating and filtering to the window preserves continuous inter-arrival
    statistics across tick boundaries.

    Stage RNGs are derived from the master seed via np.random.SeedSequence.spawn().
    This guarantees that each stage's RNG is statistically independent of every other
    stage's RNG, and that adding or removing a stage from the topology does not change
    the RNG sequence of any other stage — both properties would break if I derived
    stage seeds by incrementing a counter from the master seed.
    """

    def __init__(
        self,
        clock: SimulationClock,
        topology: PipelineTopologyGraph,
        workload_profile: WorkloadProfile,
        fault_injector: FaultInjector,
        rng_seed: int,
    ) -> None:
        self._clock = clock
        self._workload_profile = workload_profile
        self._fault_injector = fault_injector

        # I'm spawning one child SeedSequence per stage rather than sharing a single
        # RNG across stages. Shared state would mean that event generation for one
        # stage consumes draws that would otherwise go to another stage, making the
        # per-stage event streams dependent on stage ordering in the topology.
        stages = topology.all_stages
        seed_seq = np.random.SeedSequence(rng_seed)
        stage_seeds = seed_seq.spawn(len(stages))

        self._stage_generators: Dict[str, Tuple[PoissonEventGenerator, float]] = {
            stage.stage_id: (
                PoissonEventGenerator(
                    profile=workload_profile,
                    stage_id=stage.stage_id,
                    rng=np.random.default_rng(stage_seeds[i]),
                ),
                workload_profile.arrival_rate_lambda,
            )
            for i, stage in enumerate(stages)
        }

    def run(self, n_ticks: int) -> Iterator[PipelineEvent]:
        """
        I'm using a generous overcount of (2 × expected mean + 100) for the number
        of events to pre-generate per stage. By the Chernoff bound, the probability
        that a Poisson(μ) variable exceeds 2μ is at most exp(−μ/3), which for any
        μ > 30 is below 10^-4 and falls off exponentially from there. The +100 covers
        the degenerate case where n_ticks is very small (μ < 1) and the multiplier
        alone would underestimate.

        Events are sorted globally by event_time before injection so the FaultInjector
        sees them in timestamp order. This matters for fault types that track per-event
        state (none currently, but important for multi-fault composition in the future).
        """
        tick_interval_s = self._clock.tick_interval_ms / 1000.0
        sim_duration_s = n_ticks * tick_interval_s
        sim_end = self._clock.current_time + timedelta(seconds=sim_duration_s)
        sim_start = self._clock.current_time

        all_events: List[PipelineEvent] = []
        for stage_id, (generator, arrival_lambda) in self._stage_generators.items():
            mean_count = sim_duration_s * arrival_lambda
            n_generate = int(2 * mean_count) + 100
            stage_events = [
                e
                for e in generator.generate(n_events=n_generate, start_time=sim_start)
                if e.event_time < sim_end
            ]
            all_events.extend(stage_events)

        all_events.sort(key=lambda e: e.event_time)

        injected: List[PipelineEvent] = [
            self._fault_injector.inject(e) for e in all_events
        ]

        event_idx = 0
        for _ in range(n_ticks):
            tick_end = self._clock.current_time + timedelta(
                milliseconds=self._clock.tick_interval_ms
            )
            while event_idx < len(injected) and injected[event_idx].event_time < tick_end:
                yield injected[event_idx]
                event_idx += 1
            self._clock.advance()


class ScenarioConfig:
    """
    I'm making ScenarioConfig a thin loader that validates YAML fields and delegates
    all construction to the domain classes rather than building a parallel config
    object hierarchy. The alternative — a ScenarioConfig dataclass that mirrors every
    field of WorkloadProfile, FaultSpec, etc. — would create a second schema to keep
    in sync with the domain classes and provide no additional value.

    build_engine() returns a freshly constructed SimulatorEngine with a clean RNG
    state each time it is called. This is the reproducibility contract: given the
    same ScenarioConfig, every call to build_engine().run(n_ticks) produces an
    identical event stream. Tests that need two independent runs must call
    build_engine() twice rather than calling run() twice on the same engine.
    """

    def __init__(
        self,
        name: str,
        rng_seed: int,
        simulation_start: datetime,
        tick_interval_ms: float,
        topology: PipelineTopologyGraph,
        workload_profile: WorkloadProfile,
        fault_schedule: FaultSchedule,
    ) -> None:
        self._name = name
        self._rng_seed = rng_seed
        self._simulation_start = simulation_start
        self._tick_interval_ms = tick_interval_ms
        self._topology = topology
        self._workload_profile = workload_profile
        self._fault_schedule = fault_schedule

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def load(cls, path: Path) -> ScenarioConfig:
        """
        I'm resolving the topology path relative to the scenario YAML's parent
        directory rather than the process working directory. Relative-to-cwd paths
        would make the scenario files non-portable — the same YAML would work from
        the repo root but fail from any other working directory. Parent-relative
        paths make the scenario file self-contained.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        scenario_section = raw["scenario"]
        name = scenario_section["name"]
        rng_seed = int(scenario_section["rng_seed"])
        simulation_start = datetime.fromisoformat(scenario_section["simulation_start"])

        tick_interval_ms = float(raw["clock"]["tick_interval_ms"])

        topology_path = path.parent / raw["topology"]["path"]
        topology = TopologyLoader.from_yaml(topology_path)

        wl = raw["workload"]
        workload_profile = WorkloadProfile(
            arrival_rate_lambda=float(wl["arrival_rate_lambda"]),
            payload_mean_bytes=float(wl["payload_mean_bytes"]),
            payload_std_bytes=float(wl["payload_std_bytes"]),
            max_concurrency=int(wl["max_concurrency"]),
        )

        fault_specs: List[FaultSpec] = []
        for entry in raw.get("faults", []):
            fault_specs.append(
                FaultSpec(
                    fault_type=entry["fault_type"],
                    target_stage_id=entry["target_stage_id"],
                    start_offset_s=float(entry["start_offset_s"]),
                    duration_s=float(entry["duration_s"]),
                    magnitude=float(entry["magnitude"]),
                    seed=int(entry["seed"]),
                )
            )

        fault_schedule = FaultSchedule(
            simulation_start=simulation_start,
            fault_specs=fault_specs,
        )

        return cls(
            name=name,
            rng_seed=rng_seed,
            simulation_start=simulation_start,
            tick_interval_ms=tick_interval_ms,
            topology=topology,
            workload_profile=workload_profile,
            fault_schedule=fault_schedule,
        )

    def build_engine(self) -> SimulatorEngine:
        """
        I'm constructing a fresh SimulationClock and FaultInjector on every call so
        that each engine starts with a clean tick_count=0 and unadvanced per-spec
        RNGs. Reusing a clock or injector across engine instances would let the second
        run start from the state left by the first, breaking the reproducibility
        guarantee.
        """
        clock = SimulationClock(
            start_time=self._simulation_start,
            tick_interval_ms=self._tick_interval_ms,
        )
        fault_injector = FaultInjector(self._fault_schedule)
        return SimulatorEngine(
            clock=clock,
            topology=self._topology,
            workload_profile=self._workload_profile,
            fault_injector=fault_injector,
            rng_seed=self._rng_seed,
        )
