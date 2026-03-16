# Architecture Decisions

This document logs every major architectural decision made during the development of QueryLens, including alternatives considered and reasoning.

---

## ADR-001: Redpanda over Apache Kafka
**Date:** 2026-03-13
**Decision:** Use Redpanda as the streaming broker instead of Apache Kafka
**Alternatives considered:** Apache Kafka, AWS Kinesis, RabbitMQ
**Reasoning:** Redpanda is Kafka-API compatible so all consumer/producer code is identical, but eliminates ZooKeeper dependency which significantly reduces local development complexity. For a single-node development environment this removes an entire infrastructure component without sacrificing any functionality we need.

## ADR-005: SimulationClock uses multiplicative timestamp computation to prevent drift
**Date:** 2026-03-16
**Decision:** Compute `current_time` as `start_time + tick_count × tick_interval_ms` rather than accumulating via repeated timedelta addition
**Alternatives considered:** Cumulative `current_time += timedelta(milliseconds=tick_interval_ms)` on each advance() call
**Reasoning:** IEEE 754 floating-point addition is not associative — after 100,000 additions of 1.0ms, the accumulated error is 10–50 microseconds depending on platform. Multiplying from a fixed integer tick_count keeps every timestamp exact. This matters for fault window boundary checks: a drifted clock could include or exclude events at the boundary of a FaultSpec's start/end offsets, corrupting ground-truth label counts.

## ADR-006: SimulatorEngine pre-generates all events before yielding by tick
**Date:** 2026-03-16
**Decision:** Generate all events for the full simulation window upfront, sort globally, then yield tick by tick
**Alternatives considered:** Generate events per tick by calling PoissonEventGenerator.generate() once per tick per stage
**Reasoning:** Per-tick generation would restart the Poisson inter-arrival chain at each tick boundary, introducing artificial clustering at tick edges (all events generated within a tick start from time=tick_start). Pre-generating and filtering to the window preserves the continuous Poisson inter-arrival statistics that CUSUM and EWMA detectors depend on for accurate baseline fitting.

## ADR-007: Stage RNGs derived via SeedSequence.spawn() rather than counter-incremented seeds
**Date:** 2026-03-16
**Decision:** Derive per-stage RNGs using `np.random.SeedSequence.spawn(n_stages)` from the master seed
**Alternatives considered:** `np.random.default_rng(master_seed + stage_index)` — offset seeds per stage
**Reasoning:** Offset seeds are not statistically independent — seeds that differ by small integers can produce correlated sequences for certain numpy RNG algorithms. `SeedSequence.spawn()` is numpy's documented API for producing statistically independent child generators. Additionally, offset seeds would change every stage's RNG when a stage is inserted at a lower index, whereas spawn() produces child sequences that are stable relative to the master seed regardless of how many children are spawned.

## ADR-003: Dimensionless magnitude parameter across all fault types
**Date:** 2026-03-16
**Decision:** Use a single `magnitude` float on `FaultSpec` whose interpretation is fault-type-specific rather than defining a typed parameter class per fault type
**Alternatives considered:** `LatencySpikeFaultSpec(multiplier=5.0)`, `DroppedConnectionFaultSpec(drop_probability=0.9)`, etc.
**Reasoning:** A typed parameter class per fault type would require six separate dataclasses and a union type at the `FaultSchedule` level, complicating YAML serialisation and ScenarioConfig construction without providing meaningful type safety — the fault types are validated at runtime anyway. A single dimensionless `magnitude` with contract documented in the injector keeps `FaultSpec` flat, YAML-round-trippable, and easy to extend with new fault types.

## ADR-004: fault_label set on all events in the fault window, not only mutated events
**Date:** 2026-03-16
**Decision:** `FaultInjector._apply_fault()` sets `fault_label` on every event whose `event_time` falls inside the active fault window, regardless of whether the probabilistic branch chose to mutate that individual event
**Alternatives considered:** Set `fault_label` only on events where `status` was changed or a field was mutated
**Reasoning:** `fault_label` is ground truth about the fault window, not about per-event mutation outcomes. A detector that observes a fault-window event and produces no anomaly signal is a false negative — the fault was active, the detector missed it. If `fault_label` were only set on mutated events, the benchmark would silently count probabilistic non-mutations as true negatives and inflate recall scores.

## ADR-002: PostgreSQL as the pipeline metadata store
**Date:** 2026-03-13
**Decision:** Use PostgreSQL to store pipeline events, anomaly records, and healing actions
**Alternatives considered:** TimescaleDB, ClickHouse, SQLite
**Reasoning:** TimescaleDB was a strong candidate given the time-series nature of pipeline metrics, but PostgreSQL with a well-indexed events table is sufficient for our data volumes and avoids adding another service dependency. Can migrate to TimescaleDB later if query performance degrades.
