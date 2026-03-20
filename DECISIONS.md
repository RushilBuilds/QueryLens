# Architecture Decisions

This document logs every major architectural decision made during the development of QueryLens, including alternatives considered and reasoning.

---

## ADR-008: pipeline_metrics uses PARTITION BY RANGE on event_time with pre-created monthly child tables
**Date:** 2026-03-20
**Decision:** Declare `pipeline_metrics` as a range-partitioned table with 12 monthly partitions for 2024 plus a DEFAULT partition, created via raw SQL in the Alembic migration
**Alternatives considered:** Single heap table with a `(stage_id, event_time)` index; hash partitioning by `stage_id`; partition-by-hash on `id`
**Reasoning:** The dominant query pattern for the sliding window aggregator is `WHERE stage_id = ? AND event_time > NOW() - INTERVAL '5 minutes'`. Range partitioning on `event_time` lets PostgreSQL skip all partitions outside the query window (partition pruning), which on a 90-day dataset means scanning ~3 partitions instead of the full table. Hash partitioning on `stage_id` would have helped for stage-scoped full scans but offers no time-range pruning. A single heap table with an index works at small scale but degrades as the table grows past ~50M rows — building partitioning in from the start avoids a zero-downtime repartitioning operation later.

## ADR-009: Raw SQL for CREATE TABLE PARTITION BY RANGE rather than op.create_table()
**Date:** 2026-03-20
**Decision:** Use `op.execute()` with hand-written DDL for the partitioned table, while using `op.create_table()` for the stub tables
**Alternatives considered:** `op.create_table()` with SQLAlchemy table args for partitioning; a custom Alembic DDL construct
**Reasoning:** SQLAlchemy's DDL compiler does not emit `PARTITION BY RANGE` when generating `CREATE TABLE` — there is no first-class `partition_by` argument in `op.create_table()`. A custom DDL construct would be correct but would add ~50 lines of SQLAlchemy extension code that provides no value beyond making the migration look idiomatic. Raw SQL is more readable, makes the DDL intent explicit, and is the approach documented in SQLAlchemy's own partitioning notes.

## ADR-010: testcontainers over mocking or SQLite for migration smoke test
**Date:** 2026-03-20
**Decision:** Run the migration smoke test against a real `postgres:16-alpine` container via testcontainers-python
**Alternatives considered:** SQLite in-memory database; mocking the SQLAlchemy session; a persistent local Postgres instance
**Reasoning:** SQLite does not support `PARTITION BY RANGE`, `GENERATED ALWAYS AS IDENTITY`, or partial indexes with `WHERE` clauses. An SQLite fixture would accept the ORM code and fail only on deployment — the worst possible time. A persistent local Postgres instance creates CI portability problems (different engineers have different local Postgres versions). testcontainers spins up the exact same `postgres:16-alpine` image used in docker-compose.yml, ensuring the test exercises the same engine version in both local and CI environments.

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
