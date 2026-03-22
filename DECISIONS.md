# Architecture Decisions

This document logs every major architectural decision made during the development of QueryLens, including alternatives considered and reasoning.

---

## ADR-020: RingBuffer backed by numpy arrays rather than collections.deque
**Date:** 2026-03-22
**Decision:** Use two parallel `np.float64` arrays (timestamps + values) as the ring buffer backing store
**Alternatives considered:** `collections.deque(maxlen=capacity)` of `(timestamp, value)` tuples
**Reasoning:** `window_values()` feeds directly into `np.percentile()`. With a deque, every `compute()` call would require converting the deque to a list and then to a numpy array — at 1,000 samples that is ~40µs of allocation overhead per tick, multiplied by the number of stages and metrics. The numpy array lets us index and mask in-place with no allocation on the hot path.

## ADR-021: WindowStats returns None for all stats when is_stable is False
**Date:** 2026-03-22
**Decision:** Set all stat fields to `None` when `sample_count < min_sample_count`
**Alternatives considered:** Return 0.0; return `float("nan")`
**Reasoning:** `0.0` looks like a real latency measurement (0ms latency) and would trigger CUSUM/EWMA false alerts during the startup warmup period before the window has enough samples. `float("nan")` propagates silently through numpy arithmetic, meaning a detector that forgets to check `is_stable` would produce `nan` anomaly scores with no error. `None` raises `TypeError` immediately if the caller tries to use the value without checking `is_stable`, which surfaces the bug at the point of the mistake rather than downstream.

## ADR-017: Prometheus metrics as module-level singletons in observability.py
**Date:** 2026-03-22
**Decision:** Define all `prometheus_client` Counter/Histogram/Gauge objects at module level rather than as instance variables on MetricConsumer or IngestionWorker
**Alternatives considered:** Per-instance metrics objects; passing a custom `CollectorRegistry` to each class constructor
**Reasoning:** `prometheus_client` raises `ValueError: Duplicated timeseries` if two collectors with the same name are registered to the same registry. Instance variables would trigger this on the second construction of MetricConsumer or IngestionWorker in the same process — which happens in every test that creates a fresh consumer. Module-level singletons are registered exactly once at import time. A custom per-instance registry was rejected because it would require threading the registry through the WSGI app, breaking the default scrape endpoint.

## ADR-018: MetricsServer uses wsgiref.simple_server instead of prometheus_client.start_http_server
**Date:** 2026-03-22
**Decision:** Implement MetricsServer using `wsgiref.simple_server.make_server` with `prometheus_client.make_wsgi_app()`
**Alternatives considered:** `prometheus_client.start_http_server(port)` — one-liner but return type changed across versions
**Reasoning:** `start_http_server` returned `None` in prometheus_client ≤0.16 and a `(server, thread)` tuple in ≥0.17. Calling `shutdown()` portably would require a version check. `wsgiref` is stdlib, stable, and gives clean lifecycle control: `serve_forever()` in a daemon thread, `shutdown()` synchronously. `port=0` lets the OS assign a free port, which prevents port collisions in parallel test runs.

## ADR-019: DLQ_EVENTS counter carries no stage_id label
**Date:** 2026-03-22
**Decision:** `ingestion_dlq_events_total` is a flat counter with no labels
**Alternatives considered:** Label by `stage_id="unknown"` for all DLQ events
**Reasoning:** The stage_id is extracted during deserialization. A message lands in the DLQ precisely because deserialization failed — we do not have a stage_id to label with. Emitting `stage_id="unknown"` would look meaningful in Grafana but carry no actual signal. A flat counter is honest: it says "something was unreadable" without implying we know which stage it came from.

## ADR-014: Manual Kafka offset commit — commit only after successful PostgreSQL write
**Date:** 2026-03-20
**Decision:** Disable `enable.auto.commit` and commit offsets synchronously via `consumer.commit(asynchronous=False)` only after `session.commit()` succeeds
**Alternatives considered:** `enable.auto.commit=true` (commit on timer); async manual commit (commit without waiting for broker ack)
**Reasoning:** Auto-commit advances the offset on a wall-clock interval regardless of write success. A PostgreSQL failure between auto-commits silently drops the unwritten batch — on restart the consumer resumes from the committed offset, skipping those records permanently. Manual synchronous commit after write gives us at-least-once delivery: on crash-restart, the uncommitted batch replays from the last committed Kafka offset. Duplicate rows from replay are tolerable (GENERATED ALWAYS AS IDENTITY assigns new IDs; aggregates are robust to occasional duplicates) whereas silent data loss is not.

## ADR-015: Dual-trigger flush (batch_size OR flush_interval_s) in IngestionWorker
**Date:** 2026-03-20
**Decision:** Flush when `len(pending) >= batch_size` OR when `flush_interval_s` seconds have elapsed with at least one pending record
**Alternatives considered:** Batch-size-only flush; interval-only flush; per-record commit
**Reasoning:** Batch-size-only flush strands records on a quiet topic indefinitely — if the topic receives 499 events and goes quiet, no flush fires until the 500th event arrives, adding unbounded latency to the detection layer. Interval-only flush creates a large crash-replay window during a burst (500ms × 5,000 events/sec = 2,500 events replayed on crash). Per-record commit eliminates batching entirely and is 500× slower than bulk INSERT. The dual trigger bounds latency to `flush_interval_s` and bounds replay volume to `batch_size × records/s × flush_interval_s`.

## ADR-016: DLQ offset advances alongside valid offsets — malformed messages are not replayed
**Date:** 2026-03-20
**Decision:** Store and commit the offset for a malformed message after routing it to the DLQ, treating it identically to a valid message for offset advancement purposes
**Alternatives considered:** Skip offset commit for malformed messages (replay on restart); route to DLQ and crash (force operator intervention)
**Reasoning:** Not advancing the offset for a malformed message would cause infinite replay: on every restart, the consumer re-reads the same malformed record, re-routes it to DLQ, and never makes progress past that offset. The DLQ already has the record with full error context; replaying it adds duplicate DLQ entries without recovering anything. Crashing on malformed messages would halt the ingestion pipeline entirely for a single bad producer record, which is too brittle for a production system.

## ADR-011: JSON wire format with schema_version over Avro or MessagePack
**Date:** 2026-03-20
**Decision:** Encode `PipelineEvent` as UTF-8 JSON with a `schema_version` integer field
**Alternatives considered:** Avro with Confluent Schema Registry; MessagePack binary encoding
**Reasoning:** At single-digit thousands of events per second, JSON throughput is not a bottleneck. Avro requires a schema registry — an additional infrastructure dependency that adds operational complexity and a potential single point of failure for the ingestion path. MessagePack saves ~30% payload size but produces non-human-readable DLQ messages, which is the wrong trade-off when DLQ debuggability is more operationally valuable than the bandwidth savings. `schema_version` enables consumer-side compatibility checks without topic proliferation.

## ADR-012: confluent_kafka idempotent producer with acks=all and retries=5
**Date:** 2026-03-20
**Decision:** Configure `RedpandaProducer` with `enable.idempotence=True`, `acks=all`, `retries=5`
**Alternatives considered:** `acks=1` (leader-only acknowledgement); `acks=0` (fire-and-forget); no retries
**Reasoning:** `acks=all` ensures all in-sync replicas acknowledge before the producer considers a message delivered — a leader crash after `acks=1` would lose any messages not yet replicated to followers. `enable.idempotence` with retries guarantees that a retry caused by a lost ack does not produce a duplicate record in the partition. The cost is ~10% throughput reduction vs `acks=1`; the gain is a delivery guarantee the downstream detection layer can rely on.

## ADR-013: stage_id as Kafka message key for per-stage partition affinity
**Date:** 2026-03-20
**Decision:** Encode `event.stage_id` as the Kafka message key in `RedpandaProducer.publish()`
**Alternatives considered:** No key (round-robin partition assignment); random UUID key; hash of (stage_id, event_time)
**Reasoning:** The sliding window aggregator (M9) reads metrics per stage and needs them in time order within each stage. With round-robin assignment, events from the same stage land on different partitions and arrive out of order at the consumer — requiring an additional sort step on every aggregation window. Keying by stage_id routes all events for a stage to the same partition, preserving order without any consumer-side sorting. A composite key (stage_id, event_time) would further sub-partition within a stage but offers no ordering benefit since partition assignment is hash-based, not range-based.

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
