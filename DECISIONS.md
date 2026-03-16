# Architecture Decisions

This document logs every major architectural decision made during the development of QueryLens, including alternatives considered and reasoning.

---

## ADR-001: Redpanda over Apache Kafka
**Date:** 2026-03-13
**Decision:** Use Redpanda as the streaming broker instead of Apache Kafka
**Alternatives considered:** Apache Kafka, AWS Kinesis, RabbitMQ
**Reasoning:** Redpanda is Kafka-API compatible so all consumer/producer code is identical, but eliminates ZooKeeper dependency which significantly reduces local development complexity. For a single-node development environment this removes an entire infrastructure component without sacrificing any functionality we need.

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
