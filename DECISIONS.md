# Architecture Decisions

This document logs every major architectural decision made during the development of QueryLens, including alternatives considered and reasoning.

---

## ADR-001: Redpanda over Apache Kafka
**Date:** 2026-03-13
**Decision:** Use Redpanda as the streaming broker instead of Apache Kafka
**Alternatives considered:** Apache Kafka, AWS Kinesis, RabbitMQ
**Reasoning:** Redpanda is Kafka-API compatible so all consumer/producer code is identical, but eliminates ZooKeeper dependency which significantly reduces local development complexity. For a single-node development environment this removes an entire infrastructure component without sacrificing any functionality we need.

## ADR-002: PostgreSQL as the pipeline metadata store
**Date:** 2026-03-13
**Decision:** Use PostgreSQL to store pipeline events, anomaly records, and healing actions
**Alternatives considered:** TimescaleDB, ClickHouse, SQLite
**Reasoning:** TimescaleDB was a strong candidate given the time-series nature of pipeline metrics, but PostgreSQL with a well-indexed events table is sufficient for our data volumes and avoids adding another service dependency. Can migrate to TimescaleDB later if query performance degrades.
