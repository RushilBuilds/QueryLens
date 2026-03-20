"""initial schema — pipeline_metrics partitioned table, anomaly_events and fault_localizations stubs

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # I'm using raw SQL for pipeline_metrics rather than op.create_table() because
    # SQLAlchemy's DDL compiler does not emit the PARTITION BY clause when generating
    # CREATE TABLE statements — it only supports it as a table-level storage parameter
    # in newer versions, and Alembic's op.create_table() wraps the compiler directly.
    # Writing the DDL explicitly gives us full control over the partition declaration
    # and the child partition CREATE TABLE statements, which must reference the parent
    # and specify their own PARTITION OF ... FOR VALUES FROM/TO clauses.
    op.execute("""
        CREATE TABLE pipeline_metrics (
            id          BIGINT GENERATED ALWAYS AS IDENTITY,
            stage_id    VARCHAR(64)                  NOT NULL,
            event_time  TIMESTAMP WITH TIME ZONE     NOT NULL,
            latency_ms  DOUBLE PRECISION             NOT NULL,
            row_count   INTEGER                      NOT NULL,
            payload_bytes BIGINT                     NOT NULL,
            status      VARCHAR(32)                  NOT NULL,
            fault_label VARCHAR(64),
            trace_id    VARCHAR(32),
            PRIMARY KEY (id, event_time)
        ) PARTITION BY RANGE (event_time)
    """)

    # I'm pre-creating 12 monthly partitions for 2024 because the scenario fixtures
    # use 2024-01-01 as their simulation start date. Without at least the 2024-01
    # partition, any INSERT from the smoke test or the scenario runner would land in
    # the DEFAULT partition — which is fine functionally but would hide missing-partition
    # bugs until the first production deployment. Pre-creating known partitions makes
    # the schema self-documenting about the expected data range.
    months_2024 = [
        ("2024_01", "2024-01-01", "2024-02-01"),
        ("2024_02", "2024-02-01", "2024-03-01"),
        ("2024_03", "2024-03-01", "2024-04-01"),
        ("2024_04", "2024-04-01", "2024-05-01"),
        ("2024_05", "2024-05-01", "2024-06-01"),
        ("2024_06", "2024-06-01", "2024-07-01"),
        ("2024_07", "2024-07-01", "2024-08-01"),
        ("2024_08", "2024-08-01", "2024-09-01"),
        ("2024_09", "2024-09-01", "2024-10-01"),
        ("2024_10", "2024-10-01", "2024-11-01"),
        ("2024_11", "2024-11-01", "2024-12-01"),
        ("2024_12", "2024-12-01", "2025-01-01"),
    ]
    for suffix, start, end in months_2024:
        op.execute(f"""
            CREATE TABLE pipeline_metrics_{suffix}
            PARTITION OF pipeline_metrics
            FOR VALUES FROM ('{start}') TO ('{end}')
        """)

    # DEFAULT partition catches any event_time outside the 2024 monthly range.
    # I'm including this from day one rather than waiting for an out-of-range insert
    # to fail — failing inserts on partition misses would surface as data loss in the
    # ingestion worker, which is much harder to diagnose than a partition count mismatch.
    op.execute("""
        CREATE TABLE pipeline_metrics_default
        PARTITION OF pipeline_metrics DEFAULT
    """)

    # Per-stage time-range scans are the dominant query pattern for the detection layer
    # (sliding window aggregator fetches the last N seconds of metrics for a given stage).
    # A (stage_id, event_time) index on the parent table is automatically propagated
    # to each child partition by PostgreSQL's partitioned index infrastructure.
    op.execute("""
        CREATE INDEX ix_pipeline_metrics_stage_event
        ON pipeline_metrics (stage_id, event_time)
    """)

    # Sparse index on fault_label to support quick ground-truth recall queries in the
    # detection benchmark without paying index maintenance overhead on the majority of
    # rows where fault_label IS NULL.
    op.execute("""
        CREATE INDEX ix_pipeline_metrics_fault_label
        ON pipeline_metrics (fault_label)
        WHERE fault_label IS NOT NULL
    """)

    # Stub tables for AnomalyEvent and FaultLocalization — created now so that future
    # phases can add columns via ALTER TABLE rather than CREATE TABLE, preserving a
    # clean migration history. I'm using op.create_table() here because these are
    # ordinary heap tables with no partitioning requirements.
    op.create_table(
        "anomaly_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("stage_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_anomaly_events_stage_id", "anomaly_events", ["stage_id"])

    op.create_table(
        "fault_localizations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("stage_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_fault_localizations_stage_id", "fault_localizations", ["stage_id"]
    )


def downgrade() -> None:
    # I'm dropping indexes before tables to avoid the brief period where the index
    # exists on a non-existent table, which would cause a schema validation error if
    # alembic inspect runs between the DROP TABLE and DROP INDEX calls.
    op.drop_index("ix_fault_localizations_stage_id", table_name="fault_localizations")
    op.drop_table("fault_localizations")

    op.drop_index("ix_anomaly_events_stage_id", table_name="anomaly_events")
    op.drop_table("anomaly_events")

    # Dropping the parent table cascades to all child partitions and partition-local
    # indexes automatically — no need to drop each monthly partition individually.
    op.execute("DROP TABLE IF EXISTS pipeline_metrics CASCADE")
