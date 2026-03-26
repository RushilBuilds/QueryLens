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
    # Raw SQL used for pipeline_metrics rather than op.create_table(): SQLAlchemy's
    # DDL compiler does not emit the PARTITION BY clause. Explicit DDL gives full
    # control over the partition declaration and child PARTITION OF ... FOR VALUES
    # FROM/TO clauses.
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

    # 12 monthly partitions pre-created for 2024 because scenario fixtures use
    # 2024-01-01 as simulation start. Without them, inserts land in DEFAULT —
    # functionally acceptable but hides missing-partition bugs until production.
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

    # DEFAULT partition catches event_time values outside the 2024 monthly range.
    # Created from day one: failing inserts on partition misses surface as ingestion
    # data loss, which is harder to diagnose than a partition count mismatch.
    op.execute("""
        CREATE TABLE pipeline_metrics_default
        PARTITION OF pipeline_metrics DEFAULT
    """)

    # (stage_id, event_time) index on the parent table: the sliding window aggregator's
    # dominant query is a per-stage time-range scan. PostgreSQL propagates partitioned
    # indexes to child partitions automatically.
    op.execute("""
        CREATE INDEX ix_pipeline_metrics_stage_event
        ON pipeline_metrics (stage_id, event_time)
    """)

    # Sparse partial index on fault_label: supports ground-truth recall queries in
    # the detection benchmark without index maintenance overhead on the majority of
    # rows where fault_label IS NULL.
    op.execute("""
        CREATE INDEX ix_pipeline_metrics_fault_label
        ON pipeline_metrics (fault_label)
        WHERE fault_label IS NOT NULL
    """)

    # Stub tables created now so future phases add columns via ALTER TABLE rather than
    # CREATE TABLE, preserving a clean migration history. op.create_table() is safe
    # here — these are ordinary heap tables with no partitioning requirements.
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
    # Indexes dropped before tables: avoids a window where the index exists on a
    # non-existent table, which would cause a schema validation error if alembic
    # inspect runs between DROP TABLE and DROP INDEX.
    op.drop_index("ix_fault_localizations_stage_id", table_name="fault_localizations")
    op.drop_table("fault_localizations")

    op.drop_index("ix_anomaly_events_stage_id", table_name="anomaly_events")
    op.drop_table("anomaly_events")

    # DROP ... CASCADE on the parent cascades to all child partitions and their
    # partition-local indexes automatically.
    op.execute("DROP TABLE IF EXISTS pipeline_metrics CASCADE")
