"""expand anomaly_events — add detector fields for CUSUM/EWMA output and fault_label alignment

Revision ID: 003
Revises: 002
Create Date: 2026-03-23 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # I'm using server_default on every new NOT NULL column because the table
    # may have rows from earlier test runs (it was created as a stub in 001).
    # The server_default satisfies PostgreSQL's constraint check during ALTER
    # TABLE; new inserts from AnomalyPersister always provide explicit values
    # so the default is never used in practice after migration.
    op.add_column(
        "anomaly_events",
        sa.Column(
            "detector_type",
            sa.String(16),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.add_column(
        "anomaly_events",
        sa.Column(
            "metric",
            sa.String(32),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.add_column(
        "anomaly_events",
        sa.Column(
            "signal",
            sa.String(8),
            nullable=False,
            server_default="upper",
        ),
    )
    op.add_column(
        "anomaly_events",
        sa.Column(
            "detector_value",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
    )
    op.add_column(
        "anomaly_events",
        sa.Column(
            "threshold",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
    )
    op.add_column(
        "anomaly_events",
        sa.Column(
            "z_score",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
    )
    op.add_column(
        "anomaly_events",
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.add_column(
        "anomaly_events",
        sa.Column("fault_label", sa.String(64), nullable=True),
    )
    op.add_column(
        "anomaly_events",
        sa.Column(
            "schema_version",
            sa.SmallInteger(),
            nullable=False,
            server_default="1",
        ),
    )

    # Index on detected_at to support time-range queries from the causal layer
    # without a full-table scan. The detection benchmark (M14) will query
    # anomaly_events filtered by detected_at to compute per-fault-type recall.
    op.create_index(
        "ix_anomaly_events_detected_at",
        "anomaly_events",
        ["detected_at"],
    )

    # Sparse index on fault_label mirrors the pipeline_metrics convention — most
    # anomalies in production have no fault_label (healthy pipeline), so a
    # partial index avoids paying index maintenance on the NULL majority.
    op.create_index(
        "ix_anomaly_events_fault_label",
        "anomaly_events",
        ["fault_label"],
        postgresql_where=sa.text("fault_label IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_anomaly_events_fault_label", table_name="anomaly_events")
    op.drop_index("ix_anomaly_events_detected_at", table_name="anomaly_events")
    op.drop_column("anomaly_events", "schema_version")
    op.drop_column("anomaly_events", "fault_label")
    op.drop_column("anomaly_events", "detected_at")
    op.drop_column("anomaly_events", "z_score")
    op.drop_column("anomaly_events", "threshold")
    op.drop_column("anomaly_events", "detector_value")
    op.drop_column("anomaly_events", "signal")
    op.drop_column("anomaly_events", "metric")
    op.drop_column("anomaly_events", "detector_type")
