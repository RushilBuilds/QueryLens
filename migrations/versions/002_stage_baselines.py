"""stage_baselines — per-stage, per-hour-of-week fitted baseline means and stds

Revision ID: 002
Revises: 001
Create Date: 2026-03-22 00:00:00.000000

Separate migration rather than the initial schema because the columns depend
on the SeasonalBaselineModel contract finalized in Milestone 10. Including
them in 001 would have required renaming columns once the detection API was
designed, producing a corrective migration that obscures the history.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SMALLINT for hour_of_week (0-167) rather than INTEGER makes the domain
    # constraint self-documenting. A value outside 0-167 indicates a fitter
    # bug, and the narrower type signals to schema browsers that this is a
    # bounded enumeration, not a general-purpose integer.
    op.execute("""
        CREATE TABLE stage_baselines (
            id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            stage_id        TEXT NOT NULL,
            hour_of_week    SMALLINT NOT NULL
                                CHECK (hour_of_week >= 0 AND hour_of_week <= 167),
            metric          TEXT NOT NULL
                                CHECK (metric IN ('latency_ms', 'row_count', 'error_rate')),
            baseline_mean   DOUBLE PRECISION NOT NULL,
            baseline_std    DOUBLE PRECISION NOT NULL
                                CHECK (baseline_std >= 0.0),
            sample_count    INTEGER NOT NULL
                                CHECK (sample_count >= 1),
            fitted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (stage_id, hour_of_week, metric)
        )
    """)

    # Compound index on (stage_id, metric) for the most common access pattern:
    # loading all 168 hour-of-week baselines for a given stage and metric.
    # The UNIQUE constraint index has stage_id as its leftmost key already;
    # this index makes metric the second filter without scanning all 168 slots.
    op.execute("""
        CREATE INDEX idx_stage_baselines_stage_metric
            ON stage_baselines (stage_id, metric)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS stage_baselines")
