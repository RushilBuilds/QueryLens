"""stage_baselines — per-stage, per-hour-of-week fitted baseline means and stds

Revision ID: 002
Revises: 001
Create Date: 2026-03-22 00:00:00.000000

I'm adding this table in a separate migration rather than the initial schema
because the columns depend on the SeasonalBaselineModel contract finalized in
Milestone 10. Putting it in 001 would have required changing the column names
once the detection API was designed, producing a corrective migration that
makes the history harder to follow.
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
    # I'm using SMALLINT for hour_of_week (0-167) rather than INTEGER to make
    # the domain constraint self-documenting at the type level. A SMALLINT
    # value outside 0-167 would indicate a computation bug in the fitter, and
    # the narrower type makes it obvious in any schema browser that this column
    # is a bounded enumeration, not a general-purpose integer.
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

    # I'm creating a compound index on (stage_id, metric) because the most
    # common access pattern is "give me all hour-of-week baselines for this
    # stage and metric" — the CUSUM/EWMA detectors load an entire stage's
    # baseline vector at startup, not individual slots. The UNIQUE constraint
    # on (stage_id, hour_of_week, metric) already creates an index, but its
    # leftmost key is stage_id — this index makes metric the second filter
    # without a full scan of all 168 hour slots first.
    op.execute("""
        CREATE INDEX idx_stage_baselines_stage_metric
            ON stage_baselines (stage_id, metric)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS stage_baselines")
