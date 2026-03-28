"""add replayed column to pipeline_metrics — flags rows inserted by ReplayOrchestrator

Revision ID: 006
Revises: 005
Create Date: 2026-03-28 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ALTER TABLE on the partitioned parent propagates to all existing child
    # partitions automatically. server_default=False means existing rows are
    # marked as original ingestion, not replay — correct semantics without a
    # backfill.
    op.add_column(
        "pipeline_metrics",
        sa.Column(
            "replayed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # Partial index covers only replayed=True rows — the dominant query is
    # "show me what was replayed for hypothesis X". The NULL majority (original
    # rows) never needs this index.
    op.execute("""
        CREATE INDEX ix_pipeline_metrics_replayed
        ON pipeline_metrics (replayed)
        WHERE replayed = TRUE
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_pipeline_metrics_replayed")
    op.drop_column("pipeline_metrics", "replayed")
