"""create healing_actions table — audit log for every automated healing action

Revision ID: 007
Revises: 006
Create Date: 2026-03-28 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # outcome uses VARCHAR rather than a Postgres ENUM type so the table
    # remains portable to SQLite in unit tests without a type-cast shim.
    op.create_table(
        "healing_actions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("hypothesis_id", sa.String(36), nullable=False),
        sa.Column("stage_id", sa.String(64), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("fault_type", sa.String(64), nullable=True),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    # Most queries filter by hypothesis_id to correlate with fault_localizations.
    op.create_index(
        "ix_healing_actions_hypothesis_id",
        "healing_actions",
        ["hypothesis_id"],
    )
    # Secondary index for per-stage history views in the dashboard.
    op.create_index(
        "ix_healing_actions_stage_id",
        "healing_actions",
        ["stage_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_healing_actions_stage_id", table_name="healing_actions")
    op.drop_index("ix_healing_actions_hypothesis_id", table_name="healing_actions")
    op.drop_table("healing_actions")
