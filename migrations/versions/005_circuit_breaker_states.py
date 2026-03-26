"""add circuit_breaker_states table — per-stage FSM persistence

Revision ID: 005
Revises: 004
Create Date: 2026-03-27 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "circuit_breaker_states",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("stage_id", sa.String(64), nullable=False),
        sa.Column(
            "state",
            sa.String(16),
            nullable=False,
            server_default="closed",
        ),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("trip_count", sa.Integer(), nullable=False, server_default="0"),
        # opened_at is NULL when the breaker is CLOSED — persisted NULL is the
        # correct representation, not a sentinel value, so the registry can
        # reconstruct exact opened_at on restart.
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # stage_id is the registry lookup key — must be unique and fast.
    op.create_index(
        "ix_circuit_breaker_states_stage_id",
        "circuit_breaker_states",
        ["stage_id"],
        unique=True,
    )
    # updated_at index supports dashboard queries that show recently changed breakers.
    op.create_index(
        "ix_circuit_breaker_states_updated_at",
        "circuit_breaker_states",
        ["updated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_circuit_breaker_states_updated_at",
        table_name="circuit_breaker_states",
    )
    op.drop_index(
        "ix_circuit_breaker_states_stage_id",
        table_name="circuit_breaker_states",
    )
    op.drop_table("circuit_breaker_states")
