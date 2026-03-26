"""expand fault_localizations stub — add hypothesis_id, ranked candidates, evidence JSON

Revision ID: 004
Revises: 003
Create Date: 2026-03-27 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the stub column that has no equivalent in the full schema.
    op.drop_index("ix_fault_localizations_stage_id", table_name="fault_localizations")
    op.drop_column("fault_localizations", "stage_id")

    # hypothesis_id is the join key used by the healing layer and audit log.
    # server_default guards against any stub rows from earlier test runs.
    op.add_column(
        "fault_localizations",
        sa.Column(
            "hypothesis_id",
            sa.String(36),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.add_column(
        "fault_localizations",
        sa.Column(
            "triggered_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    # Nullable — None when no candidates were found (source-only fault with no ancestors).
    op.add_column(
        "fault_localizations",
        sa.Column("root_cause_stage_id", sa.String(64), nullable=True),
    )
    op.add_column(
        "fault_localizations",
        sa.Column("posterior_probability", sa.Float(), nullable=True),
    )
    # JSON text columns rather than JSONB: avoids a PostgreSQL-specific type that
    # would require special handling in SQLite during unit tests. The data is
    # always round-tripped through Python json.loads — no in-DB JSON operators needed.
    op.add_column(
        "fault_localizations",
        sa.Column(
            "ranked_candidates_json",
            sa.Text(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "fault_localizations",
        sa.Column(
            "evidence_json",
            sa.Text(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "fault_localizations",
        sa.Column(
            "evidence_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    # true_label is filled post-hoc when the operator confirms the root cause.
    # The HealingAuditLog (M22) reads this to compute online accuracy metrics.
    op.add_column(
        "fault_localizations",
        sa.Column("true_label", sa.String(64), nullable=True),
    )

    op.create_index(
        "ix_fault_localizations_hypothesis_id",
        "fault_localizations",
        ["hypothesis_id"],
        unique=True,
    )
    op.create_index(
        "ix_fault_localizations_triggered_at",
        "fault_localizations",
        ["triggered_at"],
    )
    # Partial index on root_cause_stage_id: most localization queries filter by
    # a specific stage; NULL rows (no candidates found) are never queried by stage.
    op.create_index(
        "ix_fault_localizations_root_cause",
        "fault_localizations",
        ["root_cause_stage_id"],
        postgresql_where=sa.text("root_cause_stage_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_fault_localizations_root_cause", table_name="fault_localizations")
    op.drop_index("ix_fault_localizations_triggered_at", table_name="fault_localizations")
    op.drop_index("ix_fault_localizations_hypothesis_id", table_name="fault_localizations")
    op.drop_column("fault_localizations", "true_label")
    op.drop_column("fault_localizations", "evidence_count")
    op.drop_column("fault_localizations", "evidence_json")
    op.drop_column("fault_localizations", "ranked_candidates_json")
    op.drop_column("fault_localizations", "posterior_probability")
    op.drop_column("fault_localizations", "root_cause_stage_id")
    op.drop_column("fault_localizations", "triggered_at")
    op.drop_column("fault_localizations", "hypothesis_id")
    op.add_column(
        "fault_localizations",
        sa.Column("stage_id", sa.String(64), nullable=False, server_default="unknown"),
    )
    op.create_index(
        "ix_fault_localizations_stage_id",
        "fault_localizations",
        ["stage_id"],
    )
