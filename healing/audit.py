from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import structlog
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from healing.policy import HealingDecision
from ingestion.models import HealingActionRow

_log = structlog.get_logger(__name__)


class HealingOutcome(Enum):
    PENDING   = "pending"    # action started, awaiting confirmation
    SUCCESS   = "success"    # stage recovered, action confirmed effective
    FAILED    = "failed"     # action executed but stage did not recover
    CANCELLED = "cancelled"  # operator overrode via API before execution


class HealingAuditLog:
    """
    Writes and updates healing action records in the healing_actions table.
    Each call to record() inserts a PENDING row and returns the row id so
    callers can later call resolve() with the final outcome.

    Separating insert from update lets the executor record the action before
    it runs — a crash between insert and resolve leaves a PENDING row that the
    operator can inspect, which is better than losing the record entirely.
    """

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url, pool_pre_ping=True)

    def record(self, decision: HealingDecision) -> int:
        """
        Inserts a PENDING row for the given decision. Returns the new row id.
        triggered_at is set to UTC now so the row timestamp reflects when the
        executor received the decision, not when the policy engine produced it.
        """
        row = HealingActionRow(
            hypothesis_id=decision.hypothesis_id,
            stage_id=decision.target_stage_id,
            action=decision.action.value,
            fault_type=decision.fault_type,
            severity=decision.severity,
            outcome=HealingOutcome.PENDING.value,
            triggered_at=datetime.now(tz=timezone.utc),
            resolved_at=None,
            notes=None,
        )
        with Session(self._engine) as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            row_id = row.id

        _log.info(
            "healing_action_recorded",
            row_id=row_id,
            hypothesis_id=decision.hypothesis_id,
            action=decision.action.value,
            stage_id=decision.target_stage_id,
        )
        return row_id

    def resolve(
        self,
        row_id: int,
        outcome: HealingOutcome,
        notes: Optional[str] = None,
    ) -> None:
        """
        Updates the outcome and resolved_at for an existing PENDING row.
        Raises ValueError if the row does not exist or is already resolved —
        double-resolve would silently overwrite a legitimate outcome.
        """
        with Session(self._engine) as session:
            row = session.get(HealingActionRow, row_id)
            if row is None:
                raise ValueError(f"HealingActionRow id={row_id} not found")
            if row.outcome != HealingOutcome.PENDING.value:
                raise ValueError(
                    f"Row id={row_id} already resolved as '{row.outcome}'"
                )
            row.outcome = outcome.value
            row.resolved_at = datetime.now(tz=timezone.utc)
            row.notes = notes
            session.commit()

        _log.info(
            "healing_action_resolved",
            row_id=row_id,
            outcome=outcome.value,
            notes=notes,
        )

    def get(self, row_id: int) -> Optional[HealingActionRow]:
        """
        Fetches a single row by id. Returns None if not found.
        Exposes the ORM row directly so callers can read any field without
        an additional DTO layer — the table schema is stable across the audit log's
        lifecycle.
        """
        with Session(self._engine) as session:
            return session.get(HealingActionRow, row_id)

    def by_hypothesis(self, hypothesis_id: str) -> list[HealingActionRow]:
        """
        Returns all rows for the given hypothesis_id ordered by triggered_at.
        One hypothesis can produce multiple rows if the operator triggers a
        re-evaluation after FAILED — this returns the full sequence.
        """
        with Session(self._engine) as session:
            stmt = (
                select(HealingActionRow)
                .where(HealingActionRow.hypothesis_id == hypothesis_id)
                .order_by(HealingActionRow.triggered_at)
            )
            return list(session.scalars(stmt))

    def close(self) -> None:
        self._engine.dispose()
