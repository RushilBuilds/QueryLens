"""
Integration test for HealingAuditLog against a real PostgreSQL container.

Test flow:
  1. Run migrations to head
  2. record() inserts a PENDING row and returns its id
  3. resolve() transitions outcome and sets resolved_at
  4. Double-resolve raises ValueError
  5. get() returns None for unknown id
  6. by_hypothesis() returns all rows for a hypothesis_id in order
"""
from __future__ import annotations

import time
from datetime import timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from healing.audit import HealingAuditLog, HealingOutcome
from healing.policy import HealingAction, HealingDecision

try:
    from testcontainers.postgres import PostgresContainer
    _CONTAINERS_AVAILABLE = True
except ImportError:
    _CONTAINERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CONTAINERS_AVAILABLE,
    reason="testcontainers not installed",
)

POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI    = Path(__file__).parent.parent / "alembic.ini"


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _decision(hypothesis_id: str = "hyp-audit-001") -> HealingDecision:
    return HealingDecision(
        action=HealingAction.CIRCUIT_BREAK,
        target_stage_id="source_pg",
        fault_type="latency_spike",
        severity="high",
        rule_matched="latency_spike/high/source→CIRCUIT_BREAK",
        hypothesis_id=hypothesis_id,
    )


@pytest.fixture(scope="module")
def db_url():
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        url = pg.get_connection_url().replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
        _run_migrations(url)
        yield url


class TestHealingAuditLogIntegration:

    def test_record_inserts_pending_row(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        row_id = log.record(_decision("hyp-a-001"))
        log.close()

        engine = create_engine(db_url)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT outcome FROM healing_actions WHERE id = :id"),
                {"id": row_id},
            ).fetchone()
        engine.dispose()

        assert row is not None
        assert row.outcome == "pending"

    def test_record_returns_unique_ids(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        id1 = log.record(_decision("hyp-a-002"))
        id2 = log.record(_decision("hyp-a-003"))
        log.close()
        assert id1 != id2

    def test_resolve_success_updates_outcome(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        row_id = log.record(_decision("hyp-a-004"))
        log.resolve(row_id, HealingOutcome.SUCCESS)
        log.close()

        engine = create_engine(db_url)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT outcome, resolved_at FROM healing_actions WHERE id = :id"),
                {"id": row_id},
            ).fetchone()
        engine.dispose()

        assert row.outcome == "success"
        assert row.resolved_at is not None

    def test_resolve_failed_outcome(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        row_id = log.record(_decision("hyp-a-005"))
        log.resolve(row_id, HealingOutcome.FAILED, notes="stage still degraded after CB trip")
        log.close()

        engine = create_engine(db_url)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT outcome, notes FROM healing_actions WHERE id = :id"),
                {"id": row_id},
            ).fetchone()
        engine.dispose()

        assert row.outcome == "failed"
        assert "degraded" in row.notes

    def test_resolve_cancelled_outcome(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        row_id = log.record(_decision("hyp-a-006"))
        log.resolve(row_id, HealingOutcome.CANCELLED, notes="operator override")
        log.close()

        engine = create_engine(db_url)
        with engine.connect() as conn:
            outcome = conn.execute(
                text("SELECT outcome FROM healing_actions WHERE id = :id"),
                {"id": row_id},
            ).scalar()
        engine.dispose()

        assert outcome == "cancelled"

    def test_double_resolve_raises(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        row_id = log.record(_decision("hyp-a-007"))
        log.resolve(row_id, HealingOutcome.SUCCESS)
        with pytest.raises(ValueError, match="already resolved"):
            log.resolve(row_id, HealingOutcome.FAILED)
        log.close()

    def test_resolve_unknown_id_raises(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        with pytest.raises(ValueError, match="not found"):
            log.resolve(999_999_999, HealingOutcome.SUCCESS)
        log.close()

    def test_get_returns_row(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        row_id = log.record(_decision("hyp-a-008"))
        row = log.get(row_id)
        log.close()

        assert row is not None
        assert row.hypothesis_id == "hyp-a-008"
        assert row.action == "circuit_break"
        assert row.stage_id == "source_pg"

    def test_get_returns_none_for_missing(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        row = log.get(999_999_998)
        log.close()
        assert row is None

    def test_by_hypothesis_returns_all_rows_in_order(self, db_url: str) -> None:
        hyp = "hyp-a-009"
        log = HealingAuditLog(db_url)
        id1 = log.record(_decision(hyp))
        time.sleep(0.01)   # ensure distinct triggered_at timestamps
        id2 = log.record(_decision(hyp))
        rows = log.by_hypothesis(hyp)
        log.close()

        assert len(rows) == 2
        assert rows[0].id == id1
        assert rows[1].id == id2

    def test_by_hypothesis_empty_for_unknown(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        rows = log.by_hypothesis("hyp-does-not-exist")
        log.close()
        assert rows == []

    def test_triggered_at_is_utc(self, db_url: str) -> None:
        log = HealingAuditLog(db_url)
        row_id = log.record(_decision("hyp-a-010"))
        row = log.get(row_id)
        log.close()

        # psycopg2 returns tz-aware datetimes for TIMESTAMPTZ columns
        assert row is not None
        assert row.triggered_at.tzinfo is not None
