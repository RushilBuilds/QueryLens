"""
Tests for /localizations and /healing/actions endpoints (M25).

Uses in-memory SQLite with manual DDL — same pattern as test_api_stages.py.
Covers pagination, detail view, 404 for unknown hypothesis, healing action
list with outcome filter, and the override flow (pending → cancelled, 409 on
double-cancel, 404 on missing hypothesis).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ingestion.models import FaultLocalizationRow, HealingActionRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _build_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE fault_localizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id VARCHAR(36) NOT NULL UNIQUE,
                triggered_at TIMESTAMP NOT NULL,
                root_cause_stage_id VARCHAR(64),
                posterior_probability FLOAT,
                ranked_candidates_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                true_label VARCHAR(64),
                created_at TIMESTAMP NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE healing_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id VARCHAR(36) NOT NULL,
                stage_id VARCHAR(64) NOT NULL,
                action VARCHAR(32) NOT NULL,
                fault_type VARCHAR(64),
                severity VARCHAR(16) NOT NULL,
                outcome VARCHAR(16) NOT NULL DEFAULT 'pending',
                triggered_at TIMESTAMP NOT NULL,
                resolved_at TIMESTAMP,
                notes TEXT
            )
        """))
        conn.commit()
    return engine


def _seed_localizations(session: Session) -> None:
    for i in range(3):
        session.add(FaultLocalizationRow(
            hypothesis_id=f"hyp-{i:03d}",
            triggered_at=T0 + timedelta(minutes=i),
            root_cause_stage_id="source_a" if i < 2 else None,
            posterior_probability=0.85 - i * 0.1 if i < 2 else None,
            ranked_candidates_json=json.dumps([["source_a", 0.85 - i * 0.1]]) if i < 2 else "[]",
            evidence_json=json.dumps([{"stage_id": "source_a", "metric": "latency_ms"}]),
            evidence_count=1,
            true_label=None,
            created_at=T0 + timedelta(minutes=i, seconds=5),
        ))
    session.commit()


def _seed_healing_actions(session: Session) -> None:
    session.add(HealingActionRow(
        hypothesis_id="hyp-000",
        stage_id="source_a",
        action="circuit_break",
        fault_type="latency_spike",
        severity="high",
        outcome="pending",
        triggered_at=T0,
        resolved_at=None,
        notes=None,
    ))
    session.add(HealingActionRow(
        hypothesis_id="hyp-001",
        stage_id="source_a",
        action="replay_range",
        fault_type="dropped_connection",
        severity="medium",
        outcome="success",
        triggered_at=T0 + timedelta(minutes=1),
        resolved_at=T0 + timedelta(minutes=2),
        notes="replayed 500 records",
    ))
    session.commit()


@pytest.fixture()
def client():
    engine = _build_engine()
    with Session(engine) as session:
        _seed_localizations(session)
        _seed_healing_actions(session)

    from api.main import create_app
    app = create_app()
    app.state.db_engine = engine
    app.state.kafka_producer = MagicMock()
    app.state.settings = MagicMock()

    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GET /localizations
# ---------------------------------------------------------------------------


class TestListLocalizations:

    def test_returns_all_localizations(self, client: TestClient) -> None:
        resp = client.get("/localizations")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3

    def test_pagination(self, client: TestClient) -> None:
        resp = client.get("/localizations?page=1&page_size=2")
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["total"] == 3

    def test_summary_includes_top_candidate(self, client: TestClient) -> None:
        resp = client.get("/localizations")
        items = resp.json()["items"]
        # Most recent first (hyp-002)
        hyp_000 = next(i for i in items if i["hypothesis_id"] == "hyp-000")
        assert hyp_000["root_cause_stage_id"] == "source_a"
        assert hyp_000["posterior_probability"] == pytest.approx(0.85, abs=0.01)


# ---------------------------------------------------------------------------
# GET /localizations/{hypothesis_id}
# ---------------------------------------------------------------------------


class TestGetLocalization:

    def test_returns_detail_with_ranked_candidates(self, client: TestClient) -> None:
        resp = client.get("/localizations/hyp-000")
        assert resp.status_code == 200
        body = resp.json()
        assert body["hypothesis_id"] == "hyp-000"
        assert len(body["ranked_candidates"]) == 1
        assert body["ranked_candidates"][0]["stage_id"] == "source_a"

    def test_returns_evidence_events(self, client: TestClient) -> None:
        resp = client.get("/localizations/hyp-000")
        body = resp.json()
        assert len(body["evidence_events"]) == 1

    def test_returns_404_for_unknown_hypothesis(self, client: TestClient) -> None:
        resp = client.get("/localizations/does-not-exist")
        assert resp.status_code == 404

    def test_no_candidates_returns_empty_list(self, client: TestClient) -> None:
        resp = client.get("/localizations/hyp-002")
        body = resp.json()
        assert body["ranked_candidates"] == []
        assert body["root_cause_stage_id"] is None


# ---------------------------------------------------------------------------
# GET /healing/actions
# ---------------------------------------------------------------------------


class TestListHealingActions:

    def test_returns_all_actions(self, client: TestClient) -> None:
        resp = client.get("/healing/actions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2

    def test_filter_by_outcome(self, client: TestClient) -> None:
        resp = client.get("/healing/actions?outcome=pending")
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["outcome"] == "pending"

    def test_action_fields_present(self, client: TestClient) -> None:
        resp = client.get("/healing/actions")
        item = resp.json()["items"][0]
        for field in ["hypothesis_id", "stage_id", "action", "severity", "outcome", "triggered_at"]:
            assert field in item


# ---------------------------------------------------------------------------
# POST /healing/actions/{hypothesis_id}/override
# ---------------------------------------------------------------------------


class TestOverrideAction:

    def test_cancels_pending_action(self, client: TestClient) -> None:
        resp = client.post(
            "/healing/actions/hyp-000/override",
            json={"operator": "oncall_engineer", "reason": "false positive"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["outcome"] == "cancelled"
        assert body["operator"] == "oncall_engineer"

    def test_double_cancel_returns_409(self, client: TestClient) -> None:
        # First cancel succeeds
        client.post(
            "/healing/actions/hyp-000/override",
            json={"operator": "eng1"},
        )
        # Second cancel returns 409
        resp = client.post(
            "/healing/actions/hyp-000/override",
            json={"operator": "eng2"},
        )
        assert resp.status_code == 409

    def test_override_already_resolved_returns_409(self, client: TestClient) -> None:
        resp = client.post(
            "/healing/actions/hyp-001/override",
            json={"operator": "eng1"},
        )
        assert resp.status_code == 409
        assert "already resolved" in resp.json()["detail"]

    def test_override_unknown_hypothesis_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/healing/actions/unknown-hyp/override",
            json={"operator": "eng1"},
        )
        assert resp.status_code == 404
