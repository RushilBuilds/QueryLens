"""
Integration test for the full API against real PostgreSQL (M26).

Spins up a testcontainers PostgreSQL instance, runs Alembic migrations, seeds
data via the ORM, and hits every API endpoint. Redpanda is mocked — the API
only uses it for /health and list_topics, which does not justify a JVM-based
container in CI.

Verifies:
  - /health returns 200 with connected PostgreSQL and 503 when engine is dead
  - /stages returns seeded stages with circuit breaker state
  - /stages/{id}/metrics returns paginated results with correct total
  - /stages/{id}/anomalies filters by detector type
  - /localizations returns seeded localizations with ranked candidates
  - /localizations/{id} returns full detail
  - /healing/actions lists actions and filters by outcome
  - POST /healing/actions/{id}/override cancels a pending action
  - Response latency for paginated queries under 200ms
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.models import (
    AnomalyEventRow,
    CircuitBreakerStateRow,
    FaultLocalizationRow,
    HealingActionRow,
    PipelineMetric,
)

try:
    from testcontainers.postgres import PostgresContainer
    import docker
    docker.from_env().ping()
    _CONTAINERS_AVAILABLE = True
except Exception:
    _CONTAINERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CONTAINERS_AVAILABLE,
    reason="testcontainers or Docker not available",
)

POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _mock_kafka_connected() -> MagicMock:
    producer = MagicMock()
    metadata = MagicMock()
    metadata.topics = {"pipeline.metrics": MagicMock()}
    producer.list_topics.return_value = metadata
    return producer


def _seed_data(db_url: str) -> None:
    """Populates all tables with enough data to exercise every endpoint."""
    engine = create_engine(db_url)
    with Session(engine) as session:
        # 200 pipeline metrics across 2 stages
        for i in range(100):
            session.add(PipelineMetric(
                stage_id="ingest_kafka",
                event_time=T0 + timedelta(seconds=i * 10),
                latency_ms=10.0 + (i % 20),
                row_count=500,
                payload_bytes=4096,
                status="ok" if i < 95 else "error",
                fault_label="latency_spike" if i >= 95 else None,
                replayed=False,
            ))
        for i in range(100):
            session.add(PipelineMetric(
                stage_id="transform_enrich",
                event_time=T0 + timedelta(seconds=i * 10),
                latency_ms=25.0 + (i % 30),
                row_count=450,
                payload_bytes=8192,
                status="ok",
                fault_label=None,
                replayed=False,
            ))

        # Anomaly events
        for i in range(10):
            session.add(AnomalyEventRow(
                stage_id="ingest_kafka",
                detector_type="cusum" if i < 5 else "ewma",
                metric="latency_ms",
                signal="upper",
                detector_value=5.0 + i,
                threshold=4.0,
                z_score=3.0,
                detected_at=T0 + timedelta(minutes=i),
                fault_label="latency_spike",
                schema_version=1,
                created_at=T0 + timedelta(minutes=i, seconds=1),
            ))

        # Circuit breaker states
        session.add(CircuitBreakerStateRow(
            stage_id="ingest_kafka",
            state="closed",
            failure_count=0,
            trip_count=1,
            opened_at=None,
            updated_at=T0,
        ))
        session.add(CircuitBreakerStateRow(
            stage_id="transform_enrich",
            state="open",
            failure_count=0,
            trip_count=3,
            opened_at=T0,
            updated_at=T0,
        ))

        # Localizations
        session.add(FaultLocalizationRow(
            hypothesis_id="hyp-int-001",
            triggered_at=T0,
            root_cause_stage_id="ingest_kafka",
            posterior_probability=0.82,
            ranked_candidates_json=json.dumps([["ingest_kafka", 0.82], ["transform_enrich", 0.18]]),
            evidence_json=json.dumps([{"stage_id": "ingest_kafka", "metric": "latency_ms"}]),
            evidence_count=1,
            true_label=None,
            created_at=T0 + timedelta(seconds=5),
        ))

        # Healing actions
        session.add(HealingActionRow(
            hypothesis_id="hyp-int-001",
            stage_id="ingest_kafka",
            action="circuit_break",
            fault_type="latency_spike",
            severity="high",
            outcome="pending",
            triggered_at=T0,
            resolved_at=None,
            notes=None,
        ))

        session.commit()
    engine.dispose()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_url():
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        url = pg.get_connection_url().replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
        _run_migrations(url)
        _seed_data(url)
        yield url


@pytest.fixture(scope="module")
def client(db_url: str):
    from api.main import create_app

    engine = create_engine(db_url, pool_pre_ping=True)
    app = create_app()
    app.state.db_engine = engine
    app.state.kafka_producer = _mock_kafka_connected()
    app.state.settings = MagicMock()

    yield TestClient(app, raise_server_exceptions=True)
    engine.dispose()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthIntegration:

    def test_health_200_with_live_db(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["db"] == "connected"

    def test_health_503_with_dead_db(self, db_url: str) -> None:
        from api.main import create_app

        dead_engine = create_engine(
            "postgresql+psycopg2://nobody:bad@127.0.0.1:1/none",
            pool_pre_ping=True,
        )
        app = create_app()
        app.state.db_engine = dead_engine
        app.state.kafka_producer = _mock_kafka_connected()
        app.state.settings = MagicMock()

        dead_client = TestClient(app, raise_server_exceptions=False)
        resp = dead_client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["db"] == "error"
        dead_engine.dispose()


# ---------------------------------------------------------------------------
# /stages
# ---------------------------------------------------------------------------


class TestStagesIntegration:

    def test_returns_both_stages(self, client: TestClient) -> None:
        resp = client.get("/stages")
        assert resp.status_code == 200
        ids = [s["stage_id"] for s in resp.json()]
        assert "ingest_kafka" in ids
        assert "transform_enrich" in ids

    def test_circuit_breaker_state_present(self, client: TestClient) -> None:
        stages = client.get("/stages").json()
        kafka = next(s for s in stages if s["stage_id"] == "ingest_kafka")
        assert kafka["circuit_breaker"]["state"] == "closed"
        enrich = next(s for s in stages if s["stage_id"] == "transform_enrich")
        assert enrich["circuit_breaker"]["state"] == "open"

    def test_p99_latency_is_numeric(self, client: TestClient) -> None:
        stages = client.get("/stages").json()
        for stage in stages:
            assert isinstance(stage["p99_latency_ms"], (int, float))
            assert stage["p99_latency_ms"] > 0


# ---------------------------------------------------------------------------
# /stages/{stage_id}/metrics
# ---------------------------------------------------------------------------


class TestStageMetricsIntegration:

    def test_returns_paginated_metrics(self, client: TestClient) -> None:
        resp = client.get("/stages/ingest_kafka/metrics?page_size=10")
        body = resp.json()
        assert body["total"] == 100
        assert len(body["items"]) == 10

    def test_response_latency_under_200ms(self, client: TestClient) -> None:
        start = time.monotonic()
        client.get("/stages/ingest_kafka/metrics?page_size=100")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 200, f"Response took {elapsed_ms:.0f}ms, expected < 200ms"


# ---------------------------------------------------------------------------
# /stages/{stage_id}/anomalies
# ---------------------------------------------------------------------------


class TestStageAnomaliesIntegration:

    def test_returns_all_anomalies(self, client: TestClient) -> None:
        resp = client.get("/stages/ingest_kafka/anomalies")
        assert resp.json()["total"] == 10

    def test_filter_by_detector_type(self, client: TestClient) -> None:
        resp = client.get("/stages/ingest_kafka/anomalies?detector_type=ewma")
        assert resp.json()["total"] == 5


# ---------------------------------------------------------------------------
# /localizations
# ---------------------------------------------------------------------------


class TestLocalizationsIntegration:

    def test_returns_localization(self, client: TestClient) -> None:
        resp = client.get("/localizations")
        assert resp.json()["total"] == 1

    def test_detail_includes_ranked_candidates(self, client: TestClient) -> None:
        resp = client.get("/localizations/hyp-int-001")
        body = resp.json()
        assert len(body["ranked_candidates"]) == 2
        assert body["ranked_candidates"][0]["stage_id"] == "ingest_kafka"


# ---------------------------------------------------------------------------
# /healing/actions + override
# ---------------------------------------------------------------------------


class TestHealingActionsIntegration:

    def test_returns_pending_action(self, client: TestClient) -> None:
        resp = client.get("/healing/actions?outcome=pending")
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["action"] == "circuit_break"

    def test_override_cancels_action(self, client: TestClient) -> None:
        resp = client.post(
            "/healing/actions/hyp-int-001/override",
            json={"operator": "integration_test", "reason": "automated test"},
        )
        assert resp.status_code == 200
        assert resp.json()["outcome"] == "cancelled"

        # Verify it's now cancelled in the list
        resp = client.get("/healing/actions?outcome=cancelled")
        assert resp.json()["total"] >= 1
