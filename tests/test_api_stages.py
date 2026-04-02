"""
Tests for /stages, /stages/{stage_id}/metrics, and /stages/{stage_id}/anomalies.

Uses an in-memory SQLite database with fixture data rather than PostgreSQL
containers: these tests verify route logic, pagination, and filtering — not
SQL dialect specifics. The integration test suite (M26) covers real PostgreSQL.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ingestion.models import (
    AnomalyEventRow,
    Base,
    CircuitBreakerStateRow,
    PipelineMetric,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _build_engine():
    """
    In-memory SQLite with simplified DDL. PipelineMetric has a composite PK
    (id, event_time) for PostgreSQL partitioning that SQLite cannot handle with
    autoincrement, so tables are created via raw DDL instead of metadata.create_all.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE pipeline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage_id VARCHAR(64) NOT NULL,
                event_time TIMESTAMP NOT NULL,
                latency_ms FLOAT NOT NULL,
                row_count INTEGER NOT NULL,
                payload_bytes BIGINT NOT NULL,
                status VARCHAR(32) NOT NULL,
                fault_label VARCHAR(64),
                trace_id VARCHAR(32),
                replayed BOOLEAN NOT NULL DEFAULT 0
            )
        """))
        conn.execute(text("""
            CREATE TABLE anomaly_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage_id VARCHAR(64) NOT NULL,
                detector_type VARCHAR(16) NOT NULL,
                metric VARCHAR(32) NOT NULL,
                signal VARCHAR(8) NOT NULL,
                detector_value FLOAT NOT NULL,
                threshold FLOAT NOT NULL,
                z_score FLOAT NOT NULL,
                detected_at TIMESTAMP NOT NULL,
                fault_label VARCHAR(64),
                schema_version SMALLINT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE circuit_breaker_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage_id VARCHAR(64) NOT NULL UNIQUE,
                state VARCHAR(16) NOT NULL,
                failure_count INTEGER NOT NULL,
                trip_count INTEGER NOT NULL,
                opened_at TIMESTAMP,
                updated_at TIMESTAMP NOT NULL
            )
        """))
        conn.commit()
    return engine


def _seed_metrics(session: Session) -> None:
    """Inserts 5 metrics for stage_a and 3 for stage_b."""
    from datetime import timedelta

    for i in range(5):
        session.add(PipelineMetric(
            stage_id="stage_a",
            event_time=T0 + timedelta(minutes=i),
            latency_ms=10.0 + i * 5.0,
            row_count=100 + i,
            payload_bytes=1024,
            status="ok",
            fault_label=None,
            replayed=False,
        ))
    for i in range(3):
        session.add(PipelineMetric(
            stage_id="stage_b",
            event_time=T0 + timedelta(minutes=i),
            latency_ms=50.0 + i * 10.0,
            row_count=200,
            payload_bytes=2048,
            status="ok" if i < 2 else "error",
            fault_label="latency_spike" if i == 2 else None,
            replayed=False,
        ))
    session.commit()


def _seed_anomalies(session: Session) -> None:
    """Inserts 4 anomalies for stage_a: 2 cusum, 2 ewma."""
    from datetime import timedelta

    for i, dtype in enumerate(["cusum", "cusum", "ewma", "ewma"]):
        session.add(AnomalyEventRow(
            stage_id="stage_a",
            detector_type=dtype,
            metric="latency_ms",
            signal="upper",
            detector_value=5.0 + i,
            threshold=4.0,
            z_score=3.0 + i * 0.1,
            detected_at=T0 + timedelta(minutes=i),
            fault_label="latency_spike",
            schema_version=1,
            created_at=T0 + timedelta(minutes=i, seconds=1),
        ))
    session.commit()


def _seed_breaker(session: Session) -> None:
    session.add(CircuitBreakerStateRow(
        stage_id="stage_a",
        state="closed",
        failure_count=0,
        trip_count=2,
        opened_at=None,
        updated_at=T0,
    ))
    session.add(CircuitBreakerStateRow(
        stage_id="stage_b",
        state="open",
        failure_count=0,
        trip_count=1,
        opened_at=T0,
        updated_at=T0,
    ))
    session.commit()


@pytest.fixture()
def client():
    engine = _build_engine()
    with Session(engine) as session:
        _seed_metrics(session)
        _seed_anomalies(session)
        _seed_breaker(session)

    from api.main import create_app
    app = create_app()
    app.state.db_engine = engine
    app.state.kafka_producer = MagicMock()
    app.state.settings = MagicMock()

    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GET /stages
# ---------------------------------------------------------------------------


class TestListStages:

    def test_returns_all_stages(self, client: TestClient) -> None:
        resp = client.get("/stages")
        assert resp.status_code == 200
        body = resp.json()
        stage_ids = [s["stage_id"] for s in body]
        assert "stage_a" in stage_ids
        assert "stage_b" in stage_ids

    def test_includes_circuit_breaker_state(self, client: TestClient) -> None:
        resp = client.get("/stages")
        body = resp.json()
        stage_a = next(s for s in body if s["stage_id"] == "stage_a")
        assert stage_a["circuit_breaker"]["state"] == "closed"
        assert stage_a["circuit_breaker"]["trip_count"] == 2
        stage_b = next(s for s in body if s["stage_id"] == "stage_b")
        assert stage_b["circuit_breaker"]["state"] == "open"

    def test_includes_event_count(self, client: TestClient) -> None:
        resp = client.get("/stages")
        body = resp.json()
        stage_a = next(s for s in body if s["stage_id"] == "stage_a")
        assert stage_a["event_count"] == 5
        stage_b = next(s for s in body if s["stage_id"] == "stage_b")
        assert stage_b["event_count"] == 3


# ---------------------------------------------------------------------------
# GET /stages/{stage_id}/metrics
# ---------------------------------------------------------------------------


class TestStageMetrics:

    def test_returns_metrics_for_stage(self, client: TestClient) -> None:
        resp = client.get("/stages/stage_a/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["stage_id"] == "stage_a"
        assert body["total"] == 5
        assert len(body["items"]) == 5

    def test_pagination_limits_results(self, client: TestClient) -> None:
        resp = client.get("/stages/stage_a/metrics?page=1&page_size=2")
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["total"] == 5
        assert body["page"] == 1

    def test_page_two_returns_remaining(self, client: TestClient) -> None:
        resp = client.get("/stages/stage_a/metrics?page=2&page_size=3")
        body = resp.json()
        assert len(body["items"]) == 2  # 5 total, page 2 of 3-per-page

    def test_empty_stage_returns_zero_items(self, client: TestClient) -> None:
        resp = client.get("/stages/nonexistent/metrics")
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_time_range_filter(self, client: TestClient) -> None:
        from datetime import timedelta
        start = (T0 + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
        end = (T0 + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%S")
        resp = client.get(f"/stages/stage_a/metrics?start={start}&end={end}")
        body = resp.json()
        # minutes 1 and 2 fall in [start, end)
        assert body["total"] == 2


# ---------------------------------------------------------------------------
# GET /stages/{stage_id}/anomalies
# ---------------------------------------------------------------------------


class TestStageAnomalies:

    def test_returns_anomalies_for_stage(self, client: TestClient) -> None:
        resp = client.get("/stages/stage_a/anomalies")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 4

    def test_filter_by_detector_type(self, client: TestClient) -> None:
        resp = client.get("/stages/stage_a/anomalies?detector_type=cusum")
        body = resp.json()
        assert body["total"] == 2
        for item in body["items"]:
            assert item["detector_type"] == "cusum"

    def test_pagination(self, client: TestClient) -> None:
        resp = client.get("/stages/stage_a/anomalies?page=1&page_size=2")
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["total"] == 4

    def test_empty_for_unknown_stage(self, client: TestClient) -> None:
        resp = client.get("/stages/nonexistent/anomalies")
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_anomaly_fields_present(self, client: TestClient) -> None:
        resp = client.get("/stages/stage_a/anomalies?page_size=1")
        item = resp.json()["items"][0]
        assert "detector_type" in item
        assert "metric" in item
        assert "signal" in item
        assert "detector_value" in item
        assert "threshold" in item
        assert "z_score" in item
        assert "detected_at" in item
