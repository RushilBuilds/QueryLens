"""
Tests for the FastAPI /health endpoint.

Two modes:
  1. Integration (real PostgreSQL via testcontainers) — verifies 200 when DB is
     reachable and 503 when the engine points at a dead host.
  2. Unit (mocked Kafka) — Redpanda connectivity is tested via a mock producer
     because testcontainers-redpanda requires a JVM-based image pull that doubles
     CI wall time for a single boolean check.

The test overrides app.state directly rather than patching get_settings() because
lifespan has already run by the time the TestClient enters scope. Direct state
mutation is the documented FastAPI pattern for test-time resource swapping.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

try:
    from testcontainers.postgres import PostgresContainer
    _CONTAINERS_AVAILABLE = True
except ImportError:
    _CONTAINERS_AVAILABLE = False

POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _mock_kafka_connected() -> MagicMock:
    """Returns a mock KafkaProducer whose list_topics succeeds."""
    producer = MagicMock()
    metadata = MagicMock()
    metadata.topics = {"pipeline.metrics": MagicMock()}
    producer.list_topics.return_value = metadata
    return producer


def _mock_kafka_disconnected() -> MagicMock:
    """Returns a mock KafkaProducer whose list_topics raises."""
    producer = MagicMock()
    producer.list_topics.side_effect = RuntimeError("broker unreachable")
    return producer


def _build_client(db_engine, kafka_producer) -> TestClient:
    """
    Builds a TestClient with overridden app state. Imports create_app fresh to
    avoid cross-test state leakage from module-level app singleton.
    """
    from api.main import create_app

    test_app = create_app()
    test_app.state.db_engine = db_engine
    test_app.state.kafka_producer = kafka_producer
    test_app.state.settings = MagicMock()
    return TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Integration tests — real PostgreSQL
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CONTAINERS_AVAILABLE, reason="testcontainers not installed")
class TestHealthIntegration:

    @pytest.fixture(scope="class")
    def pg(self):
        with PostgresContainer(POSTGRES_IMAGE) as pg:
            url = pg.get_connection_url().replace(
                "postgresql://", "postgresql+psycopg2://", 1
            )
            _run_migrations(url)
            yield url

    def test_health_returns_200_when_db_and_redpanda_connected(self, pg: str) -> None:
        engine = create_engine(pg, pool_pre_ping=True)
        client = _build_client(engine, _mock_kafka_connected())
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["db"] == "connected"
        assert body["redpanda"] == "connected"
        engine.dispose()

    def test_health_returns_503_when_redpanda_down(self, pg: str) -> None:
        engine = create_engine(pg, pool_pre_ping=True)
        client = _build_client(engine, _mock_kafka_disconnected())
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["db"] == "connected"
        assert body["redpanda"] == "error"
        engine.dispose()

    def test_health_returns_503_when_db_down(self, pg: str) -> None:
        dead_engine = create_engine(
            "postgresql+psycopg2://nobody:bad@127.0.0.1:1/nonexistent",
            pool_pre_ping=True,
        )
        client = _build_client(dead_engine, _mock_kafka_connected())
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["db"] == "error"
        assert body["redpanda"] == "connected"
        dead_engine.dispose()


# ---------------------------------------------------------------------------
# Unit tests — no containers
# ---------------------------------------------------------------------------


class TestHealthUnit:

    def test_both_down_returns_503(self) -> None:
        dead_engine = create_engine(
            "postgresql+psycopg2://nobody:bad@127.0.0.1:1/nonexistent",
            pool_pre_ping=True,
        )
        client = _build_client(dead_engine, _mock_kafka_disconnected())
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["db"] == "error"
        assert body["redpanda"] == "error"
        dead_engine.dispose()

    def test_response_contains_required_fields(self) -> None:
        dead_engine = create_engine(
            "postgresql+psycopg2://nobody:bad@127.0.0.1:1/nonexistent",
            pool_pre_ping=True,
        )
        client = _build_client(dead_engine, _mock_kafka_connected())
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body
        assert "db" in body
        assert "redpanda" in body
        dead_engine.dispose()

    def test_request_id_header_returned(self) -> None:
        dead_engine = create_engine(
            "postgresql+psycopg2://nobody:bad@127.0.0.1:1/nonexistent",
            pool_pre_ping=True,
        )
        client = _build_client(dead_engine, _mock_kafka_connected())
        resp = client.get("/health", headers={"x-request-id": "test-req-42"})
        assert resp.headers.get("x-request-id") == "test-req-42"
        dead_engine.dispose()
