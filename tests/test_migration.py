"""
Migration smoke test — applies the Alembic migration against a real PostgreSQL
container and verifies that a PipelineMetric row survives a full insert/select
round-trip.

testcontainers used rather than mocking: migration bugs are schema-level bugs that
only surface against a real engine. SQLite accepts DDL that PostgreSQL rejects
(PARTITION BY RANGE, GENERATED ALWAYS AS IDENTITY, partial indexes with WHERE
clauses). A false-positive from an in-memory fixture would mean the migration fails
on first real deployment.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from testcontainers.postgres import PostgresContainer

from ingestion.models import AnomalyEventRow, FaultLocalizationRow, PipelineMetric

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"

# Same postgres:16-alpine image as docker-compose.yml: exercises the exact engine
# version used in production. postgres:latest would silently upgrade on the next
# CI run and could surface version-specific DDL differences unpredictably.
POSTGRES_IMAGE = "postgres:16-alpine"


def _run_migrations(db_url: str) -> None:
    """
    Runs migrations via the Alembic Python API rather than shelling out: the shell
    approach requires alembic on PATH and assumes the working directory matches
    script_location in alembic.ini. The Python API is environment-independent.
    """
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


@pytest.fixture(scope="module")
def pg_engine():
    """
    Module-scoped: container startup takes 2-4 seconds on a cold pull. All tests
    share the same migrated database and are read-mostly after the initial insert.
    """
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        # testcontainers returns a psycopg2 URL; we need to ensure it uses the
        # psycopg2 driver explicitly so SQLAlchemy picks the sync driver, not asyncpg.
        raw_url = pg.get_connection_url()
        db_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        _run_migrations(db_url)
        engine = create_engine(db_url)
        yield engine
        engine.dispose()


@pytest.fixture(scope="module")
def inserted_metric(pg_engine):
    """
    Fixture row inserted once at module scope. Per-test insertion would generate a
    new id each time, making primary key assertions in later tests non-deterministic.
    """
    with Session(pg_engine) as session:
        metric = PipelineMetric(
            stage_id="source_postgres",
            event_time=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            latency_ms=47.3,
            row_count=1_000,
            payload_bytes=4_096,
            status="ok",
            fault_label=None,
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        )
        session.add(metric)
        session.commit()
        session.refresh(metric)
        return metric


class TestMigrationApplied:
    """
    Migration-level assertions separated from round-trip assertions so a failing
    schema check points at the migration file, not the ORM code.
    """

    def test_pipeline_metrics_table_exists(self, pg_engine) -> None:
        """Verifies the migration ran to completion and created the parent table."""
        with pg_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename = 'pipeline_metrics'"
                )
            )
            assert result.fetchone() is not None, (
                "pipeline_metrics table missing — migration did not apply"
            )

    def test_monthly_partitions_exist(self, pg_engine) -> None:
        """
        All 12 monthly partitions plus DEFAULT must exist. A partial set would
        silently route out-of-range inserts to DEFAULT and make range queries miss
        data without raising an error.
        """
        with pg_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT COUNT(*) FROM pg_inherits i "
                    "JOIN pg_class c ON c.oid = i.inhrelid "
                    "WHERE i.inhparent = 'pipeline_metrics'::regclass"
                )
            )
            partition_count = result.scalar()
        # 12 monthly partitions + 1 DEFAULT partition
        assert partition_count == 13, (
            f"Expected 13 partitions (12 monthly + DEFAULT), found {partition_count}"
        )

    def test_stub_tables_exist(self, pg_engine) -> None:
        """Verifies both stub tables were created by the migration."""
        with pg_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' "
                    "AND tablename IN ('anomaly_events', 'fault_localizations')"
                )
            )
            found = {row[0] for row in result}
        assert "anomaly_events" in found
        assert "fault_localizations" in found

    def test_stage_event_index_exists(self, pg_engine) -> None:
        """
        The (stage_id, event_time) index is required by the sliding window aggregator's
        WHERE stage_id = ? AND event_time > ? queries. Without it, every query does a
        full partition scan.
        """
        with pg_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'pipeline_metrics' "
                    "AND indexname = 'ix_pipeline_metrics_stage_event'"
                )
            )
            assert result.fetchone() is not None, (
                "ix_pipeline_metrics_stage_event index missing"
            )


class TestPipelineMetricRoundTrip:
    """
    Round-trip assertions split into single-field tests so a type mismatch on
    latency_ms produces a targeted failure message rather than a bare AssertionError
    with no field context.
    """

    def test_row_is_persisted(self, pg_engine, inserted_metric) -> None:
        with Session(pg_engine) as session:
            fetched = session.get(
                PipelineMetric, (inserted_metric.id, inserted_metric.event_time)
            )
            assert fetched is not None, (
                f"No row found for PK ({inserted_metric.id}, {inserted_metric.event_time})"
            )

    def test_stage_id_round_trips(self, pg_engine, inserted_metric) -> None:
        with Session(pg_engine) as session:
            fetched = session.get(
                PipelineMetric, (inserted_metric.id, inserted_metric.event_time)
            )
            assert fetched.stage_id == "source_postgres"

    def test_latency_ms_round_trips(self, pg_engine, inserted_metric) -> None:
        with Session(pg_engine) as session:
            fetched = session.get(
                PipelineMetric, (inserted_metric.id, inserted_metric.event_time)
            )
            assert abs(fetched.latency_ms - 47.3) < 1e-6, (
                f"latency_ms round-trip lost precision: expected 47.3, got {fetched.latency_ms}"
            )

    def test_nullable_fault_label_round_trips(self, pg_engine, inserted_metric) -> None:
        with Session(pg_engine) as session:
            fetched = session.get(
                PipelineMetric, (inserted_metric.id, inserted_metric.event_time)
            )
            assert fetched.fault_label is None

    def test_trace_id_round_trips(self, pg_engine, inserted_metric) -> None:
        with Session(pg_engine) as session:
            fetched = session.get(
                PipelineMetric, (inserted_metric.id, inserted_metric.event_time)
            )
            assert fetched.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_row_lands_in_correct_monthly_partition(self, pg_engine, inserted_metric) -> None:
        """
        Child partition queried directly rather than the parent to confirm PostgreSQL
        routed the row to pipeline_metrics_2024_01. A row in DEFAULT means the
        partition range is misconfigured and the monthly partition would be empty
        during time-range queries.
        """
        with pg_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT id FROM pipeline_metrics_2024_01 "
                    "WHERE id = :row_id"
                ),
                {"row_id": inserted_metric.id},
            )
            assert result.fetchone() is not None, (
                "Row did not land in pipeline_metrics_2024_01 — "
                "partition routing or range boundary may be incorrect"
            )
