"""
Migration smoke test — applies the Alembic migration against a real PostgreSQL
container and verifies that a PipelineMetric row survives a full insert/select
round-trip.

I'm using testcontainers rather than mocking the database layer because migration
bugs are schema-level bugs — they only surface against a real database engine. A
mock or an in-memory SQLite database would accept DDL that PostgreSQL rejects
(e.g. PARTITION BY RANGE syntax, GENERATED ALWAYS AS IDENTITY, partial indexes with
WHERE clauses). Any false-positive from an in-memory fixture would mean the
migration fails on the first real deployment, which is the worst time to find out.
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

from ingestion.models import AnomalyEvent, FaultLocalization, PipelineMetric

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"

# I'm pinning the same postgres:16-alpine image used in docker-compose.yml so that
# the smoke test exercises exactly the same engine version as production. Using
# postgres:latest would silently upgrade on the next CI run and could surface
# Postgres-version-specific DDL differences at an unpredictable time.
POSTGRES_IMAGE = "postgres:16-alpine"


def _run_migrations(db_url: str) -> None:
    """
    I'm running migrations programmatically via the Alembic Python API rather than
    shelling out to `alembic upgrade head` because shelling out requires the alembic
    binary to be on PATH and assumes the working directory matches script_location in
    alembic.ini. The Python API is environment-independent and integrable into pytest
    without any subprocess plumbing.
    """
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


@pytest.fixture(scope="module")
def pg_engine():
    """
    I'm scoping this fixture to the module rather than to each test function because
    spinning up a Postgres container takes 2–4 seconds on a cold Docker pull. All
    tests in this module share the same migrated database; they are read-mostly after
    the initial insert and do not mutate each other's rows.
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
    I'm inserting the fixture row once at module scope and sharing it across tests
    that verify individual fields. Re-inserting per test would generate a new id each
    time and make the primary key assertions in later tests non-deterministic.
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
    I'm grouping migration-level assertions separately from round-trip assertions so
    that a failing schema check produces a clear error message pointing at the migration
    file, not at the application-level ORM code.
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
        I'm checking that all 12 monthly partitions plus the DEFAULT partition were
        created. A partial partition set would silently accept inserts into DEFAULT
        for out-of-range months and make range queries miss data without raising an
        error.
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
        I'm verifying the (stage_id, event_time) index was created because the sliding
        window aggregator in Milestone 9 will issue queries of the form
        WHERE stage_id = ? AND event_time > ? — without this index those queries would
        do a full partition scan.
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
    I'm splitting round-trip assertions into focused single-field tests rather than
    one monolithic assert block so that a type-mismatch on latency_ms produces
    'AssertionError: latency mismatch' rather than 'AssertionError' with no context
    about which field failed.
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
        I'm querying the child partition directly rather than the parent to confirm
        that PostgreSQL's partition routing sent the row to pipeline_metrics_2024_01
        and not to the DEFAULT partition. A row in DEFAULT means the partition range
        is misconfigured and the monthly partition would be empty during time-range
        queries.
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
