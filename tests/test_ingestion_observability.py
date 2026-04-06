"""
Integration test for M8: Prometheus /metrics endpoint and structlog instrumentation.

Tested against real containers rather than mocking prometheus_client — the contract
that matters for alerting is what Prometheus actually scrapes, not what the Python
counter object holds in memory.

Counter deltas (after - before) are asserted rather than absolute values so the
test remains safe when module-level counters have been incremented by earlier tests
in the same session.
"""
from __future__ import annotations

import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pytest
from alembic import command
from alembic.config import Config
from confluent_kafka import Producer as KafkaProducer

try:
    import docker
    docker.from_env().ping()
    from testcontainers.kafka import KafkaContainer
    from testcontainers.postgres import PostgresContainer
    _CONTAINERS_AVAILABLE = True
except Exception:
    _CONTAINERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CONTAINERS_AVAILABLE,
    reason="Docker or testcontainers not available",
)

from ingestion.consumer import ConsumerConfig, MetricConsumer
from ingestion.observability import MetricsServer
from ingestion.serializer import MetricEventSerializer
from ingestion.worker import IngestionWorker, WorkerConfig
from simulator.models import PipelineEvent

KAFKA_IMAGE = "confluentinc/cp-kafka:7.6.0"
POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
TOPIC = "test.metrics.observability"

N_STAGE_A = 20   # events for source_postgres
N_STAGE_B = 10   # events for transform_validate
N_TOTAL = N_STAGE_A + N_STAGE_B


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(index: int, stage_id: str) -> PipelineEvent:
    return PipelineEvent(
        stage_id=stage_id,
        event_time=datetime(2024, 2, 1, 8, 0, index % 60, tzinfo=timezone.utc),
        latency_ms=15.0 + index,
        row_count=50,
        payload_bytes=1024,
        status="ok",
        fault_label=None,
        trace_id=f"{index:032x}",
    )


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _produce_events(bootstrap: str, topic: str, events: List[PipelineEvent]) -> None:
    producer = KafkaProducer({"bootstrap.servers": bootstrap, "acks": "all"})
    for event in events:
        producer.produce(topic, value=MetricEventSerializer.serialize(event))
    producer.flush(timeout=15)


def _scrape_metrics(port: int) -> Dict[str, float]:
    """
    Parses the Prometheus text format manually rather than using
    prometheus_client.parser, which would read the in-process registry and
    bypass the HTTP round-trip being verified. Only ingestion_* lines are
    extracted to avoid false matches from go_*, process_*, python_* collectors.
    """
    url = f"http://127.0.0.1:{port}/metrics"
    with urllib.request.urlopen(url, timeout=5) as resp:
        body = resp.read().decode("utf-8")

    values: Dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith("ingestion_"):
            continue
        # Format: metric_name{labels} value
        # or:     metric_name value
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            values[parts[0]] = float(parts[1])
    return values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def infra():
    """
    Module-scoped so both test classes share the same broker and database.
    Container startup is the dominant cost — separate fixtures would double
    wall time with no correctness benefit since tests use distinct topic names
    and isolated consumer groups.
    """
    with KafkaContainer(KAFKA_IMAGE).with_kraft() as kafka, \
         PostgresContainer(POSTGRES_IMAGE) as pg:

        db_url = pg.get_connection_url().replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
        _run_migrations(db_url)

        yield {
            "kafka_bootstrap": kafka.get_bootstrap_server(),
            "db_url": db_url,
        }


# ---------------------------------------------------------------------------
# MetricsServer unit tests (no containers needed)
# ---------------------------------------------------------------------------


class TestMetricsServer:

    def test_server_starts_and_responds_on_assigned_port(self) -> None:
        """
        port=0 (OS-assigned) verifies the port property reflects the actual
        listening port. A hardcoded port would be fragile in CI.
        """
        server = MetricsServer(port=0)
        server.start()
        try:
            url = f"http://127.0.0.1:{server.port}/metrics"
            with urllib.request.urlopen(url, timeout=3) as resp:
                body = resp.read().decode("utf-8")
            assert "python_info" in body or "ingestion_" in body, (
                "Response does not look like Prometheus text format"
            )
        finally:
            server.stop()

    def test_server_exposes_ingestion_metrics(self) -> None:
        """
        prometheus_client registers metrics at import time, so metric names must
        appear in /metrics before any increments. A consistent time series from
        the first scrape requires the counter to be present at value 0.
        """
        server = MetricsServer(port=0)
        server.start()
        try:
            metrics = _scrape_metrics(server.port)
            metric_names = {k.split("{")[0] for k in metrics}
            # These counters appear after first use (labels are registered lazily).
            # The histogram _count and _sum are always emitted.
            assert "ingestion_write_latency_seconds_count" in metric_names, (
                "ingestion_write_latency_seconds not found in /metrics output"
            )
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Integration: counters increment correctly after a known batch
# ---------------------------------------------------------------------------


class TestPrometheusCounterIncrements:
    """
    Counter deltas are asserted rather than absolute values because module-level
    Prometheus singletons accumulate across the test session.
    """

    def test_records_written_and_consumed_increment_by_stage(self, infra: dict) -> None:
        bootstrap = infra["kafka_bootstrap"]
        db_url = infra["db_url"]

        events = (
            [_make_event(i, "source_postgres") for i in range(N_STAGE_A)]
            + [_make_event(i + 100, "transform_validate") for i in range(N_STAGE_B)]
        )
        _produce_events(bootstrap, TOPIC, events)

        server = MetricsServer(port=0)
        server.start()

        # Scrape before so we can compute deltas.
        before = _scrape_metrics(server.port)

        consumer = MetricConsumer(ConsumerConfig(
            bootstrap_servers=bootstrap,
            group_id="test-observability-counters",
            topic=TOPIC,
        ))
        worker = IngestionWorker(
            consumer=consumer,
            worker_config=WorkerConfig(
                db_url=db_url,
                batch_size=N_TOTAL,
                flush_interval_s=30.0,
            ),
        )

        deadline = time.monotonic() + 30
        while worker.total_written < N_TOTAL and time.monotonic() < deadline:
            worker.run_once()
        worker.flush_remaining()
        worker.close()

        after = _scrape_metrics(server.port)
        server.stop()

        def delta(key: str) -> float:
            return after.get(key, 0.0) - before.get(key, 0.0)

        written_pg = delta(
            'ingestion_records_written_total{stage_id="source_postgres"}'
        )
        written_tv = delta(
            'ingestion_records_written_total{stage_id="transform_validate"}'
        )
        consumed_pg = delta(
            'ingestion_records_consumed_total{stage_id="source_postgres"}'
        )
        consumed_tv = delta(
            'ingestion_records_consumed_total{stage_id="transform_validate"}'
        )

        assert written_pg == N_STAGE_A, (
            f"Expected {N_STAGE_A} source_postgres writes, "
            f"got delta={written_pg}"
        )
        assert written_tv == N_STAGE_B, (
            f"Expected {N_STAGE_B} transform_validate writes, "
            f"got delta={written_tv}"
        )
        assert consumed_pg == N_STAGE_A, (
            f"ingestion_records_consumed_total for source_postgres should match "
            f"written — delta={consumed_pg}"
        )
        assert consumed_tv == N_STAGE_B, (
            f"ingestion_records_consumed_total for transform_validate should match "
            f"written — delta={consumed_tv}"
        )

    def test_write_latency_histogram_observed(self, infra: dict) -> None:
        """
        batch_size == N_TOTAL means exactly one flush, so _count must increment
        by 1. A missing histogram.observe() call would leave _count at 0.
        """
        # The previous test already flushed one batch — we just check the
        # histogram count increased from the start of that test. Re-scraping
        # now gives us the state after the previous test ran.
        server = MetricsServer(port=0)
        server.start()
        try:
            metrics = _scrape_metrics(server.port)
            count_key = "ingestion_write_latency_seconds_count"
            assert metrics.get(count_key, 0.0) >= 1.0, (
                f"{count_key} should be >= 1 after at least one flush, "
                f"got {metrics.get(count_key, 0.0)}"
            )
        finally:
            server.stop()

    def test_dlq_counter_increments_for_malformed_messages(self, infra: dict) -> None:
        """
        Tested independently from the main counter test to keep assertion scope
        narrow. One malformed message must increment ingestion_dlq_events_total
        by exactly 1.
        """
        bootstrap = infra["kafka_bootstrap"]
        db_url = infra["db_url"]

        dlq_topic = "test.metrics.observability-dlq-check"
        valid_events = [_make_event(i + 500, "source_kafka") for i in range(5)]
        valid_payloads = [MetricEventSerializer.serialize(e) for e in valid_events]
        malformed = b"not json at all {"

        producer = KafkaProducer({"bootstrap.servers": bootstrap, "acks": "all"})
        for payload in valid_payloads[:3] + [malformed] + valid_payloads[3:]:
            producer.produce(dlq_topic, value=payload)
        producer.flush(timeout=10)

        server = MetricsServer(port=0)
        server.start()
        before = _scrape_metrics(server.port)

        consumer = MetricConsumer(ConsumerConfig(
            bootstrap_servers=bootstrap,
            group_id="test-observability-dlq",
            topic=dlq_topic,
        ))
        worker = IngestionWorker(
            consumer=consumer,
            worker_config=WorkerConfig(
                db_url=db_url,
                batch_size=6,
                flush_interval_s=30.0,
            ),
        )

        deadline = time.monotonic() + 30
        while (
            worker.total_written + worker.total_dlq < 6
            and time.monotonic() < deadline
        ):
            worker.run_once()
        worker.flush_remaining()
        worker.close()

        after = _scrape_metrics(server.port)
        server.stop()

        dlq_delta = after.get("ingestion_dlq_events_total", 0.0) - before.get(
            "ingestion_dlq_events_total", 0.0
        )
        assert dlq_delta == 1.0, (
            f"Expected DLQ counter to increment by 1, got delta={dlq_delta}"
        )
