"""
Integration test for ReplayOrchestrator against real Redpanda and PostgreSQL containers.

Test flow:
  1. Produce N messages to pipeline.metrics
  2. Record the exact partition/offset range
  3. Run ReplayOrchestrator over that range
  4. Assert N rows with replayed=True land in pipeline_metrics
  5. Assert original ingestion rows (replayed=False) are unaffected

The replayed=False assertion is important: the orchestrator must not touch
existing rows or commit offsets that would interfere with the ingestion consumer.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from confluent_kafka import Consumer, Producer, TopicPartition
from sqlalchemy import create_engine, text

from healing.replay import ReplayOrchestrator, ReplayRequest
from ingestion.producer import ProducerHealthCheck
from ingestion.serializer import MetricEventSerializer
from simulator.models import PipelineEvent

try:
    from testcontainers.kafka import KafkaContainer
    from testcontainers.postgres import PostgresContainer
    _CONTAINERS_AVAILABLE = True
except ImportError:
    _CONTAINERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CONTAINERS_AVAILABLE,
    reason="testcontainers not installed",
)

KAFKA_IMAGE  = "confluentinc/cp-kafka:7.6.0"
POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI  = Path(__file__).parent.parent / "alembic.ini"
TOPIC        = "pipeline.metrics"
SIM_START    = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _make_event(i: int) -> PipelineEvent:
    from datetime import timedelta
    return PipelineEvent(
        stage_id="source_pg",
        event_time=SIM_START + timedelta(seconds=i),
        latency_ms=10.0 + i,
        row_count=100,
        payload_bytes=512,
        status="ok",
        fault_label=None,
        trace_id=None,
    )


def _produce_messages(broker: str, n: int) -> None:
    """Produces n messages synchronously and flushes."""
    health = ProducerHealthCheck()
    from confluent_kafka import Producer as _Producer
    producer = _Producer({
        "bootstrap.servers": broker,
        "acks": "all",
        "retries": 3,
    })

    serializer = MetricEventSerializer()
    for i in range(n):
        event = _make_event(i)
        producer.produce(
            topic=TOPIC,
            value=serializer.serialize(event),
            on_delivery=health.on_delivery,
        )
    producer.flush(timeout=30)


def _get_latest_offsets(broker: str, partition: int = 0) -> tuple[int, int]:
    """
    Returns (start_offset, end_offset) of the messages currently on the topic.
    Uses a temporary consumer to fetch watermarks.
    """
    c = Consumer({
        "bootstrap.servers": broker,
        "group.id": f"offset-probe-{uuid.uuid4().hex}",
        "auto.offset.reset": "earliest",
    })
    tp = TopicPartition(TOPIC, partition)
    low, high = c.get_watermark_offsets(tp, timeout=10)
    c.close()
    # high watermark is the next offset to be written (exclusive)
    return low, high - 1


def _ensure_topic(broker: str) -> None:
    """Trigger topic auto-creation by producing one throw-away message."""
    p = Producer({"bootstrap.servers": broker})
    p.produce(TOPIC, b"init")
    p.flush(10)
    time.sleep(1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def broker():
    with KafkaContainer(KAFKA_IMAGE) as kafka:
        servers = kafka.get_bootstrap_server()
        _ensure_topic(servers)
        yield servers


@pytest.fixture(scope="module")
def db_url():
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        url = pg.get_connection_url().replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
        _run_migrations(url)
        yield url


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReplayOrchestratorIntegration:

    def test_replayed_rows_land_in_postgres(
        self, broker: str, db_url: str
    ) -> None:
        n = 5
        # Capture high watermark before producing so we target exactly the
        # n messages we write, not any earlier init messages on the topic.
        _, pre_high = _get_latest_offsets(broker)
        start_offset = pre_high + 1

        _produce_messages(broker, n)
        time.sleep(1.0)

        _, post_high = _get_latest_offsets(broker)
        end_offset = post_high

        assert end_offset >= start_offset

        request = ReplayRequest(
            topic=TOPIC,
            partition=0,
            start_offset=start_offset,
            end_offset=end_offset,
            hypothesis_id="hyp-replay-001",
            replay_rate_limit_rps=500.0,  # fast in tests
        )

        orchestrator = ReplayOrchestrator(broker, db_url)
        written = orchestrator.replay(request)
        orchestrator.close()

        assert written == request.message_count

        engine = create_engine(db_url)
        with engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM pipeline_metrics WHERE replayed = TRUE"
                )
            ).scalar()
        engine.dispose()

        assert count == written

    def test_replayed_rows_have_correct_stage_id(
        self, broker: str, db_url: str
    ) -> None:
        _produce_messages(broker, 3)
        time.sleep(1.0)

        start_offset, end_offset = _get_latest_offsets(broker)
        request = ReplayRequest(
            topic=TOPIC,
            partition=0,
            start_offset=start_offset,
            end_offset=end_offset,
            hypothesis_id="hyp-replay-002",
            replay_rate_limit_rps=500.0,
        )

        orchestrator = ReplayOrchestrator(broker, db_url)
        orchestrator.replay(request)
        orchestrator.close()

        engine = create_engine(db_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT stage_id FROM pipeline_metrics "
                    "WHERE replayed = TRUE"
                )
            ).fetchall()
        engine.dispose()

        stage_ids = {r.stage_id for r in rows}
        assert "source_pg" in stage_ids

    def test_replay_does_not_affect_original_rows(
        self, broker: str, db_url: str
    ) -> None:
        """
        Original rows written by the normal ingestion path (replayed=FALSE by
        default) must not be touched by the ReplayOrchestrator.
        """
        engine = create_engine(db_url)

        # Count existing non-replayed rows before replay.
        with engine.connect() as conn:
            before = conn.execute(
                text(
                    "SELECT COUNT(*) FROM pipeline_metrics WHERE replayed = FALSE"
                )
            ).scalar()

        _produce_messages(broker, 2)
        time.sleep(1.0)
        start_offset, end_offset = _get_latest_offsets(broker)
        request = ReplayRequest(
            topic=TOPIC,
            partition=0,
            start_offset=start_offset,
            end_offset=end_offset,
            hypothesis_id="hyp-replay-003",
            replay_rate_limit_rps=500.0,
        )
        orchestrator = ReplayOrchestrator(broker, db_url)
        orchestrator.replay(request)
        orchestrator.close()

        with engine.connect() as conn:
            after = conn.execute(
                text(
                    "SELECT COUNT(*) FROM pipeline_metrics WHERE replayed = FALSE"
                )
            ).scalar()
        engine.dispose()

        # Non-replayed count unchanged — orchestrator only inserts replayed=TRUE rows.
        assert after == before

    def test_replay_request_message_count(self) -> None:
        r = ReplayRequest(
            topic=TOPIC, partition=0,
            start_offset=10, end_offset=19,
            hypothesis_id="hyp-count",
        )
        assert r.message_count == 10

    def test_replay_request_rejects_negative_start_offset(self) -> None:
        with pytest.raises(ValueError, match="start_offset"):
            ReplayRequest(
                topic=TOPIC, partition=0,
                start_offset=-1, end_offset=5,
                hypothesis_id="hyp-bad",
            )

    def test_replay_request_rejects_end_before_start(self) -> None:
        with pytest.raises(ValueError, match="end_offset"):
            ReplayRequest(
                topic=TOPIC, partition=0,
                start_offset=10, end_offset=5,
                hypothesis_id="hyp-bad",
            )

    def test_replay_request_rejects_zero_rate_limit(self) -> None:
        with pytest.raises(ValueError, match="replay_rate_limit_rps"):
            ReplayRequest(
                topic=TOPIC, partition=0,
                start_offset=0, end_offset=5,
                hypothesis_id="hyp-bad",
                replay_rate_limit_rps=0.0,
            )
