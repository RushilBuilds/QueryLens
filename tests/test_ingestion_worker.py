"""
Integration test for MetricConsumer and IngestionWorker.

Run against real Kafka (KRaft) and real PostgreSQL containers rather than mocks:
the guarantee being verified is end-to-end delivery — valid messages land in
PostgreSQL, malformed messages land in the DLQ. Mocking either layer would
verify glue code but not the delivery contract.

The DLQ test exercises the at-least-once guarantee: a batch with one malformed
record must not silently drop the valid records. This failure mode depends on the
interaction between offset commit ordering and DB write ordering — unit tests
cannot catch it.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest
from alembic import command
from alembic.config import Config
from confluent_kafka import Consumer as KafkaConsumer
from confluent_kafka import Producer as KafkaProducer
from confluent_kafka import TopicPartition
from sqlalchemy import create_engine, text

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
from ingestion.serializer import MetricEventSerializer
from ingestion.worker import IngestionWorker, WorkerConfig
from simulator.models import PipelineEvent

KAFKA_IMAGE = "confluentinc/cp-kafka:7.6.0"
POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
# Class-specific topic names rather than a shared SOURCE_TOPIC: the broker is
# shared across test classes (module-scoped infra). A shared topic would cause
# the DLQ test consumer (auto.offset.reset=earliest) to read the happy-path's
# 50 messages before the 11 DLQ-test messages, producing a wrong total count.
TOPIC_HAPPY = "test.metrics.happy"
TOPIC_DLQ_INPUT = "test.metrics.dlq-input"
DLQ_SUFFIX = ".dlq"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(index: int) -> PipelineEvent:
    return PipelineEvent(
        stage_id="source_postgres" if index % 2 == 0 else "transform_validate",
        event_time=datetime(2024, 1, 15, 12, 0, index % 60, tzinfo=timezone.utc),
        latency_ms=20.0 + index,
        row_count=100,
        payload_bytes=2048,
        status="ok",
        fault_label=None,
        trace_id=f"{index:032x}",
    )


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _produce_messages(bootstrap: str, topic: str, payloads: List[bytes]) -> None:
    """Produce a list of raw byte payloads synchronously."""
    producer = KafkaProducer({"bootstrap.servers": bootstrap, "acks": "all"})
    for payload in payloads:
        producer.produce(topic, value=payload)
    producer.flush(timeout=15)


def _consume_all_from_dlq(bootstrap: str, dlq_topic: str, expected_count: int) -> List[dict]:
    """
    Reads from offset 0 on all partitions rather than joining a consumer group,
    so the read is reproducible across test re-runs without restarting the broker.

    list_topics retried up to 10 times: the DLQ topic is auto-created on first
    write and may not be visible to the metadata API immediately.
    """
    consumer = KafkaConsumer({
        "bootstrap.servers": bootstrap,
        "group.id": f"test-dlq-reader-{dlq_topic}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    partitions = []
    for _ in range(10):
        metadata = consumer.list_topics(dlq_topic, timeout=3)
        topic_meta = metadata.topics.get(dlq_topic)
        if topic_meta and topic_meta.partitions:
            partitions = [TopicPartition(dlq_topic, p, 0) for p in topic_meta.partitions]
            break
        time.sleep(0.5)

    if not partitions:
        consumer.close()
        return []

    consumer.assign(partitions)

    messages = []
    empty_polls = 0
    while len(messages) < expected_count and empty_polls < 15:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            empty_polls += 1
            continue
        if msg.error():
            raise RuntimeError(f"DLQ consumer error: {msg.error()}")
        messages.append(json.loads(msg.value().decode("utf-8")))
        empty_polls = 0

    consumer.close()
    return messages


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def infra():
    """
    Both containers started in a single fixture so setup cost (two starts +
    migration) is paid once per module. Separate fixtures would require careful
    ordering and make teardown order non-obvious.
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
# WorkerConfig validation unit tests (no containers needed)
# ---------------------------------------------------------------------------


class TestWorkerConfig:

    def test_rejects_zero_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            WorkerConfig(db_url="postgresql://x", batch_size=0)

    def test_rejects_negative_flush_interval(self) -> None:
        with pytest.raises(ValueError, match="flush_interval_s"):
            WorkerConfig(db_url="postgresql://x", flush_interval_s=-1.0)


# ---------------------------------------------------------------------------
# End-to-end: valid events land in PostgreSQL
# ---------------------------------------------------------------------------


class TestIngestionWorkerHappyPath:
    """
    50 events is sufficient: the correctness property (all events written,
    offsets committed) requires coverage of write → commit ordering, not
    volume. Volume tests belong in load benchmarks.
    """

    N_EVENTS = 50

    def test_valid_events_land_in_postgres(self, infra: dict) -> None:
        bootstrap = infra["kafka_bootstrap"]
        db_url = infra["db_url"]

        # Produce N valid events.
        payloads = [
            MetricEventSerializer.serialize(_make_event(i))
            for i in range(self.N_EVENTS)
        ]
        _produce_messages(bootstrap, TOPIC_HAPPY, payloads)

        consumer_cfg = ConsumerConfig(
            bootstrap_servers=bootstrap,
            group_id="test-happy-path",
            topic=TOPIC_HAPPY,
        )
        worker_cfg = WorkerConfig(
            db_url=db_url,
            batch_size=self.N_EVENTS,
            flush_interval_s=30.0,  # high — we'll trigger via batch_size
        )
        consumer = MetricConsumer(consumer_cfg)
        worker = IngestionWorker(consumer=consumer, worker_config=worker_cfg)

        # Drive the worker until it has consumed and flushed all events.
        deadline = time.monotonic() + 30
        while worker.total_written < self.N_EVENTS and time.monotonic() < deadline:
            worker.run_once()
        worker.flush_remaining()
        worker.close()

        assert worker.total_written == self.N_EVENTS, (
            f"Expected {self.N_EVENTS} rows written, got {worker.total_written}"
        )
        assert worker.total_dlq == 0

        engine = create_engine(db_url)
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM pipeline_metrics WHERE stage_id IN "
                     "('source_postgres', 'transform_validate')")
            ).scalar()
        engine.dispose()

        assert count == self.N_EVENTS, (
            f"Expected {self.N_EVENTS} rows in pipeline_metrics, found {count}"
        )

    def test_offset_advances_after_write(self, infra: dict) -> None:
        """
        A second consumer on the same group_id after the worker closes must see
        no messages if offsets were committed. A 2-second poll window is enough —
        the topic is drained and any remaining messages would appear immediately.
        """
        bootstrap = infra["kafka_bootstrap"]

        # The happy-path test already consumed all messages on group "test-happy-path".
        probe = KafkaConsumer({
            "bootstrap.servers": bootstrap,
            "group.id": "test-happy-path",  # same group
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        })
        probe.subscribe([TOPIC_HAPPY])
        msg = probe.poll(timeout=2.0)
        probe.close()

        assert msg is None, (
            "Consumer group still has uncommitted messages — "
            "offset commit did not happen after DB write"
        )


# ---------------------------------------------------------------------------
# End-to-end: malformed record goes to DLQ, valid records go to PostgreSQL
# ---------------------------------------------------------------------------


class TestDLQRouting:
    """
    Separate class with its own group_id to avoid sharing offset state with the
    happy-path test. Shared group state would skip the valid records produced
    here by starting from where the happy-path consumer left off.
    """

    N_VALID = 10
    N_MALFORMED = 1

    def test_malformed_record_lands_in_dlq(self, infra: dict) -> None:
        bootstrap = infra["kafka_bootstrap"]
        db_url = infra["db_url"]

        valid_payloads = [
            MetricEventSerializer.serialize(_make_event(i + 200))
            for i in range(self.N_VALID)
        ]
        malformed_payload = b"this is not valid json {"

        # Interleave: 5 valid, 1 malformed, 5 valid.
        payloads = (
            valid_payloads[:5]
            + [malformed_payload]
            + valid_payloads[5:]
        )
        _produce_messages(bootstrap, TOPIC_DLQ_INPUT, payloads)

        consumer_cfg = ConsumerConfig(
            bootstrap_servers=bootstrap,
            group_id="test-dlq-routing",
            topic=TOPIC_DLQ_INPUT,
        )
        worker_cfg = WorkerConfig(
            db_url=db_url,
            batch_size=self.N_VALID + self.N_MALFORMED,
            flush_interval_s=30.0,
        )
        consumer = MetricConsumer(consumer_cfg)
        worker = IngestionWorker(consumer=consumer, worker_config=worker_cfg)

        deadline = time.monotonic() + 30
        while (
            worker.total_written + worker.total_dlq
            < self.N_VALID + self.N_MALFORMED
            and time.monotonic() < deadline
        ):
            worker.run_once()
        worker.flush_remaining()
        worker.close()

        assert worker.total_written == self.N_VALID, (
            f"Expected {self.N_VALID} valid rows written, got {worker.total_written}"
        )
        assert worker.total_dlq == self.N_MALFORMED, (
            f"Expected {self.N_MALFORMED} DLQ record, got {worker.total_dlq}"
        )

    def test_dlq_message_contains_error_context(self, infra: dict) -> None:
        """
        Reads the DLQ after the routing test has run. Key fields only are
        asserted rather than the full envelope — minor format changes should
        not require updating this test.
        """
        bootstrap = infra["kafka_bootstrap"]
        dlq_messages = _consume_all_from_dlq(
            bootstrap, TOPIC_DLQ_INPUT + DLQ_SUFFIX, expected_count=1
        )

        assert len(dlq_messages) >= 1, "No messages found in DLQ topic"
        envelope = dlq_messages[0]

        assert envelope["source_topic"] == TOPIC_DLQ_INPUT
        assert "error" in envelope
        assert "raw_payload" in envelope
        assert "source_partition" in envelope
        assert "source_offset" in envelope
