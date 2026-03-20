"""
Integration test for MetricConsumer and IngestionWorker.

I'm running this against real Kafka (KRaft) and real PostgreSQL containers
rather than mocking either because the correctness guarantee we care about is
end-to-end: a message produced to the topic must land in PostgreSQL unless it
is malformed, in which case it must land in the DLQ topic. Mocking the consumer
or the DB session would verify the glue code but not the delivery contract.

The DLQ test specifically exercises the at-least-once guarantee: a batch that
contains one malformed record must not cause the valid records in that batch
to be silently dropped. This is the failure mode that unit tests cannot catch
because it depends on the interaction between offset commit ordering and DB
write ordering.
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
from testcontainers.kafka import KafkaContainer
from testcontainers.postgres import PostgresContainer

from ingestion.consumer import ConsumerConfig, MetricConsumer
from ingestion.serializer import MetricEventSerializer
from ingestion.worker import IngestionWorker, WorkerConfig
from simulator.models import PipelineEvent

KAFKA_IMAGE = "confluentinc/cp-kafka:7.6.0"
POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
# I'm using class-specific topic names rather than a shared SOURCE_TOPIC
# because module-scoped infra means the Kafka broker is shared across all
# test classes. A shared topic causes the DLQ test consumer (which uses
# auto.offset.reset=earliest) to read the happy-path test's 50 messages
# before reaching the 11 DLQ-test messages, producing a wrong total count.
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
    I'm reading from offset 0 on all partitions rather than joining a consumer
    group so the read is reproducible even if the test is re-run without
    restarting the broker. Consumer group offsets would advance on first read
    and miss messages on subsequent runs.

    I'm retrying list_topics up to 5 times because the DLQ topic is
    auto-created by the producer's first write and may not be visible to the
    metadata API immediately. Without a retry, list_topics returns 0 partitions
    and the assign() call leaves the consumer with nothing to read.
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
    I'm starting both containers in a single fixture and yielding a dict of
    connection strings so that the setup cost (two container starts + migration)
    is paid once for the whole module. The alternative — separate fixtures for
    kafka and postgres — would require careful fixture ordering and would make
    the test teardown order non-obvious.
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
    I'm testing with 50 events rather than a larger number because the
    correctness property we're verifying (all events written, offsets
    committed) doesn't require volume — it requires coverage of the
    write → commit ordering. Volume tests belong in load benchmarks, not
    correctness tests.
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
        I'm verifying offset advancement by starting a second consumer on the
        same group_id after the worker closes. If offsets were committed, the
        new consumer should see no messages (they were already consumed). If
        offsets were NOT committed, the new consumer would re-read everything.

        I'm using a 2-second poll window rather than a longer timeout because
        the topic is drained — any messages left would appear immediately.
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
    I'm placing the DLQ test in a separate class with its own group_id so it
    doesn't share offset state with the happy-path test. Shared group state
    would mean the DLQ test consumer starts from where the happy-path consumer
    left off, skipping the valid records we specifically produce here.
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
        I'm reading the DLQ after the routing test has run. The DLQ contains
        the envelope written by MetricConsumer._send_to_dlq — I'm asserting
        the key fields rather than the full envelope so that minor envelope
        format changes don't require updating this test.
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
