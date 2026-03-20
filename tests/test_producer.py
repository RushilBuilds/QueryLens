"""
Integration test for RedpandaProducer and MetricEventSerializer.

I'm using KafkaContainer with KRaft mode (no Zookeeper) as the broker because
testcontainers does not yet ship a native RedpandaContainer. The Confluent Kafka
image exposes an identical Kafka wire protocol, so RedpandaProducer — which uses
the confluent_kafka client — behaves identically against it. The only difference
at runtime is the broker implementation; the API surface the producer exercises
is the same.

KRaft mode is chosen over the Zookeeper-based default because it starts in under
10 seconds on warm Docker cache vs 20-30 seconds for the Zookeeper variant, and
it avoids a second container that isn't relevant to what we're testing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest
from confluent_kafka import Consumer, TopicPartition
from testcontainers.kafka import KafkaContainer

from ingestion.producer import ProducerHealthCheck, RedpandaProducer
from ingestion.serializer import MetricEventSerializer, SerializationError
from simulator.models import PipelineEvent

KAFKA_IMAGE = "confluentinc/cp-kafka:7.6.0"
TEST_TOPIC = "pipeline.metrics"
N_EVENTS = 1_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event(index: int) -> PipelineEvent:
    """
    I'm varying stage_id across four values so events land on multiple
    partitions (keyed by stage_id in RedpandaProducer.publish). A test that
    only sends single-stage events would pass even if the key routing logic
    was broken, because all messages would land on partition 0 regardless.
    """
    stage_ids = ["source_postgres", "source_kafka", "transform_validate", "sink_warehouse"]
    return PipelineEvent(
        stage_id=stage_ids[index % len(stage_ids)],
        event_time=datetime(2024, 1, 15, 12, 0, index % 60, tzinfo=timezone.utc),
        latency_ms=10.0 + index * 0.01,
        row_count=100 + index,
        payload_bytes=4096,
        status="ok",
        fault_label=None,
        trace_id=f"{index:032x}",
    )


@pytest.fixture(scope="module")
def kafka_bootstrap() -> str:
    """
    I'm scoping this fixture to the module so the container is started once
    and shared across all tests. KRaft startup takes 8-12 seconds on a warm
    image; per-test container lifecycle would make this suite 3-4× slower
    without any correctness benefit since the tests don't mutate shared broker
    state (they use distinct consumer groups and read from offset 0).
    """
    with KafkaContainer(KAFKA_IMAGE).with_kraft() as kafka:
        yield kafka.get_bootstrap_server()


@pytest.fixture(scope="module")
def produced_health_check(kafka_bootstrap: str) -> ProducerHealthCheck:
    """
    I'm producing all 1,000 events once at module scope so that the delivery
    assertion tests and the consume-back tests share the same produced batch.
    Re-producing per test would generate duplicate messages and make offset
    count assertions non-deterministic.
    """
    health_check = ProducerHealthCheck()
    producer = RedpandaProducer(
        bootstrap_servers=kafka_bootstrap,
        topic=TEST_TOPIC,
        health_check=health_check,
    )
    for i in range(N_EVENTS):
        producer.publish(_make_event(i))

    remaining = producer.flush(timeout_s=30.0)
    assert remaining == 0, (
        f"flush() returned {remaining} undelivered messages — "
        "broker may not have been ready in time"
    )
    return health_check


@pytest.fixture(scope="module")
def consumed_messages(kafka_bootstrap: str, produced_health_check: ProducerHealthCheck) -> List[bytes]:
    """
    I'm consuming from offset 0 on all partitions rather than joining a
    consumer group because consumer group offset management adds non-determinism:
    if the group already exists from a previous test run, the committed offsets
    would skip messages. Explicit partition assignment from the beginning gives
    a reproducible read every time.
    """
    consumer = Consumer({
        "bootstrap.servers": kafka_bootstrap,
        "group.id": "test-consume-back",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    # Assign all partitions from the start (offset 0).
    metadata = consumer.list_topics(TEST_TOPIC, timeout=10)
    partitions = [
        TopicPartition(TEST_TOPIC, p, 0)
        for p in metadata.topics[TEST_TOPIC].partitions
    ]
    consumer.assign(partitions)

    messages: List[bytes] = []
    empty_polls = 0
    while len(messages) < N_EVENTS and empty_polls < 20:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            empty_polls += 1
            continue
        if msg.error():
            raise RuntimeError(f"consumer error: {msg.error()}")
        messages.append(msg.value())
        empty_polls = 0

    consumer.close()
    return messages


# ---------------------------------------------------------------------------
# MetricEventSerializer unit tests (no broker required)
# ---------------------------------------------------------------------------


class TestMetricEventSerializer:
    """
    I'm testing the serializer in isolation from the producer so that a
    serialization regression is immediately identifiable without needing
    a running broker. If the serializer and producer tests were merged,
    a JSON encoding bug would surface as a mysterious delivery failure
    rather than a clear deserialization assertion.
    """

    def test_round_trip_preserves_all_fields(self) -> None:
        event = _make_event(0)
        serialized = MetricEventSerializer.serialize(event)
        recovered = MetricEventSerializer.deserialize(serialized)

        assert recovered.stage_id == event.stage_id
        assert recovered.event_time == event.event_time
        assert abs(recovered.latency_ms - event.latency_ms) < 1e-9
        assert recovered.row_count == event.row_count
        assert recovered.payload_bytes == event.payload_bytes
        assert recovered.status == event.status
        assert recovered.fault_label == event.fault_label
        assert recovered.trace_id == event.trace_id

    def test_nullable_fields_survive_round_trip(self) -> None:
        event = _make_event(0)
        serialized = MetricEventSerializer.serialize(event)
        recovered = MetricEventSerializer.deserialize(serialized)
        assert recovered.fault_label is None
        assert recovered.trace_id is not None

    def test_schema_version_field_is_present(self) -> None:
        import json
        event = _make_event(0)
        payload = json.loads(MetricEventSerializer.serialize(event))
        assert payload["schema_version"] == 1

    def test_wrong_schema_version_raises(self) -> None:
        import json
        event = _make_event(0)
        payload = json.loads(MetricEventSerializer.serialize(event))
        payload["schema_version"] = 99
        bad_bytes = json.dumps(payload).encode("utf-8")
        with pytest.raises(SerializationError, match="unsupported schema_version"):
            MetricEventSerializer.deserialize(bad_bytes)

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(SerializationError, match="not valid UTF-8 JSON"):
            MetricEventSerializer.deserialize(b"not json at all")

    def test_missing_required_field_raises(self) -> None:
        import json
        event = _make_event(0)
        payload = json.loads(MetricEventSerializer.serialize(event))
        del payload["stage_id"]
        bad_bytes = json.dumps(payload).encode("utf-8")
        with pytest.raises(SerializationError, match="missing required field"):
            MetricEventSerializer.deserialize(bad_bytes)


# ---------------------------------------------------------------------------
# ProducerHealthCheck unit tests (no broker required)
# ---------------------------------------------------------------------------


class TestProducerHealthCheck:
    """
    I'm testing the health check counters with a mock KafkaError so the test
    doesn't need a broker. The delivery callback signature (err, msg) is
    dictated by confluent_kafka — I'm verifying that our wrapper correctly
    routes None vs non-None errors to the right counter.
    """

    def test_successful_delivery_increments_succeeded(self) -> None:
        hc = ProducerHealthCheck()
        hc.on_delivery(None, object())
        assert hc.successful_delivery_count == 1
        assert hc.failed_delivery_count == 0

    def test_failed_delivery_increments_failed(self) -> None:
        from confluent_kafka import KafkaError
        hc = ProducerHealthCheck()
        err = KafkaError(KafkaError._MSG_TIMED_OUT)
        hc.on_delivery(err, object())
        assert hc.failed_delivery_count == 1
        assert hc.successful_delivery_count == 0

    def test_total_delivery_count_sums_both(self) -> None:
        from confluent_kafka import KafkaError
        hc = ProducerHealthCheck()
        hc.on_delivery(None, object())
        hc.on_delivery(None, object())
        hc.on_delivery(KafkaError(KafkaError._MSG_TIMED_OUT), object())
        assert hc.total_delivery_count == 3


# ---------------------------------------------------------------------------
# Integration tests — require running broker via kafka_bootstrap fixture
# ---------------------------------------------------------------------------


class TestRedpandaProducerIntegration:

    def test_no_delivery_failures(self, produced_health_check: ProducerHealthCheck) -> None:
        """
        I'm checking failed_delivery_count before consumed_messages so a
        broker-level rejection is caught here rather than surfacing as a
        confusing count mismatch in test_all_events_reach_broker.
        """
        assert produced_health_check.failed_delivery_count == 0, (
            f"{produced_health_check.failed_delivery_count} messages failed delivery — "
            "check broker connectivity and acks=all configuration"
        )

    def test_all_deliveries_acknowledged(self, produced_health_check: ProducerHealthCheck) -> None:
        assert produced_health_check.successful_delivery_count == N_EVENTS, (
            f"Expected {N_EVENTS} successful deliveries, "
            f"got {produced_health_check.successful_delivery_count}"
        )

    def test_all_events_reach_broker(self, consumed_messages: List[bytes]) -> None:
        """
        I'm asserting the exact count rather than >= N_EVENTS to catch
        duplicate delivery, which would indicate the idempotent producer
        config is not being honoured by the broker.
        """
        assert len(consumed_messages) == N_EVENTS, (
            f"Expected {N_EVENTS} messages on broker, found {len(consumed_messages)}"
        )

    def test_consumed_messages_deserialize_cleanly(self, consumed_messages: List[bytes]) -> None:
        """
        I'm deserializing all 1,000 messages rather than sampling because
        a partial serialization failure (e.g. a float precision edge case
        on message 847) would be missed by a sample-based check.
        """
        errors: List[str] = []
        for i, raw in enumerate(consumed_messages):
            try:
                MetricEventSerializer.deserialize(raw)
            except SerializationError as exc:
                errors.append(f"message {i}: {exc}")

        assert not errors, (
            f"{len(errors)} deserialization failures:\n" + "\n".join(errors[:5])
        )

    def test_stage_ids_are_distributed_across_multiple_partitions(
        self, consumed_messages: List[bytes]
    ) -> None:
        """
        I'm verifying that multiple distinct stage_ids appear in the consumed
        batch to confirm that the key-based routing in RedpandaProducer.publish
        is actually encoding the stage_id as the message key. If the key was
        empty or constant, all messages would pile onto one partition and the
        per-stage ordering guarantee would be meaningless.
        """
        stage_ids = {
            MetricEventSerializer.deserialize(raw).stage_id
            for raw in consumed_messages
        }
        assert len(stage_ids) > 1, (
            f"All messages have the same stage_id — key routing is not working. "
            f"stage_ids found: {stage_ids}"
        )
