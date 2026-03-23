from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from confluent_kafka import KafkaError, Message, Producer

from detection.anomaly import AnomalyEvent
from ingestion.producer import ProducerHealthCheck

# I'm using a dedicated topic for anomaly events rather than a sub-partition of
# pipeline.metrics because the consumer groups for anomaly processing (causal
# analysis, healing engine) are completely different from the ingestion consumer
# group. Mixing both event types on one topic would force each consumer to skip
# irrelevant messages, which adds per-message overhead and complicates
# independent scaling of the two processing paths.
ANOMALY_TOPIC = "pipeline.anomalies"

# I'm pinning the wire format to schema_version=1 using the same convention as
# MetricEventSerializer. The AnomalyPersister checks this version before
# attempting field extraction — a mismatch produces a clear error rather than a
# KeyError on a renamed field.
CURRENT_SCHEMA_VERSION: int = 1


class AnomalySerializationError(Exception):
    """Raised when an anomaly message cannot be deserialized."""


class AnomalyEventSerializer:
    """
    I'm keeping the serializer as a separate class from AnomalyEventBus so that
    AnomalyPersister can deserialize messages without constructing a producer.
    The alternative — embedding serialize/deserialize as AnomalyEventBus methods —
    would force the persister to depend on producer infrastructure (librdkafka
    config, delivery callbacks) just to parse bytes.
    """

    @staticmethod
    def serialize(anomaly: AnomalyEvent) -> bytes:
        """
        I'm encoding detected_at as ISO 8601 with UTC offset for the same reason
        as MetricEventSerializer: isoformat() with a tz-aware datetime always
        includes the +00:00 offset, which is unambiguous across all Python versions
        that handle tz-aware datetimes.
        """
        payload: Dict[str, Any] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "detector_type": anomaly.detector_type,
            "stage_id": anomaly.stage_id,
            "metric": anomaly.metric,
            "signal": anomaly.signal,
            "detector_value": anomaly.detector_value,
            "threshold": anomaly.threshold,
            "z_score": anomaly.z_score,
            "detected_at": anomaly.detected_at.isoformat(),
            "fault_label": anomaly.fault_label,
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def deserialize(data: bytes) -> AnomalyEvent:
        """
        I'm validating schema_version before field extraction for the same
        reason as MetricEventSerializer: a version mismatch should produce an
        actionable error, not a KeyError that hides which field changed.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AnomalySerializationError(
                f"anomaly message is not valid UTF-8 JSON: {exc}"
            ) from exc

        version = payload.get("schema_version")
        if version != CURRENT_SCHEMA_VERSION:
            raise AnomalySerializationError(
                f"unsupported schema_version {version!r}; "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )

        try:
            detected_at_raw = payload["detected_at"]
            detected_at = datetime.fromisoformat(detected_at_raw)
            if detected_at.tzinfo is None:
                detected_at = detected_at.replace(tzinfo=timezone.utc)

            return AnomalyEvent(
                detector_type=payload["detector_type"],
                stage_id=payload["stage_id"],
                metric=payload["metric"],
                signal=payload["signal"],
                detector_value=float(payload["detector_value"]),
                threshold=float(payload["threshold"]),
                z_score=float(payload["z_score"]),
                detected_at=detected_at,
                fault_label=payload.get("fault_label"),
            )
        except KeyError as exc:
            raise AnomalySerializationError(
                f"anomaly message missing required field: {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise AnomalySerializationError(
                f"anomaly message field type error: {exc}"
            ) from exc


class AnomalyEventBus:
    """
    I'm wrapping confluent_kafka.Producer directly here rather than composing a
    RedpandaProducer because RedpandaProducer has a hard-coded MetricEventSerializer
    dependency and no way to inject a different serializer. Extracting a generic
    producer base class would be the right long-term move, but at this stage it
    would add abstraction without a second consumer. Wrapping directly mirrors the
    same durability settings (acks=all, idempotence) at the cost of a small
    duplication that is easy to spot and rectify later.

    I'm reusing ProducerHealthCheck from ingestion.producer because it is a pure
    delivery-callback counter with no serialization dependency — the same counter
    logic works identically for both metric events and anomaly events.
    """

    # I'm using the same durability settings as RedpandaProducer. Anomaly events
    # are the trigger for the healing layer — losing one is as bad as losing a
    # pipeline metric. acks=all + idempotence + retries=5 gives us the same
    # guarantee here that we have on the ingestion path.
    _PRODUCER_CONFIG = {
        "acks": "all",
        "retries": 5,
        "linger.ms": 5,
        "message.timeout.ms": 30_000,
        "enable.idempotence": True,
    }

    def __init__(
        self,
        bootstrap_servers: str,
        health_check: ProducerHealthCheck,
    ) -> None:
        self._health_check = health_check
        config = {
            "bootstrap.servers": bootstrap_servers,
            **self._PRODUCER_CONFIG,
        }
        self._producer = Producer(config)

    def publish(self, anomaly: AnomalyEvent) -> None:
        """
        I'm using stage_id as the message key so all anomalies from the same
        stage land on the same partition. The causal analysis layer groups
        anomalies by stage; having them co-partitioned means a single consumer
        can build a stage's anomaly timeline without cross-partition seeks.
        """
        payload = AnomalyEventSerializer.serialize(anomaly)
        self._producer.produce(
            topic=ANOMALY_TOPIC,
            key=anomaly.stage_id.encode("utf-8"),
            value=payload,
            on_delivery=self._health_check.on_delivery,
        )
        self._producer.poll(0)

    def flush(self, timeout_s: float = 10.0) -> int:
        """Returns count of messages still in-flight after timeout_s."""
        return self._producer.flush(timeout=timeout_s)

    def close(self) -> None:
        """
        I'm flushing explicitly on close for the same reason as RedpandaProducer:
        the librdkafka destructor ordering in Python is non-deterministic and
        buffered anomaly events would be silently dropped at process shutdown.
        """
        remaining = self._producer.flush(timeout=30.0)
        if remaining > 0:
            raise RuntimeError(
                f"AnomalyEventBus closed with {remaining} undelivered messages"
            )
