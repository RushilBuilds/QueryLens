from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from simulator.models import PipelineEvent

# I'm pinning the current wire format to schema_version=1 so the consumer can
# detect and handle format changes without coordinating a simultaneous deploy.
# The alternative — versioning via Kafka topic name (pipeline.metrics.v2) —
# forces a new consumer group and replay of all historical data every time the
# schema evolves. Field-level versioning lets the consumer decide whether it
# can handle an older or newer version without topic proliferation.
CURRENT_SCHEMA_VERSION: int = 1


class SerializationError(Exception):
    """Raised when a message cannot be deserialized into a PipelineEvent."""


class MetricEventSerializer:
    """
    I'm using JSON rather than Avro or MessagePack for the wire format because
    we have no schema registry in the current infrastructure and the event
    throughput (single-digit thousands per second) is well within JSON's
    performance envelope. Avro would buy binary compactness and schema
    enforcement at the cost of a registry dependency; MessagePack buys ~30%
    size reduction at the cost of non-human-readable DLQ messages. Both
    trade-offs are wrong at this scale — DLQ debuggability matters more than
    throughput headroom we don't need yet.

    datetime fields are serialized as ISO 8601 strings with UTC offset rather
    than Unix timestamps. Unix timestamps are unambiguous but require the reader
    to know the expected precision (seconds vs milliseconds vs microseconds),
    which has caused data loss bugs in the past when the precision assumption
    drifted between producer and consumer versions.
    """

    @staticmethod
    def serialize(event: PipelineEvent) -> bytes:
        """
        I'm encoding event_time with explicit UTC offset (+00:00) rather than
        relying on the Z suffix because Python's datetime.fromisoformat() did
        not support the Z suffix until 3.11, and we may run this consumer on
        3.10 in some environments. The +00:00 form is unambiguous across all
        Python versions that support timezone-aware datetimes.
        """
        payload: Dict[str, Any] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "stage_id": event.stage_id,
            "event_time": event.event_time.isoformat(),
            "latency_ms": event.latency_ms,
            "row_count": event.row_count,
            "payload_bytes": event.payload_bytes,
            "status": event.status,
            "fault_label": event.fault_label,
            "trace_id": event.trace_id,
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def deserialize(data: bytes) -> PipelineEvent:
        """
        I'm validating schema_version before attempting field extraction so
        that a version mismatch produces a clear SerializationError with the
        actual version number rather than a KeyError on a field that no longer
        exists. The consumer's dead-letter handler depends on a structured
        exception to write useful context to the DLQ message.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SerializationError(f"message is not valid UTF-8 JSON: {exc}") from exc

        version = payload.get("schema_version")
        if version != CURRENT_SCHEMA_VERSION:
            raise SerializationError(
                f"unsupported schema_version {version!r}; "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )

        try:
            event_time_raw = payload["event_time"]
            event_time = datetime.fromisoformat(event_time_raw)
            # Ensure timezone-aware even if the producer omitted the offset.
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)

            return PipelineEvent(
                stage_id=payload["stage_id"],
                event_time=event_time,
                latency_ms=float(payload["latency_ms"]),
                row_count=int(payload["row_count"]),
                payload_bytes=int(payload["payload_bytes"]),
                status=payload["status"],
                fault_label=payload.get("fault_label"),
                trace_id=payload.get("trace_id"),
            )
        except KeyError as exc:
            raise SerializationError(f"missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise SerializationError(f"field type error: {exc}") from exc
