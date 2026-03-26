from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from simulator.models import PipelineEvent

# schema_version=1 pins the wire format so the consumer detects changes without
# a simultaneous deploy. Topic-name versioning (pipeline.metrics.v2) would force
# a new consumer group and full historical replay on every schema change.
CURRENT_SCHEMA_VERSION: int = 1


class SerializationError(Exception):
    """Raised when a message cannot be deserialized into a PipelineEvent."""


class MetricEventSerializer:
    """
    JSON over Avro or MessagePack: no schema registry exists in the current
    infrastructure and throughput (single-digit thousands/sec) is well within
    JSON's envelope. Avro buys compactness at the cost of a registry dependency;
    MessagePack buys ~30% size reduction at the cost of non-human-readable DLQ
    messages. DLQ debuggability matters more than headroom we don't need yet.

    datetime fields serialized as ISO 8601 with UTC offset rather than Unix
    timestamps: Unix precision (seconds vs ms vs us) is an implicit assumption
    that has caused data loss when it drifted between producer and consumer.
    """

    @staticmethod
    def serialize(event: PipelineEvent) -> bytes:
        """
        event_time encoded with explicit UTC offset (+00:00) rather than Z:
        datetime.fromisoformat() did not support the Z suffix until Python 3.11,
        and the consumer may run on 3.10.
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
        schema_version validated before field extraction so a mismatch raises a
        clear SerializationError rather than a KeyError on a missing field. The
        DLQ handler depends on a structured exception to write useful context.
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
