from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException, Message, TopicPartition
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.models import PipelineMetric
from ingestion.serializer import MetricEventSerializer, SerializationError

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ReplayRequest:
    """
    Specifies a contiguous offset range on a single partition to replay.
    One request per partition rather than a multi-partition range because
    partition offsets are independent — a cross-partition range would require
    coordinating seek positions across consumer threads, adding concurrency
    complexity with no benefit for the single-partition replay path.

    replay_rate_limit_rps caps the replay throughput so a recovering stage
    is not re-flooded with the same message volume that caused the original
    failure. Lower values give the stage more time to recover; higher values
    minimise replay lag.
    """

    topic: str
    partition: int
    start_offset: int
    end_offset: int               # inclusive — replay includes this offset
    hypothesis_id: str            # links this replay to the triggering LocalizationResult
    replay_rate_limit_rps: float = 100.0

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError(
                f"start_offset must be >= 0, got {self.start_offset}"
            )
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be >= "
                f"start_offset ({self.start_offset})"
            )
        if self.replay_rate_limit_rps <= 0.0:
            raise ValueError(
                f"replay_rate_limit_rps must be > 0, got {self.replay_rate_limit_rps}"
            )

    @property
    def message_count(self) -> int:
        """Number of messages in the range (inclusive on both ends)."""
        return self.end_offset - self.start_offset + 1


class ReplayOrchestrator:
    """
    Seeks a dedicated consumer to a specific offset range and re-inserts those
    messages into PostgreSQL with replayed=True. Uses a unique group_id per
    instance so replay consumers never affect the normal ingestion consumer
    group's committed offsets.

    Rate-limiting is enforced per-message rather than per-batch: a burst-then-wait
    pattern would defeat the purpose of protecting a recovering stage. Sleeping
    1/rps seconds after each write keeps the throughput steady at exactly rps.

    Does not commit offsets back to Redpanda: replay is a read-only operation
    on the topic. The source messages are not consumed from the normal group's
    perspective — they remain available for further replay if needed.
    """

    def __init__(self, bootstrap_servers: str, database_url: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._serializer = MetricEventSerializer()

    def replay(self, request: ReplayRequest) -> int:
        """
        Replays messages in request's offset range. Returns the count of rows
        successfully written to PostgreSQL. Messages that fail deserialization
        are logged and skipped — a replay must not abort on a single bad record.
        """
        consumer = Consumer({
            "bootstrap.servers": self._bootstrap_servers,
            # Unique group_id: replay consumers must not share offset state with
            # the ingestion consumer group or with each other.
            "group.id": f"replay-{uuid.uuid4().hex}",
            "enable.auto.commit": "false",
            "auto.offset.reset": "earliest",
            "session.timeout.ms": 30_000,
        })

        tp = TopicPartition(request.topic, request.partition, request.start_offset)
        consumer.assign([tp])
        consumer.seek(tp)

        rows: List[PipelineMetric] = []
        interval_s = 1.0 / request.replay_rate_limit_rps

        try:
            while True:
                msg: Optional[Message] = consumer.poll(timeout=5.0)
                if msg is None:
                    break
                if msg.error():
                    err = msg.error()
                    if err.code() == KafkaError._PARTITION_EOF:
                        break
                    raise KafkaException(err)

                if msg.offset() > request.end_offset:
                    break

                try:
                    event = MetricEventSerializer.deserialize(msg.value())
                except SerializationError as exc:
                    _log.warning(
                        "replay_deserialization_error",
                        offset=msg.offset(),
                        error=str(exc),
                        hypothesis_id=request.hypothesis_id,
                    )
                    continue

                rows.append(PipelineMetric(
                    stage_id=event.stage_id,
                    event_time=event.event_time,
                    latency_ms=event.latency_ms,
                    row_count=event.row_count,
                    payload_bytes=event.payload_bytes,
                    status=event.status,
                    fault_label=event.fault_label,
                    trace_id=event.trace_id,
                    replayed=True,
                ))

                time.sleep(interval_s)

                if msg.offset() >= request.end_offset:
                    break
        finally:
            consumer.close()

        if not rows:
            return 0

        with Session(self._engine) as session:
            session.add_all(rows)
            session.commit()

        _log.info(
            "replay_complete",
            hypothesis_id=request.hypothesis_id,
            topic=request.topic,
            partition=request.partition,
            start_offset=request.start_offset,
            end_offset=request.end_offset,
            rows_written=len(rows),
        )
        return len(rows)

    def close(self) -> None:
        self._engine.dispose()
