from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Tuple

import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException, Message, Producer

from ingestion.observability import DLQ_EVENTS
from ingestion.serializer import MetricEventSerializer, SerializationError
from simulator.models import PipelineEvent

_log = structlog.get_logger(__name__)

# DLQ topic receives any message that fails deserialization or schema validation.
# Fixed suffix (.dlq) rather than a configurable value: the DLQ topic must always
# be derivable from the source topic name. A configurable name could silently
# discard bad messages to /dev/null on misconfiguration.
DLQ_TOPIC_SUFFIX = ".dlq"


@dataclass(frozen=True)
class ConsumerConfig:
    """
    Named fields rather than a raw librdkafka config dict. auto.offset.reset
    and enable.auto.commit are easy to misconfigure silently — explicit named
    fields with documented defaults make the intent clear.
    """

    bootstrap_servers: str
    group_id: str
    topic: str
    auto_offset_reset: str = "earliest"
    # Auto-commit disabled unconditionally. Manual commit is the only way to
    # guarantee at-least-once delivery: offset advances only after the batch
    # lands in PostgreSQL. Auto-commit on a timer would turn transient DB
    # failures into permanent data loss.
    enable_auto_commit: bool = False
    session_timeout_ms: int = 30_000
    max_poll_interval_ms: int = 300_000

    @property
    def dlq_topic(self) -> str:
        return self.topic + DLQ_TOPIC_SUFFIX


@dataclass
class ConsumedMessage:
    """
    raw_bytes retained alongside the parsed event so the DLQ writer forwards
    the original payload. Re-serializing a partially-parsed event would lose
    fields that failed to parse — the DLQ value must be exactly what the
    producer sent.
    """

    raw_bytes: bytes
    event: Optional[PipelineEvent]
    parse_error: Optional[str]
    partition: int
    offset: int

    @property
    def is_valid(self) -> bool:
        return self.event is not None and self.parse_error is None


class MetricConsumer:
    """
    Manual offset commit (store + commit after batch write) rather than
    auto-commit. Auto-commit advances the offset on a wall-clock timer
    independent of write success; a DB failure followed by auto-commit skips
    those records permanently on restart. Manual commit gives at-least-once
    delivery: uncommitted batches replay from the last committed offset.

    Synchronous rather than async: confluent_kafka.Consumer is a C extension.
    run_in_executor would add thread pool overhead with no throughput benefit
    at this event rate.
    """

    def __init__(self, config: ConsumerConfig) -> None:
        self._config = config
        self._serializer = MetricEventSerializer()

        consumer_conf = {
            "bootstrap.servers": config.bootstrap_servers,
            "group.id": config.group_id,
            "auto.offset.reset": config.auto_offset_reset,
            "enable.auto.commit": "false",
            "session.timeout.ms": config.session_timeout_ms,
            "max.poll.interval.ms": config.max_poll_interval_ms,
            # Store offsets manually so we control exactly when they advance.
            "enable.auto.offset.store": "false",
        }
        self._consumer = Consumer(consumer_conf)
        self._consumer.subscribe([config.topic])

        # Separate Producer for DLQ writes: acks=1 is acceptable (DLQ messages
        # are diagnostic, not business data), and fire-and-forget semantics
        # prevent DLQ write failures from blocking the main consume loop.
        self._dlq_producer = Producer({
            "bootstrap.servers": config.bootstrap_servers,
            "acks": 1,
            "linger.ms": 0,
        })

    def poll_batch(self, max_records: int, timeout_s: float = 1.0) -> List[ConsumedMessage]:
        """
        Polls for up to max_records messages (a single poll() returns at most
        one). Batching amortises the PostgreSQL INSERT overhead: 500 records in
        one round trip instead of 500. Short timeout (1.0s default) lets
        flush_interval_s fire even when the topic is quiet.
        """
        batch: List[ConsumedMessage] = []
        while len(batch) < max_records:
            msg: Optional[Message] = self._consumer.poll(timeout=timeout_s if not batch else 0.0)
            if msg is None:
                break
            if msg.error():
                err = msg.error()
                if err.code() == KafkaError._PARTITION_EOF:
                    break
                raise KafkaException(err)

            raw = msg.value()
            try:
                event = MetricEventSerializer.deserialize(raw)
                batch.append(ConsumedMessage(
                    raw_bytes=raw,
                    event=event,
                    parse_error=None,
                    partition=msg.partition(),
                    offset=msg.offset(),
                ))
            except SerializationError as exc:
                self._send_to_dlq(raw, str(exc), msg.partition(), msg.offset())
                batch.append(ConsumedMessage(
                    raw_bytes=raw,
                    event=None,
                    parse_error=str(exc),
                    partition=msg.partition(),
                    offset=msg.offset(),
                ))

        valid_count = sum(1 for m in batch if m.is_valid)
        dlq_count = len(batch) - valid_count
        if batch:
            _log.info(
                "poll_batch_complete",
                batch_size=len(batch),
                valid=valid_count,
                dlq=dlq_count,
                first_offset=batch[0].offset,
                last_offset=batch[-1].offset,
            )

        return batch

    def _send_to_dlq(
        self, raw_bytes: bytes, error_context: str, partition: int, offset: int
    ) -> None:
        """
        Wraps the original payload in a DLQ envelope rather than forwarding it
        bare. The envelope includes source partition and offset so the original
        message can be located in the source topic for manual inspection.
        """
        envelope = json.dumps({
            "source_topic": self._config.topic,
            "source_partition": partition,
            "source_offset": offset,
            "error": error_context,
            "received_at": datetime.now(tz=timezone.utc).isoformat(),
            "raw_payload": raw_bytes.decode("utf-8", errors="replace"),
        }, separators=(",", ":")).encode("utf-8")

        self._dlq_producer.produce(
            topic=self._config.dlq_topic,
            value=envelope,
        )
        self._dlq_producer.poll(0)
        DLQ_EVENTS.inc()
        _log.warning(
            "dlq_routed",
            source_partition=partition,
            source_offset=offset,
            error=error_context[:200],
        )

    def commit_batch(self, batch: List[ConsumedMessage]) -> None:
        """
        Stores offsets for all messages — valid and invalid alike. Invalid
        messages already went to the DLQ and must not replay on restart.

        store_offsets() marks offsets locally without a broker round-trip;
        commit() syncs all stored offsets in one request at the end of the batch.
        """
        for msg_meta in batch:
            from confluent_kafka import TopicPartition
            self._consumer.store_offsets(offsets=[
                TopicPartition(
                    self._config.topic,
                    msg_meta.partition,
                    msg_meta.offset + 1,  # +1: commit the NEXT offset to consume
                )
            ])
        self._consumer.commit(asynchronous=False)
        _log.debug("offsets_committed", batch_size=len(batch))

    def flush_dlq(self, timeout_s: float = 5.0) -> None:
        """Flush any pending DLQ messages before closing."""
        self._dlq_producer.flush(timeout=timeout_s)

    def close(self) -> None:
        self.flush_dlq()
        self._consumer.close()
