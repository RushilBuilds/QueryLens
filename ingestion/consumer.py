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
# I'm using a fixed suffix convention (.dlq) rather than a separate config
# value because the DLQ topic name must always be derivable from the source
# topic name — if it were configurable, a misconfiguration could silently
# discard bad messages to /dev/null.
DLQ_TOPIC_SUFFIX = ".dlq"


@dataclass(frozen=True)
class ConsumerConfig:
    """
    I'm grouping consumer tunables here rather than accepting a raw dict of
    librdkafka config strings because the raw dict makes it impossible to
    know at a glance which settings we actually care about. The important ones
    (auto.offset.reset, enable.auto.commit) are particularly easy to
    misconfigure silently — having them as named fields with documented
    defaults makes the intent explicit.
    """

    bootstrap_servers: str
    group_id: str
    topic: str
    auto_offset_reset: str = "earliest"
    # I'm disabling auto-commit unconditionally. Manual commit is the only way
    # to guarantee at-least-once delivery: we commit the offset only AFTER the
    # batch has been written to PostgreSQL. Auto-commit would advance the offset
    # on a timer regardless of whether the write succeeded, turning transient
    # DB failures into permanent data loss.
    enable_auto_commit: bool = False
    session_timeout_ms: int = 30_000
    max_poll_interval_ms: int = 300_000

    @property
    def dlq_topic(self) -> str:
        return self.topic + DLQ_TOPIC_SUFFIX


@dataclass
class ConsumedMessage:
    """
    I'm keeping raw_bytes alongside the parsed event so the DLQ writer can
    forward the original payload without re-serializing. Re-serializing a
    partially-parsed event would lose information from fields we failed to
    parse — the DLQ value must be exactly what the producer sent.
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
    I'm implementing manual offset commit (store + commit after batch write)
    rather than auto-commit because auto-commit advances the offset on a
    wall-clock timer independent of write success. A PostgreSQL write failure
    followed by an auto-commit would cause those records to be skipped on
    restart — the consumer would resume from the committed offset, never
    retrying the failed batch. Manual commit gives us the at-least-once
    guarantee: on crash-restart, the uncommitted batch is replayed from the
    last committed offset.

    I'm not using async/await here even though the IngestionWorker is async
    because confluent_kafka.Consumer is a synchronous C extension. Wrapping
    it in run_in_executor would add thread pool overhead and complicate error
    handling without any throughput benefit at our event rate. The worker runs
    the consumer poll in a sync loop and awaits only the DB write.
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

        # I'm using a separate Producer for DLQ writes rather than reusing the
        # main RedpandaProducer because the DLQ producer needs different settings:
        # acks=1 is acceptable here (DLQ messages are diagnostic, not business
        # data), and we want fire-and-forget semantics to avoid blocking the
        # main consume loop on DLQ write failures.
        self._dlq_producer = Producer({
            "bootstrap.servers": config.bootstrap_servers,
            "acks": 1,
            "linger.ms": 0,
        })

    def poll_batch(self, max_records: int, timeout_s: float = 1.0) -> List[ConsumedMessage]:
        """
        I'm polling for up to max_records messages rather than polling once
        because a single poll() call returns at most one message. Batching
        here amortises the per-batch PostgreSQL INSERT overhead: a 500-record
        batch uses one round trip instead of 500.

        poll() with a non-zero timeout blocks until a message arrives or the
        timeout elapses. I'm using a short timeout (1.0s default) so the
        IngestionWorker's flush_interval_s timer can fire even when the topic
        is quiet.
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
        I'm wrapping the original payload in a DLQ envelope rather than
        forwarding it bare so the DLQ consumer knows where the message came
        from and why it was rejected. The envelope includes the source
        partition and offset so the original message can be located in the
        source topic for manual inspection or replay.
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
        I'm storing offsets for all messages in the batch (valid and invalid
        alike) before committing. The DLQ already received the invalid
        messages, so we do not want to replay them on restart — their offset
        should advance just like a valid message's offset.

        store_offsets() marks the offset for the next commit without actually
        talking to the broker. commit() then syncs all stored offsets to the
        broker in one request. This two-phase approach lets us store offsets
        for every message as we process them, then commit once at the end of
        the batch rather than issuing a broker request per message.
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
