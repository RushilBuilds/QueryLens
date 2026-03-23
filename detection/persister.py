from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException, Message, TopicPartition
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from detection.anomaly import AnomalyEvent
from detection.bus import ANOMALY_TOPIC, AnomalyEventSerializer, AnomalySerializationError
from ingestion.models import AnomalyEventRow

_log = structlog.get_logger(__name__)


class AnomalyPersister:
    """
    I'm implementing AnomalyPersister as a synchronous pull-and-persist loop
    rather than an async consumer because the detection pipeline is currently
    synchronous (CUSUMDetector and EWMADetector both run on a single thread),
    and adding asyncio here would introduce concurrency without a throughput
    benefit. At the expected anomaly rate (tens per minute, not thousands per
    second), synchronous consumption is the right call.

    I'm committing offsets only after the PostgreSQL write succeeds, using the
    same write-then-commit ordering as IngestionWorker. A crash between write
    and commit replays the batch on restart, producing duplicate row inserts
    that PostgreSQL absorbs by assigning new GENERATED ALWAYS AS IDENTITY ids.
    A commit before write would silently lose anomaly events on crash — that is
    the failure mode we cannot tolerate.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        database_url: str,
        group_id: str = "anomaly-persister",
    ) -> None:
        self._engine = create_engine(database_url, pool_pre_ping=True)

        consumer_conf = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": "false",
            "enable.auto.offset.store": "false",
            "session.timeout.ms": 30_000,
        }
        self._consumer = Consumer(consumer_conf)
        self._consumer.subscribe([ANOMALY_TOPIC])

    def consume_and_persist(
        self,
        timeout_s: float = 5.0,
        max_messages: int = 1_000,
    ) -> int:
        """
        I'm draining up to max_messages from pipeline.anomalies and writing
        them to anomaly_events in a single transaction. The batch write is
        cheaper than one INSERT per message and keeps the offset commit
        horizon bounded: we commit after the full batch lands in Postgres,
        so on crash we replay at most max_messages rather than all uncommitted
        messages since the last group rebalance.

        Returns the number of anomaly rows actually persisted.
        """
        batch: List[AnomalyEvent] = []
        offsets: List[TopicPartition] = []

        # I'm using a blocking poll on the first message and non-blocking polls
        # for the rest of the batch — same pattern as MetricConsumer.poll_batch().
        # This prevents the persister from spinning in a tight loop when the
        # topic is quiet while still draining a full batch quickly when events
        # are available.
        for i in range(max_messages):
            msg: Optional[Message] = self._consumer.poll(
                timeout=timeout_s if i == 0 else 0.0
            )
            if msg is None:
                break
            if msg.error():
                err = msg.error()
                if err.code() == KafkaError._PARTITION_EOF:
                    break
                raise KafkaException(err)

            raw = msg.value()
            try:
                anomaly = AnomalyEventSerializer.deserialize(raw)
                batch.append(anomaly)
                offsets.append(TopicPartition(
                    ANOMALY_TOPIC,
                    msg.partition(),
                    msg.offset() + 1,
                ))
            except AnomalySerializationError as exc:
                # I'm logging and skipping rather than routing to a DLQ because
                # a malformed anomaly event is more likely a serializer bug than
                # a data problem — alerting is more useful than a DLQ replay.
                _log.error(
                    "anomaly_deserialization_error",
                    error=str(exc),
                    partition=msg.partition(),
                    offset=msg.offset(),
                )

        if not batch:
            return 0

        now = datetime.now(tz=timezone.utc)
        rows = [
            AnomalyEventRow(
                stage_id=a.stage_id,
                detector_type=a.detector_type,
                metric=a.metric,
                signal=a.signal,
                detector_value=a.detector_value,
                threshold=a.threshold,
                z_score=a.z_score,
                detected_at=a.detected_at,
                fault_label=a.fault_label,
                schema_version=1,
                created_at=now,
            )
            for a in batch
        ]

        with Session(self._engine) as session:
            session.add_all(rows)
            session.commit()

        # Commit offsets only after the DB write confirms.
        self._consumer.store_offsets(offsets=offsets)
        self._consumer.commit(asynchronous=False)

        _log.info(
            "anomaly_batch_persisted",
            count=len(rows),
            first_stage=batch[0].stage_id,
            last_stage=batch[-1].stage_id,
        )
        return len(rows)

    def close(self) -> None:
        self._consumer.close()
        self._engine.dispose()
