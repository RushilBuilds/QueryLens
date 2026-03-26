from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from confluent_kafka import KafkaError, Message, Producer

from ingestion.serializer import MetricEventSerializer
from simulator.models import PipelineEvent


@dataclass
class ProducerHealthCheck:
    """
    Thread-safe running total rather than a list of failed Message objects.
    Storing full messages would grow unbounded on a sustained fault and require
    a separate eviction policy; the counter gives Prometheus what it needs at
    O(1) memory cost.

    Lock protects _failed and _succeeded from concurrent writes by the
    librdkafka delivery callback thread. Without the lock, the increment is
    subject to lost-update races on CPython despite the GIL.
    """

    _failed: int = field(default=0, init=False, repr=False)
    _succeeded: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def on_delivery(self, err: Optional[KafkaError], msg: Message) -> None:
        """
        Delivery callback invoked by librdkafka's internal thread. Per-message
        logging is omitted: at 5,000 events/sec a log line per failure would
        saturate the log pipeline before the alert fires. The Prometheus counter
        is the right signal; structured logs belong at the batch level.
        """
        with self._lock:
            if err is not None:
                self._failed += 1
            else:
                self._succeeded += 1

    @property
    def failed_delivery_count(self) -> int:
        with self._lock:
            return self._failed

    @property
    def successful_delivery_count(self) -> int:
        with self._lock:
            return self._succeeded

    @property
    def total_delivery_count(self) -> int:
        with self._lock:
            return self._failed + self._succeeded


class RedpandaProducer:
    """
    Wraps confluent_kafka.Producer to enforce durability settings (acks=all,
    retries=5) at construction time. The raw API exposes librdkafka config
    strings that are easy to misconfigure silently.

    poll(0) called after every produce() because delivery callbacks only fire
    during poll calls. Without per-produce polling, failed_delivery_count would
    read zero until flush() completed, making it useless as a real-time signal.
    """

    # linger.ms=5 batches messages within 5ms into a single request, halving
    # round trips at 5,000 events/sec. The 5ms latency is invisible at detection
    # window granularity (seconds).
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
        topic: str,
        health_check: ProducerHealthCheck,
    ) -> None:
        self._topic = topic
        self._health_check = health_check
        self._serializer = MetricEventSerializer()

        config = {
            "bootstrap.servers": bootstrap_servers,
            **self._PRODUCER_CONFIG,
        }
        # enable.idempotence=True with acks=all and retries=5 guarantees
        # exactly-once delivery at the producer level. Without idempotence,
        # retries can duplicate messages when the ack is lost in transit.
        self._producer = Producer(config)

    def publish(self, event: PipelineEvent) -> None:
        """
        stage_id used as the message key so all events from the same stage land
        on the same partition, preserving per-stage ordering for the sliding
        window aggregator without a secondary sort step.
        """
        payload = MetricEventSerializer.serialize(event)
        self._producer.produce(
            topic=self._topic,
            key=event.stage_id.encode("utf-8"),
            value=payload,
            on_delivery=self._health_check.on_delivery,
        )
        # poll(0): drain ready callbacks without blocking. A non-zero timeout
        # would add per-message latency.
        self._producer.poll(0)

    def flush(self, timeout_s: float = 10.0) -> int:
        """
        Returns in-flight message count rather than raising on timeout, so the
        caller decides whether to retry, alert, or accept partial delivery.
        0 means all messages were acknowledged.
        """
        return self._producer.flush(timeout=timeout_s)

    def close(self) -> None:
        """
        Explicit flush on close rather than relying on the destructor: Python
        destructor ordering is non-deterministic and buffered messages would be
        silently dropped if the destructor runs after the network stack tears down.
        """
        remaining = self._producer.flush(timeout=30.0)
        if remaining > 0:
            raise RuntimeError(
                f"RedpandaProducer closed with {remaining} undelivered messages"
            )
