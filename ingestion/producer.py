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
    I'm tracking delivery outcomes in a thread-safe counter rather than
    collecting failed messages in a list because the failure signal we need
    for Prometheus is a running total, not a replay buffer. Storing full
    Message objects would grow unbounded on a sustained fault and would require
    a separate eviction policy. The counter gives Prometheus what it needs
    (rate of delivery failures) at O(1) memory cost.

    The lock protects _failed and _succeeded from concurrent writes by the
    librdkafka delivery callback thread, which is distinct from the producer
    thread. Without the lock, the counters would be subject to lost-update
    races on CPython (despite the GIL) because the increment is not a single
    bytecode operation.
    """

    _failed: int = field(default=0, init=False, repr=False)
    _succeeded: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def on_delivery(self, err: Optional[KafkaError], msg: Message) -> None:
        """
        Delivery callback invoked by librdkafka's internal thread after each
        message is either acknowledged or permanently failed. I'm not logging
        per-message failures here because at 5,000 events/sec a log line per
        failure would saturate the log pipeline before the alert fires. The
        Prometheus counter is the right signal; structured logs belong at the
        batch level in the IngestionWorker.
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
    I'm wrapping confluent_kafka.Producer rather than using it directly in the
    ingestion layer because the raw Producer API exposes librdkafka configuration
    strings directly (e.g. 'acks' instead of 'request.required.acks'), which are
    easy to misconfigure silently. The wrapper enforces the durability settings we
    need (acks=all, retries=5) at construction time and hides the poll/flush
    mechanics that callers would otherwise have to replicate.

    I'm calling poll(0) after every produce() rather than only at flush() time
    because librdkafka's delivery callbacks are only triggered during poll calls.
    Without per-produce polling, ProducerHealthCheck.failed_delivery_count would
    read zero until flush() completed, making it useless as a real-time health
    signal between flushes.
    """

    # I'm setting linger.ms=5 to allow librdkafka to batch messages that arrive
    # within 5ms of each other into a single batch request. At 5,000 events/sec
    # this halves the number of round trips compared to linger.ms=0. The 5ms
    # added latency is invisible at our detection window granularity (seconds).
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
        # I'm using enable.idempotence=True alongside acks=all and retries=5
        # to guarantee exactly-once delivery at the producer level. Without
        # idempotence, retries can produce duplicate messages when the broker
        # receives the message but the ack is lost in transit.
        self._producer = Producer(config)

    def publish(self, event: PipelineEvent) -> None:
        """
        I'm using stage_id as the message key so that all events from the same
        pipeline stage land on the same partition. This preserves per-stage
        event ordering, which the sliding window aggregator depends on to
        produce correct time-ordered windows without a secondary sort step.
        """
        payload = MetricEventSerializer.serialize(event)
        self._producer.produce(
            topic=self._topic,
            key=event.stage_id.encode("utf-8"),
            value=payload,
            on_delivery=self._health_check.on_delivery,
        )
        # Service the delivery callback queue without blocking. A non-zero
        # poll timeout here would add per-message latency; 0 lets librdkafka
        # drain any callbacks that are already ready.
        self._producer.poll(0)

    def flush(self, timeout_s: float = 10.0) -> int:
        """
        I'm returning the number of messages still in-flight rather than
        raising on timeout because the caller (IngestionWorker) needs to decide
        whether to retry, alert, or accept partial delivery depending on context.
        Raising here would force a single error-handling strategy on all callers.

        Returns the count of messages that did not deliver within timeout_s.
        A return value of 0 means all messages were acknowledged.
        """
        return self._producer.flush(timeout=timeout_s)

    def close(self) -> None:
        """
        I'm flushing with a generous timeout on close rather than letting the
        Producer destructor handle it because destructor ordering in Python is
        non-deterministic. Messages buffered at process shutdown would be silently
        dropped if the destructor runs after the network stack is torn down.
        """
        remaining = self._producer.flush(timeout=30.0)
        if remaining > 0:
            raise RuntimeError(
                f"RedpandaProducer closed with {remaining} undelivered messages"
            )
