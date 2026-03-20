from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.consumer import ConsumedMessage, ConsumerConfig, MetricConsumer
from ingestion.models import PipelineMetric


@dataclass(frozen=True)
class WorkerConfig:
    """
    I'm separating WorkerConfig from ConsumerConfig because the batch and
    flush parameters belong to the writer, not the broker client. Merging them
    would make it impossible to test the worker's flush logic with a mock
    consumer without also constructing valid broker config.
    """

    db_url: str
    batch_size: int = 500
    flush_interval_s: float = 5.0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.flush_interval_s <= 0:
            raise ValueError(f"flush_interval_s must be > 0, got {self.flush_interval_s}")


class IngestionWorker:
    """
    I'm using a dual-trigger flush (batch_size OR flush_interval_s, whichever
    fires first) because either trigger alone has a failure mode. batch_size
    alone means a quiet topic with 499 events sitting in the buffer never
    flushes until the 500th arrives — adding arbitrary latency visible to the
    detection layer. flush_interval_s alone means a burst of 50,000 events
    holds the offset commit hostage until the timer fires, growing the replay
    window on crash. The dual trigger bounds both latency and replay volume.

    I'm using a synchronous SQLAlchemy Session rather than asyncpg because the
    IngestionWorker runs in a single-threaded loop where the bottleneck is
    broker poll latency, not DB write latency. Adding asyncio machinery would
    complicate the offset commit sequencing (must commit after write confirms)
    without measurable throughput gain at our event rate.

    The write → commit ordering is strict: we INSERT the batch into PostgreSQL
    and only then call consumer.commit_batch(). A crash between write and
    commit replays the batch from the last committed offset, producing
    duplicate writes that PostgreSQL's GENERATED ALWAYS AS IDENTITY absorbs
    by assigning new IDs. A crash between commit and write would lose the
    batch permanently — that ordering is forbidden.
    """

    def __init__(
        self,
        consumer: MetricConsumer,
        worker_config: WorkerConfig,
    ) -> None:
        self._consumer = consumer
        self._config = worker_config
        self._engine = create_engine(worker_config.db_url, pool_pre_ping=True)
        self._pending: List[ConsumedMessage] = []
        self._last_flush_time: float = time.monotonic()
        self._total_written: int = 0
        self._total_dlq: int = 0

    def _should_flush(self) -> bool:
        batch_full = len(self._pending) >= self._config.batch_size
        interval_elapsed = (
            time.monotonic() - self._last_flush_time >= self._config.flush_interval_s
        )
        return batch_full or (interval_elapsed and bool(self._pending))

    def _flush(self) -> None:
        """
        I'm counting DLQ records here (not in poll_batch) so the worker's
        metrics reflect what was actually processed in each flush cycle, not
        what was routed to DLQ during polling. This distinction matters when
        DLQ routing and DB writes happen in different code paths — having one
        counter source of truth avoids double-counting.
        """
        if not self._pending:
            return

        valid = [m for m in self._pending if m.is_valid]
        invalid_count = len(self._pending) - len(valid)

        if valid:
            self._write_to_db(valid)

        self._consumer.commit_batch(self._pending)
        self._total_written += len(valid)
        self._total_dlq += invalid_count
        self._pending.clear()
        self._last_flush_time = time.monotonic()

    def _write_to_db(self, messages: List[ConsumedMessage]) -> None:
        """
        I'm using a bulk INSERT via Session.add_all() rather than individual
        INSERT statements because the SQLAlchemy unit-of-work pattern batches
        all pending inserts into a single executemany() call at flush time.
        For 500 records this is roughly 50x faster than 500 individual INSERTs
        because it eliminates per-row round-trip latency.

        I'm not using INSERT ... ON CONFLICT DO NOTHING here even though
        replayed messages would produce duplicate rows (different id due to
        GENERATED ALWAYS AS IDENTITY). Duplicates are tolerable in the metrics
        table because the detection layer uses aggregates (mean latency, p99)
        that are robust to occasional duplicate records. Adding ON CONFLICT
        would require a natural key for deduplication that we don't have —
        (stage_id, event_time) is not unique since multiple events can share
        the same second.
        """
        orm_rows = [
            PipelineMetric(
                stage_id=m.event.stage_id,
                event_time=m.event.event_time,
                latency_ms=m.event.latency_ms,
                row_count=m.event.row_count,
                payload_bytes=m.event.payload_bytes,
                status=m.event.status,
                fault_label=m.event.fault_label,
                trace_id=m.event.trace_id,
            )
            for m in messages
            if m.event is not None
        ]
        with Session(self._engine) as session:
            session.add_all(orm_rows)
            session.commit()

    def run_once(self) -> int:
        """
        I'm exposing run_once() as the public step method rather than a
        blocking run() loop so that tests can drive the worker tick-by-tick
        without threads or async. A blocking run() would require the test to
        call it in a thread and coordinate shutdown — run_once() makes the
        control flow synchronous and deterministic in tests.

        Returns the number of records written to the DB in this call (0 if
        the batch was not yet ready to flush).
        """
        batch = self._consumer.poll_batch(
            max_records=self._config.batch_size,
            timeout_s=min(1.0, self._config.flush_interval_s),
        )
        self._pending.extend(batch)

        written = 0
        if self._should_flush():
            before = self._total_written
            self._flush()
            written = self._total_written - before

        return written

    def flush_remaining(self) -> None:
        """Force-flush whatever is left in the buffer, ignoring both triggers."""
        self._flush()

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def total_dlq(self) -> int:
        return self._total_dlq

    def close(self) -> None:
        self.flush_remaining()
        self._consumer.close()
        self._engine.dispose()
