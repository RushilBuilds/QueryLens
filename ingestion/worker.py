from __future__ import annotations

import time
from collections import Counter as _StageCounter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.consumer import ConsumedMessage, ConsumerConfig, MetricConsumer
from ingestion.models import PipelineMetric
from ingestion.observability import CONSUMER_LAG, RECORDS_CONSUMED, RECORDS_WRITTEN, WRITE_LATENCY

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    """
    Batch and flush parameters belong to the writer, not the broker client.
    Keeping them separate allows the worker's flush logic to be tested with a
    mock consumer without constructing valid broker config.
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
    Dual-trigger flush (batch_size OR flush_interval_s). batch_size alone
    would stall a quiet topic indefinitely; flush_interval_s alone lets bursts
    grow the crash-replay window unboundedly. The dual trigger bounds both
    detection latency and replay volume.

    Synchronous SQLAlchemy Session rather than asyncpg: the bottleneck is
    broker poll latency, not DB write latency. Async machinery would complicate
    offset commit sequencing without measurable throughput gain at this rate.

    Write → commit ordering is strict: INSERT to PostgreSQL first, then
    commit_batch(). A crash between write and commit replays the batch from
    the last committed offset (tolerable duplicates). A crash between commit
    and write would lose the batch permanently.
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
        DLQ records counted here (not in poll_batch) so metrics reflect what
        was processed per flush cycle. One counter source of truth avoids
        double-counting when DLQ routing and DB writes happen in different paths.
        """
        if not self._pending:
            return

        valid = [m for m in self._pending if m.is_valid]
        invalid_count = len(self._pending) - len(valid)

        # Per-stage counts computed before the DB write so Prometheus labels are
        # available even if the write raises. RECORDS_CONSUMED = parsed off the wire;
        # RECORDS_WRITTEN = landed in PostgreSQL. Delta = transient write failure rate.
        stage_counts = _StageCounter(
            m.event.stage_id for m in valid if m.event is not None
        )
        for stage_id, cnt in stage_counts.items():
            RECORDS_CONSUMED.labels(stage_id=stage_id).inc(cnt)

        # Update consumer lag: age of the oldest event per stage in this batch.
        now_utc = datetime.now(tz=timezone.utc)
        for stage_id, oldest_event_time in _oldest_event_time_per_stage(valid).items():
            lag_s = (now_utc - oldest_event_time).total_seconds()
            CONSUMER_LAG.labels(stage_id=stage_id).set(max(0.0, lag_s))

        if valid:
            self._write_to_db(valid)

        self._consumer.commit_batch(self._pending)

        for stage_id, cnt in stage_counts.items():
            RECORDS_WRITTEN.labels(stage_id=stage_id).inc(cnt)

        self._total_written += len(valid)
        self._total_dlq += invalid_count
        self._pending.clear()
        self._last_flush_time = time.monotonic()
        _log.info(
            "batch_flushed",
            valid=len(valid),
            dlq=invalid_count,
            total_written=self._total_written,
        )

    def _write_to_db(self, messages: List[ConsumedMessage]) -> None:
        """
        Bulk INSERT via Session.add_all() — SQLAlchemy batches all pending
        inserts into a single executemany() call, ~50x faster than 500
        individual INSERTs for a 500-record batch.

        No ON CONFLICT DO NOTHING: replayed messages produce duplicate rows
        with new IDs (GENERATED ALWAYS AS IDENTITY), which is tolerable because
        the detection layer uses aggregates (mean latency, p99) that are robust
        to occasional duplicates. A natural deduplication key doesn't exist —
        (stage_id, event_time) is not unique within the same second.
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
        t0 = time.monotonic()
        with Session(self._engine) as session:
            session.add_all(orm_rows)
            session.commit()
        elapsed = time.monotonic() - t0
        WRITE_LATENCY.observe(elapsed)
        _log.info("db_write_complete", n_rows=len(orm_rows), latency_s=round(elapsed, 4))

    def run_once(self) -> int:
        """
        Single-step public API instead of a blocking run() loop, so tests drive
        the worker tick-by-tick without threads or async coordination.
        Returns records written in this call (0 if batch not yet ready to flush).
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


def _oldest_event_time_per_stage(
    messages: List[ConsumedMessage],
) -> dict:
    """
    Oldest event_time computed per stage rather than globally because
    CONSUMER_LAG is a per-stage gauge. A global gauge would mask a single slow
    stage and prevent alert rules from targeting the specific lagging stage.
    """
    oldest: dict = {}
    for m in messages:
        if m.event is None:
            continue
        stage = m.event.stage_id
        t = m.event.event_time
        if stage not in oldest or t < oldest[stage]:
            oldest[stage] = t
    return oldest
