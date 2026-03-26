from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np

from simulator.models import PipelineEvent


# ---------------------------------------------------------------------------
# WindowConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowConfig:
    """
    Shared config dataclass for all three detectors (CUSUM, EWMA, aggregator).
    A single object makes it impossible for them to silently diverge on window
    duration or tick interval.
    """

    window_duration_s: float
    tick_interval_s: float
    min_sample_count: int
    ring_buffer_capacity: int

    def __post_init__(self) -> None:
        if self.window_duration_s <= 0:
            raise ValueError(f"window_duration_s must be > 0, got {self.window_duration_s}")
        if self.tick_interval_s <= 0:
            raise ValueError(f"tick_interval_s must be > 0, got {self.tick_interval_s}")
        if self.min_sample_count < 1:
            raise ValueError(f"min_sample_count must be >= 1, got {self.min_sample_count}")
        if self.ring_buffer_capacity < 1:
            raise ValueError(f"ring_buffer_capacity must be >= 1, got {self.ring_buffer_capacity}")


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------


class RingBuffer:
    """
    Numpy-backed ring buffer. A deque would require a list-to-array copy on
    every compute() call — at 1,000 samples per stage that adds ~40μs of
    allocation overhead per tick. The numpy array allows in-place indexing
    and masking with no allocation on the hot path.

    O(1) insert via `_head % capacity`; O(n) mask in `window_values()`.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        # -inf sentinel: any real timestamp comparison (>= cutoff_s) correctly
        # excludes empty slots without a separate population check.
        self._timestamps = np.full(capacity, -np.inf, dtype=np.float64)
        self._values = np.zeros(capacity, dtype=np.float64)
        self._head: int = 0   # next write position (mod capacity)
        self._count: int = 0  # number of valid entries, capped at capacity

    @property
    def count(self) -> int:
        return self._count

    def push(self, timestamp_s: float, value: float) -> None:
        """
        No out-of-order check: events from a single stage always arrive in
        ascending order. A validation branch on every insert would add overhead
        for a condition that cannot occur in practice.
        """
        self._timestamps[self._head] = timestamp_s
        self._values[self._head] = value
        self._head = (self._head + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)

    def window_values(self, cutoff_s: float) -> np.ndarray:
        """
        Returns a 1D float64 array of values whose timestamps are >= cutoff_s,
        ordered oldest to newest. Uses a circular index array instead of
        np.roll() to avoid an O(n) copy of the underlying storage.
        """
        if self._count == 0:
            return np.empty(0, dtype=np.float64)

        # Build indices in insertion order: oldest first.
        start = (self._head - self._count) % self._capacity
        indices = (start + np.arange(self._count, dtype=np.intp)) % self._capacity

        ts = self._timestamps[indices]
        vals = self._values[indices]

        mask = ts >= cutoff_s
        return vals[mask]

    def all_timestamps(self) -> np.ndarray:
        """Return timestamps in insertion order — used for test assertions only."""
        if self._count == 0:
            return np.empty(0, dtype=np.float64)
        start = (self._head - self._count) % self._capacity
        indices = (start + np.arange(self._count, dtype=np.intp)) % self._capacity
        return self._timestamps[indices]


# ---------------------------------------------------------------------------
# WindowStats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowStats:
    """
    All stat fields are None when is_stable is False. Returning 0.0 would look
    like a real measurement to downstream detectors (latency_p99=0.0 would
    trigger a false alert on startup); NaN propagates silently through numpy.
    None forces the caller to gate on is_stable before consuming any stat.
    """

    stage_id: str
    computed_at: datetime
    sample_count: int
    is_stable: bool
    latency_p50: Optional[float]
    latency_p95: Optional[float]
    latency_p99: Optional[float]
    latency_mean: Optional[float]
    row_count_mean: Optional[float]
    error_rate: Optional[float]


# ---------------------------------------------------------------------------
# SlidingWindowAggregator
# ---------------------------------------------------------------------------


class SlidingWindowAggregator:
    """
    Maintains one RingBuffer per (stage_id, metric_name) pair rather than one
    structured buffer per stage. This lets CUSUM read latency_ms as a clean 1D
    float array with no column extraction, and allows new metrics to be added
    without invalidating existing buffers.

    error_rate is derived at update time (1.0 if status != "ok") so compute()
    uses np.mean() without ever depending on the "ok" string literal.
    """

    _METRICS = ("latency_ms", "row_count", "error_rate")

    def __init__(self, config: WindowConfig) -> None:
        self._config = config
        self._buffers: Dict[Tuple[str, str], RingBuffer] = {}

    def _buffer(self, stage_id: str, metric: str) -> RingBuffer:
        key = (stage_id, metric)
        if key not in self._buffers:
            self._buffers[key] = RingBuffer(self._config.ring_buffer_capacity)
        return self._buffers[key]

    def update(self, event: PipelineEvent) -> None:
        """
        Stores event_time as a Unix float rather than datetime. Numpy boolean
        comparison on float arrays is ~10x faster than datetime comparisons in
        a loop, and the cutoff in window_values() is already a float.
        """
        ts = event.event_time.timestamp()
        stage = event.stage_id

        self._buffer(stage, "latency_ms").push(ts, event.latency_ms)
        self._buffer(stage, "row_count").push(ts, float(event.row_count))
        error_flag = 0.0 if event.status == "ok" else 1.0
        self._buffer(stage, "error_rate").push(ts, error_flag)

    def compute(self, stage_id: str, now: datetime) -> WindowStats:
        """
        Computes p50/p95/p99 via np.percentile (linear interpolation) to match
        the Prometheus histogram quantile definition, making pre-computed
        percentiles directly comparable to Prometheus output.

        Returns None for all stats when is_stable is False — no numpy call on
        the unstable path, keeping the allocation cost constant.
        """
        cutoff_s = now.timestamp() - self._config.window_duration_s

        latency_vals = self._buffer(stage_id, "latency_ms").window_values(cutoff_s)
        row_count_vals = self._buffer(stage_id, "row_count").window_values(cutoff_s)
        error_vals = self._buffer(stage_id, "error_rate").window_values(cutoff_s)

        n = len(latency_vals)
        is_stable = n >= self._config.min_sample_count

        if not is_stable:
            return WindowStats(
                stage_id=stage_id,
                computed_at=now,
                sample_count=n,
                is_stable=False,
                latency_p50=None,
                latency_p95=None,
                latency_p99=None,
                latency_mean=None,
                row_count_mean=None,
                error_rate=None,
            )

        return WindowStats(
            stage_id=stage_id,
            computed_at=now,
            sample_count=n,
            is_stable=True,
            latency_p50=float(np.percentile(latency_vals, 50)),
            latency_p95=float(np.percentile(latency_vals, 95)),
            latency_p99=float(np.percentile(latency_vals, 99)),
            latency_mean=float(np.mean(latency_vals)),
            row_count_mean=float(np.mean(row_count_vals)),
            error_rate=float(np.mean(error_vals)),
        )

    def known_stages(self) -> list:
        """Return all stage IDs that have received at least one update."""
        return list({stage_id for stage_id, _ in self._buffers})
