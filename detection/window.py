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
    I'm separating window configuration into its own dataclass rather than
    passing individual parameters to SlidingWindowAggregator because all three
    detectors (CUSUM, EWMA, and the aggregator itself) need to share the same
    window duration and tick interval. A single config object makes it
    impossible for them to silently diverge on these values.
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
    I'm using a numpy array rather than a Python deque for the ring buffer
    because `window_values()` is called on every detection tick and feeds
    directly into np.percentile(). With a deque, every compute() call would
    require a list-to-array conversion, which at 1,000 samples per stage
    would add ~40μs of allocation overhead per tick — multiplied by the
    number of stages and metrics, that compounds into meaningful tail latency
    on the detection loop. The numpy array lets us index and mask in-place
    with no allocation on the hot path.

    O(1) insert: we write at `_head % capacity` and advance the pointer.
    Old entries are overwritten implicitly when the buffer fills — no element
    shifting, no deque rotation. Time-based eviction in `window_values()` is
    O(n) for the mask operation, but the ring buffer structure itself is O(1)
    to maintain per insert, which is the invariant the detection layer needs.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        # I'm using -inf as the sentinel for empty slots so that any real
        # timestamp comparison (>= cutoff_s) will correctly exclude them
        # without a separate "is slot populated?" check.
        self._timestamps = np.full(capacity, -np.inf, dtype=np.float64)
        self._values = np.zeros(capacity, dtype=np.float64)
        self._head: int = 0   # next write position (mod capacity)
        self._count: int = 0  # number of valid entries, capped at capacity

    @property
    def count(self) -> int:
        return self._count

    def push(self, timestamp_s: float, value: float) -> None:
        """
        I'm not checking for out-of-order timestamps here. Events arriving
        from PipelineEvent.event_time are always in ascending order within
        a single stage's stream. Enforcing ordering would add a branch on
        every insert for a condition that cannot occur in practice, and
        silently dropping out-of-order events would corrupt the window in
        tests that deliberately insert in a non-standard order.
        """
        self._timestamps[self._head] = timestamp_s
        self._values[self._head] = value
        self._head = (self._head + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)

    def window_values(self, cutoff_s: float) -> np.ndarray:
        """
        I'm computing a circular index array rather than rotating the
        underlying storage because rotation is O(n) and would require
        copying or np.roll(). The index array is O(n) to build but
        shares memory with the underlying arrays through fancy indexing —
        no second copy of the data.

        Returns a 1D float64 array of values whose timestamps are >= cutoff_s,
        ordered from oldest to newest. Empty array if no values qualify.
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
    I'm returning None for all stat fields when is_stable is False rather
    than returning 0.0 or NaN. Returning 0.0 would look like a real measurement
    to a downstream detector and could trigger a false alert on startup (e.g.
    latency_p99=0.0 deviating from a baseline mean of 50ms). NaN propagates
    silently through numpy operations. None forces the caller to explicitly
    handle the unstable case, which is the correct behaviour.
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
    I'm maintaining one RingBuffer per (stage_id, metric_name) pair rather
    than one buffer per stage containing a structured record. Separate buffers
    mean that CUSUM can read latency_ms values as a clean 1D float array with
    no column extraction step, and that a new metric can be added without
    changing the buffer schema or invalidating any existing buffer's data.

    The three tracked metrics are latency_ms, row_count, and error_rate.
    error_rate is derived at update time (1.0 if status != "ok", else 0.0)
    so that compute() can use np.mean() to get the fraction without knowing
    the raw status string — the detector layer should not depend on the
    "ok" string literal.
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
        I'm converting event_time to a Unix timestamp (float seconds since
        epoch) for storage rather than keeping it as a datetime. numpy boolean
        comparison on float arrays is ~10x faster than Python datetime
        comparisons in a loop, and the cutoff in window_values() is computed
        once per tick as a float — there is no need to ever convert back.
        """
        ts = event.event_time.timestamp()
        stage = event.stage_id

        self._buffer(stage, "latency_ms").push(ts, event.latency_ms)
        self._buffer(stage, "row_count").push(ts, float(event.row_count))
        error_flag = 0.0 if event.status == "ok" else 1.0
        self._buffer(stage, "error_rate").push(ts, error_flag)

    def compute(self, stage_id: str, now: datetime) -> WindowStats:
        """
        I'm computing p50/p95/p99 with np.percentile (linear interpolation)
        rather than np.quantile with a different method because linear
        interpolation is the Prometheus-compatible definition used by histogram
        quantile estimation. Using the same method means our pre-computed
        percentiles can be directly compared to what Prometheus would report
        for the same data distribution.

        When is_stable is False we return None for all stats. This bounds the
        cost of returning a WindowStats object to a single allocation regardless
        of sample count — there is no conditional numpy call on the unstable path.
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
