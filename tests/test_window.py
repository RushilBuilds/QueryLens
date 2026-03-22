"""
Unit tests for RingBuffer, WindowConfig, and SlidingWindowAggregator.

I'm testing against numpy.percentile as the reference implementation rather
than hand-computing expected values because np.percentile with linear
interpolation is exactly the computation we call in SlidingWindowAggregator.
Any deviation between the two means the aggregator is calling a different
function — not that either value is wrong.

No containers or external services are needed here. All tests run on
synthetic event sequences with deterministic timestamps.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pytest

from detection.window import RingBuffer, SlidingWindowAggregator, WindowConfig, WindowStats
from simulator.models import PipelineEvent

SIM_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    stage_id: str,
    offset_s: float,
    latency_ms: float,
    row_count: int = 100,
    status: str = "ok",
) -> PipelineEvent:
    return PipelineEvent(
        stage_id=stage_id,
        event_time=SIM_START + timedelta(seconds=offset_s),
        latency_ms=latency_ms,
        row_count=row_count,
        payload_bytes=1024,
        status=status,
        fault_label=None,
        trace_id=None,
    )


def _default_config(**overrides) -> WindowConfig:
    defaults = dict(
        window_duration_s=60.0,
        tick_interval_s=1.0,
        min_sample_count=5,
        ring_buffer_capacity=1000,
    )
    defaults.update(overrides)
    return WindowConfig(**defaults)


# ---------------------------------------------------------------------------
# WindowConfig validation
# ---------------------------------------------------------------------------


class TestWindowConfig:

    def test_rejects_non_positive_window_duration(self) -> None:
        with pytest.raises(ValueError, match="window_duration_s"):
            WindowConfig(
                window_duration_s=0.0,
                tick_interval_s=1.0,
                min_sample_count=5,
                ring_buffer_capacity=100,
            )

    def test_rejects_non_positive_tick_interval(self) -> None:
        with pytest.raises(ValueError, match="tick_interval_s"):
            WindowConfig(
                window_duration_s=60.0,
                tick_interval_s=-1.0,
                min_sample_count=5,
                ring_buffer_capacity=100,
            )

    def test_rejects_zero_min_sample_count(self) -> None:
        with pytest.raises(ValueError, match="min_sample_count"):
            WindowConfig(
                window_duration_s=60.0,
                tick_interval_s=1.0,
                min_sample_count=0,
                ring_buffer_capacity=100,
            )


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------


class TestRingBuffer:

    def test_empty_buffer_returns_empty_array(self) -> None:
        """
        I'm testing the empty case first because it is the most common source
        of IndexError bugs — a buffer with count=0 but head=0 could produce
        indices that index into uninitialized slots.
        """
        buf = RingBuffer(capacity=10)
        result = buf.window_values(cutoff_s=0.0)
        assert len(result) == 0
        assert result.dtype == np.float64

    def test_push_and_retrieve_within_window(self) -> None:
        buf = RingBuffer(capacity=10)
        buf.push(timestamp_s=100.0, value=42.0)
        buf.push(timestamp_s=101.0, value=99.0)

        result = buf.window_values(cutoff_s=99.0)  # both qualify
        assert len(result) == 2
        np.testing.assert_array_equal(result, [42.0, 99.0])

    def test_entries_before_cutoff_are_excluded(self) -> None:
        """
        I'm verifying exclusion at the boundary specifically because off-by-one
        errors in >= vs > comparisons are the most common window bug. An entry
        at exactly cutoff_s must be included (>= semantics).
        """
        buf = RingBuffer(capacity=10)
        buf.push(timestamp_s=50.0, value=1.0)   # before window
        buf.push(timestamp_s=100.0, value=2.0)  # exactly at cutoff
        buf.push(timestamp_s=150.0, value=3.0)  # inside window

        result = buf.window_values(cutoff_s=100.0)
        np.testing.assert_array_equal(result, [2.0, 3.0])

    def test_insertion_order_preserved_in_returned_values(self) -> None:
        """
        I'm testing ordering because SlidingWindowAggregator passes the result
        directly to np.percentile, which is order-independent — but CUSUM
        will read the values in sequence to accumulate deviations. Out-of-order
        values would corrupt the accumulator.
        """
        buf = RingBuffer(capacity=5)
        for i in range(5):
            buf.push(timestamp_s=float(i), value=float(i * 10))

        result = buf.window_values(cutoff_s=-np.inf)
        np.testing.assert_array_equal(result, [0.0, 10.0, 20.0, 30.0, 40.0])

    def test_overwrites_oldest_entry_when_at_capacity(self) -> None:
        """
        I'm testing the wrap-around case with capacity=3 so the oldest entry
        is overwritten after exactly 4 inserts. A larger capacity would require
        more inserts to expose the wrap-around bug and make the test slower
        to reason about.
        """
        buf = RingBuffer(capacity=3)
        buf.push(1.0, 100.0)  # will be overwritten
        buf.push(2.0, 200.0)
        buf.push(3.0, 300.0)
        buf.push(4.0, 400.0)  # overwrites slot 0 (the 100.0 entry)

        result = buf.window_values(cutoff_s=-np.inf)
        # Should contain 200, 300, 400 — not 100
        assert len(result) == 3
        assert 100.0 not in result
        np.testing.assert_array_equal(result, [200.0, 300.0, 400.0])

    def test_count_capped_at_capacity(self) -> None:
        buf = RingBuffer(capacity=5)
        for i in range(10):
            buf.push(float(i), float(i))
        assert buf.count == 5

    def test_all_entries_excluded_when_cutoff_in_future(self) -> None:
        buf = RingBuffer(capacity=10)
        buf.push(100.0, 1.0)
        buf.push(200.0, 2.0)
        result = buf.window_values(cutoff_s=300.0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# SlidingWindowAggregator
# ---------------------------------------------------------------------------


class TestSlidingWindowAggregator:

    def test_percentiles_match_numpy_reference(self) -> None:
        """
        I'm using 20 values with a known distribution so the percentile
        computation is stable. With fewer than ~10 values, linear interpolation
        can produce results that feel surprising even though they are correct,
        which would make this test harder to reason about.
        """
        config = _default_config(window_duration_s=120.0, min_sample_count=5)
        agg = SlidingWindowAggregator(config)

        latencies = [float(x) for x in range(10, 110, 5)]  # 10, 15, ..., 105 — 20 values
        for i, lat in enumerate(latencies):
            agg.update(_event("src", offset_s=float(i), latency_ms=lat))

        now = SIM_START + timedelta(seconds=100)
        stats = agg.compute("src", now=now)

        assert stats.is_stable
        assert stats.latency_p50 == pytest.approx(np.percentile(latencies, 50))
        assert stats.latency_p95 == pytest.approx(np.percentile(latencies, 95))
        assert stats.latency_p99 == pytest.approx(np.percentile(latencies, 99))
        assert stats.latency_mean == pytest.approx(np.mean(latencies))

    def test_events_outside_window_are_excluded(self) -> None:
        """
        I'm using a 30-second window and inserting events at t=0 and t=60 so
        only the t=60 event falls inside the window when compute() is called
        at t=70. The t=0 event is 70 seconds old — outside the 30-second window.
        """
        config = _default_config(window_duration_s=30.0, min_sample_count=1)
        agg = SlidingWindowAggregator(config)

        agg.update(_event("src", offset_s=0.0, latency_ms=9999.0))   # outside window
        agg.update(_event("src", offset_s=60.0, latency_ms=50.0))    # inside window

        now = SIM_START + timedelta(seconds=70)
        stats = agg.compute("src", now=now)

        assert stats.is_stable
        assert stats.sample_count == 1
        assert stats.latency_p50 == pytest.approx(50.0)

    def test_is_stable_false_below_min_sample_count(self) -> None:
        """
        I'm verifying that all stat fields are None when is_stable is False.
        Returning 0.0 would look like a real latency measurement to CUSUM and
        trigger a false alert on startup — None forces the caller to gate on
        is_stable before using any stat.
        """
        config = _default_config(min_sample_count=10)
        agg = SlidingWindowAggregator(config)

        for i in range(5):  # fewer than min_sample_count
            agg.update(_event("src", offset_s=float(i), latency_ms=20.0))

        now = SIM_START + timedelta(seconds=100)
        stats = agg.compute("src", now=now)

        assert not stats.is_stable
        assert stats.latency_p50 is None
        assert stats.latency_p95 is None
        assert stats.latency_p99 is None
        assert stats.latency_mean is None
        assert stats.row_count_mean is None
        assert stats.error_rate is None

    def test_error_rate_computed_from_status_field(self) -> None:
        """
        I'm using 4 ok + 1 error to get an expected error rate of 0.2.
        The aggregator must derive error_rate from status at update() time
        so the detector layer never needs to see the raw "ok" string.
        """
        config = _default_config(min_sample_count=1)
        agg = SlidingWindowAggregator(config)

        for i in range(4):
            agg.update(_event("src", offset_s=float(i), latency_ms=10.0, status="ok"))
        agg.update(_event("src", offset_s=4.0, latency_ms=10.0, status="error"))

        now = SIM_START + timedelta(seconds=10)
        stats = agg.compute("src", now=now)

        assert stats.error_rate == pytest.approx(0.2)

    def test_row_count_mean_matches_numpy_reference(self) -> None:
        config = _default_config(min_sample_count=1)
        agg = SlidingWindowAggregator(config)

        row_counts = [100, 200, 150, 300, 250]
        for i, rc in enumerate(row_counts):
            agg.update(_event("src", offset_s=float(i), latency_ms=10.0, row_count=rc))

        now = SIM_START + timedelta(seconds=10)
        stats = agg.compute("src", now=now)

        assert stats.row_count_mean == pytest.approx(np.mean(row_counts))

    def test_independent_buffers_per_stage(self) -> None:
        """
        I'm asserting that stage A's stats are not contaminated by stage B's
        events. A shared buffer across stages would produce incorrect percentiles
        for every stage after the first one is populated.
        """
        config = _default_config(min_sample_count=1)
        agg = SlidingWindowAggregator(config)

        for i in range(5):
            agg.update(_event("stage_a", offset_s=float(i), latency_ms=10.0))
            agg.update(_event("stage_b", offset_s=float(i), latency_ms=999.0))

        now = SIM_START + timedelta(seconds=10)
        stats_a = agg.compute("stage_a", now=now)
        stats_b = agg.compute("stage_b", now=now)

        assert stats_a.latency_mean == pytest.approx(10.0)
        assert stats_b.latency_mean == pytest.approx(999.0)

    def test_empty_stage_returns_unstable_zero_sample_count(self) -> None:
        """
        I'm testing compute() on a stage that has never received any updates
        to ensure we don't raise a KeyError or return garbage stats. The
        aggregator lazily creates buffers — an unseen stage_id must return
        a valid WindowStats with sample_count=0.
        """
        config = _default_config(min_sample_count=1)
        agg = SlidingWindowAggregator(config)

        now = SIM_START + timedelta(seconds=10)
        stats = agg.compute("never_seen", now=now)

        assert not stats.is_stable
        assert stats.sample_count == 0

    def test_known_stages_reflects_all_updated_stage_ids(self) -> None:
        config = _default_config(min_sample_count=1)
        agg = SlidingWindowAggregator(config)

        agg.update(_event("alpha", offset_s=0.0, latency_ms=10.0))
        agg.update(_event("beta", offset_s=0.0, latency_ms=20.0))

        stages = agg.known_stages()
        assert "alpha" in stages
        assert "beta" in stages
        assert len(stages) == 2

    def test_ring_buffer_capacity_limits_retained_samples(self) -> None:
        """
        I'm using a capacity of 5 and inserting 10 events all within the
        window to verify that sample_count reflects the buffer cap, not the
        total events inserted. This ensures the detection layer cannot read
        more samples than the buffer holds and produce stale percentiles.
        """
        config = _default_config(
            window_duration_s=300.0,
            min_sample_count=1,
            ring_buffer_capacity=5,
        )
        agg = SlidingWindowAggregator(config)

        for i in range(10):
            agg.update(_event("src", offset_s=float(i), latency_ms=float(i * 10)))

        now = SIM_START + timedelta(seconds=100)
        stats = agg.compute("src", now=now)

        assert stats.sample_count == 5, (
            "sample_count should be capped at ring_buffer_capacity, "
            f"got {stats.sample_count}"
        )
        # Only the last 5 latencies (50, 60, 70, 80, 90) should be in the window.
        expected_mean = np.mean([50.0, 60.0, 70.0, 80.0, 90.0])
        assert stats.latency_mean == pytest.approx(expected_mean)
