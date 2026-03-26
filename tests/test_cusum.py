"""
Unit tests for CUSUMDetector and AnomalyEvent.

The critical property this test suite validates is that CUSUM catches
gradual ramp-change sequences that a per-event z-score threshold misses.
This is the entire reason CUSUM exists in QueryLens — a simple z-score
threshold would be far cheaper to implement but would generate false
positives on minor spikes and miss sustained drift entirely.

No containers or external services are required. All baselines are
constructed in-memory with known mean and std values.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from detection.anomaly import AnomalyEvent, extract_metric
from detection.baseline import BaselineEntry, BaselineKey, SeasonalBaselineModel
from detection.cusum import CUSUMConfig, CUSUMDetector
from simulator.models import PipelineEvent

SIM_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # Monday 00:00 → how=0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_baseline(mean: float, std: float, stage_id: str = "src") -> SeasonalBaselineModel:
    """
    Same (mean, std) for all 168 hour_of_week slots so tests can insert events
    at any timestamp without worrying about seasonal slot boundaries.
    Slot-varying behaviour is tested in test_baseline.py.
    """
    entries = {
        BaselineKey(stage_id, how, metric): BaselineEntry(
            baseline_mean=mean,
            baseline_std=std,
            sample_count=50,
            fitted_at=SIM_START,
        )
        for how in range(168)
        for metric in ("latency_ms", "row_count", "error_rate")
    }
    return SeasonalBaselineModel(entries)


def _event(
    offset_s: float,
    latency_ms: float,
    stage_id: str = "src",
    status: str = "ok",
    row_count: int = 100,
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


def _fire_events(
    detector: CUSUMDetector,
    events: List[PipelineEvent],
) -> List[AnomalyEvent]:
    """Feed all events and collect every AnomalyEvent emitted."""
    all_anomalies: List[AnomalyEvent] = []
    for event in events:
        all_anomalies.extend(detector.update(event))
    return all_anomalies


# ---------------------------------------------------------------------------
# CUSUMConfig validation
# ---------------------------------------------------------------------------


class TestCUSUMConfig:

    def test_rejects_non_positive_threshold(self) -> None:
        with pytest.raises(ValueError, match="decision_threshold"):
            CUSUMConfig(decision_threshold=0.0, slack_parameter=0.5)

    def test_rejects_negative_slack(self) -> None:
        with pytest.raises(ValueError, match="slack_parameter"):
            CUSUMConfig(decision_threshold=4.0, slack_parameter=-0.1)

    def test_valid_config_constructs(self) -> None:
        cfg = CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5)
        assert cfg.decision_threshold == 4.0
        assert cfg.slack_parameter == 0.5


# ---------------------------------------------------------------------------
# AnomalyEvent and extract_metric
# ---------------------------------------------------------------------------


class TestExtractMetric:

    def test_extracts_latency_ms(self) -> None:
        event = _event(0.0, latency_ms=75.5)
        assert extract_metric(event, "latency_ms") == pytest.approx(75.5)

    def test_extracts_row_count_as_float(self) -> None:
        event = _event(0.0, latency_ms=10.0, row_count=200)
        assert extract_metric(event, "row_count") == pytest.approx(200.0)

    def test_error_rate_is_zero_for_ok_status(self) -> None:
        event = _event(0.0, latency_ms=10.0, status="ok")
        assert extract_metric(event, "error_rate") == 0.0

    def test_error_rate_is_one_for_non_ok_status(self) -> None:
        event = _event(0.0, latency_ms=10.0, status="error")
        assert extract_metric(event, "error_rate") == 1.0

    def test_raises_for_unknown_metric(self) -> None:
        event = _event(0.0, latency_ms=10.0)
        with pytest.raises(ValueError, match="Unknown metric"):
            extract_metric(event, "throughput_rps")


# ---------------------------------------------------------------------------
# Core CUSUM behaviour
# ---------------------------------------------------------------------------


class TestCUSUMDetectorBaseline:

    def test_no_anomaly_during_on_target_events(self) -> None:
        """
        100 events at the baseline mean (z=0). With k>0, every update produces
        max(0, 0-k)=0 so accumulators stay at zero permanently. Any fire here
        means the accumulator update formula is wrong.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5, metrics=("latency_ms",)),
            baseline,
        )
        events = [_event(float(i), latency_ms=50.0) for i in range(100)]
        anomalies = _fire_events(detector, events)
        assert anomalies == [], (
            f"Expected no anomalies for on-target events, got {len(anomalies)}"
        )

    def test_accumulator_starts_at_zero(self) -> None:
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5),
            baseline,
        )
        s_up, s_lo = detector.accumulator_state("src", "latency_ms")
        assert s_up == 0.0
        assert s_lo == 0.0

    def test_accumulator_increments_correctly_for_positive_shift(self) -> None:
        """
        Asserts the accumulator after one event so a formula off-by-one (e.g.
        S = S + z instead of S = max(0, S + z - k)) fails here, not buried in
        a detection lag assertion.

        With z=1.0 and k=0.5: S_upper after 1 step = max(0, 0 + 1.0 - 0.5) = 0.5
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5),
            baseline,
        )
        detector.update(_event(0.0, latency_ms=60.0))  # z = (60-50)/10 = 1.0
        s_up, _ = detector.accumulator_state("src", "latency_ms")
        assert s_up == pytest.approx(0.5)

    def test_accumulator_floored_at_zero_for_below_baseline_events(self) -> None:
        """
        The max(0, ...) floor prevents the accumulator from going negative. A
        negative accumulator gives the detector a head start on the next upward
        shift — effectively reducing h without changing the config.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5),
            baseline,
        )
        detector.update(_event(0.0, latency_ms=30.0))  # z = -2.0
        s_up, _ = detector.accumulator_state("src", "latency_ms")
        assert s_up == 0.0, "S_upper must be floored at 0, not go negative"


# ---------------------------------------------------------------------------
# Step-change detection
# ---------------------------------------------------------------------------


class TestCUSUMStepChange:

    def test_detects_large_upward_step(self) -> None:
        """
        latency=90ms (z=4.0), k=0.5, h=4.0. After one event S_upper=3.5;
        after two events S_upper=7.0 > h → fires. Must emit signal='upper'.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5),
            baseline,
        )
        events = [_event(float(i), latency_ms=90.0) for i in range(5)]
        anomalies = _fire_events(detector, events)

        assert len(anomalies) >= 1
        latency_anomalies = [a for a in anomalies if a.metric == "latency_ms"]
        assert len(latency_anomalies) >= 1
        assert latency_anomalies[0].signal == "upper"
        assert latency_anomalies[0].detector_type == "cusum"
        assert latency_anomalies[0].stage_id == "src"

    def test_detects_large_downward_step(self) -> None:
        """
        Lower accumulator with latency=10ms (z=-4.0). Symmetric CUSUM must
        catch downward drift — a detector watching only upward shifts would
        miss a source dropping records or a sink completing suspiciously fast.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5),
            baseline,
        )
        events = [_event(float(i), latency_ms=10.0) for i in range(5)]
        anomalies = _fire_events(detector, events)

        latency_anomalies = [a for a in anomalies if a.metric == "latency_ms"]
        assert any(a.signal == "lower" for a in latency_anomalies), (
            "CUSUM must detect downward step with signal='lower'"
        )

    def test_accumulator_resets_to_zero_after_fire(self) -> None:
        """
        Verifies reset by checking the accumulator does not keep growing after
        the first fire. Without a reset, the detector would fire on every
        subsequent event once the threshold is exceeded.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5),
            baseline,
        )
        # Feed enough events to fire once (step of z=4 fires after 2 events).
        for i in range(3):
            detector.update(_event(float(i), latency_ms=90.0))

        s_up, _ = detector.accumulator_state("src", "latency_ms")
        # After the reset, the accumulator should be below h (growing from 0 again).
        assert s_up < 4.0, (
            "Accumulator must reset to 0 after firing, not continue from h"
        )


# ---------------------------------------------------------------------------
# Ramp-change detection vs z-score threshold
# ---------------------------------------------------------------------------


class TestCUSUMRampChange:
    """
    The ramp scenario is the definitive test of CUSUM's value. A simple
    z-score alarm misses gradual drift because no single measurement exceeds
    the threshold. CUSUM accumulates the sub-threshold deviations and fires
    after enough evidence has built up.

    Setup:
    - baseline mean=50ms, std=10ms
    - z-score single-event alarm threshold = 2.0 (fire if |z| > 2.0)
    - CUSUM: k=0.5, h=4.0
    - Ramp: latency increases from 56ms to 65ms in 1ms steps (z = 0.6 to 1.5)
      All individual z-scores are below 2.0 so the z-score alarm never fires.
      CUSUM accumulates: Σ(z_i - 0.5) for i=1..9
        = (0.1) + (0.2) + ... + (0.9) = 4.5 > h=4.0 → fires at step 9 of 10.
    """

    BASELINE_MEAN = 50.0
    BASELINE_STD = 10.0
    Z_SCORE_THRESHOLD = 2.0  # simple per-event alarm threshold
    CUSUM_K = 0.5
    CUSUM_H = 4.0

    def _ramp_events(self) -> List[PipelineEvent]:
        """Latencies 56, 57, 58, 59, 60, 61, 62, 63, 64, 65 → z = 0.6 to 1.5."""
        return [_event(float(i), latency_ms=56.0 + i) for i in range(10)]

    def test_z_score_alarm_does_not_fire_on_ramp(self) -> None:
        """
        If a simple z-score alarm fires here, CUSUM is redundant and a simpler
        threshold would suffice. Failure means the ramp latencies are too large
        and the test setup needs smaller steps.
        """
        for event in self._ramp_events():
            z = abs((event.latency_ms - self.BASELINE_MEAN) / self.BASELINE_STD)
            assert z < self.Z_SCORE_THRESHOLD, (
                f"latency={event.latency_ms} gives z={z:.2f} which exceeds the "
                f"z-score threshold {self.Z_SCORE_THRESHOLD} — ramp is too steep "
                "for this test; reduce step size"
            )

    def test_cusum_detects_ramp_that_z_score_misses(self) -> None:
        """
        At least one AnomalyEvent must fire across the ramp. The exact event
        index depends on accumulator arithmetic — 'at least one' is the right
        contract because CUSUM fires when sufficient evidence accumulates, not
        at a fixed lag.
        """
        baseline = _flat_baseline(self.BASELINE_MEAN, self.BASELINE_STD)
        detector = CUSUMDetector(
            CUSUMConfig(
                decision_threshold=self.CUSUM_H,
                slack_parameter=self.CUSUM_K,
                metrics=("latency_ms",),
            ),
            baseline,
        )
        anomalies = _fire_events(detector, self._ramp_events())

        latency_fires = [a for a in anomalies if a.metric == "latency_ms"]
        assert len(latency_fires) >= 1, (
            "CUSUM must detect the ramp sequence. "
            f"Accumulator did not exceed h={self.CUSUM_H} across {len(self._ramp_events())} events. "
            "Check that slack_parameter and decision_threshold are set correctly."
        )
        assert latency_fires[0].signal == "upper"

    def test_cusum_fires_before_ramp_ends(self) -> None:
        """
        Detection must happen before the last event so CUSUM is actually
        accumulating, not just catching the final z-score (indistinguishable
        from a simple alarm). The ramp has 10 events; must fire by index 8.
        """
        baseline = _flat_baseline(self.BASELINE_MEAN, self.BASELINE_STD)
        detector = CUSUMDetector(
            CUSUMConfig(
                decision_threshold=self.CUSUM_H,
                slack_parameter=self.CUSUM_K,
                metrics=("latency_ms",),
            ),
            baseline,
        )
        fire_index: Optional[int] = None
        for i, event in enumerate(self._ramp_events()):
            if detector.update(event):
                fire_index = i
                break

        assert fire_index is not None, "CUSUM never fired on the ramp sequence"
        assert fire_index < len(self._ramp_events()) - 1, (
            f"CUSUM fired on the last event (index {fire_index}) — "
            "this is not accumulation-based detection, just a single z-score alarm"
        )


# ---------------------------------------------------------------------------
# Multi-stage isolation and reset
# ---------------------------------------------------------------------------


class TestCUSUMMultiStage:

    def test_stage_accumulators_are_independent(self) -> None:
        """
        Fault events sent to stage_a only — stage_b's accumulator must stay at
        zero. A shared dict with a stage_id key bug would contaminate stage_b.
        """
        baseline_a = _flat_baseline(50.0, 10.0, stage_id="stage_a")
        # Build combined baseline for both stages
        entries = dict(baseline_a._entries)  # access internal for test setup
        for how in range(168):
            for metric in ("latency_ms", "row_count", "error_rate"):
                entries[BaselineKey("stage_b", how, metric)] = BaselineEntry(
                    baseline_mean=50.0,
                    baseline_std=10.0,
                    sample_count=50,
                    fitted_at=SIM_START,
                )
        combined_baseline = SeasonalBaselineModel(entries)

        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5),
            combined_baseline,
        )

        for i in range(10):
            detector.update(_event(float(i), latency_ms=90.0, stage_id="stage_a"))

        s_up_b, s_lo_b = detector.accumulator_state("stage_b", "latency_ms")
        assert s_up_b == 0.0
        assert s_lo_b == 0.0

    def test_reset_clears_specific_stage_metric(self) -> None:
        baseline = _flat_baseline(50.0, 10.0)
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=100.0, slack_parameter=0.0),
            baseline,
        )
        # Build up accumulator without firing (threshold is 100)
        for i in range(5):
            detector.update(_event(float(i), latency_ms=70.0))  # z=2.0 each step

        s_up_before, _ = detector.accumulator_state("src", "latency_ms")
        assert s_up_before > 0.0

        detector.reset("src", "latency_ms")
        s_up_after, _ = detector.accumulator_state("src", "latency_ms")
        assert s_up_after == 0.0

    def test_missing_baseline_skips_update_silently(self) -> None:
        """
        An event for a stage with no baseline entry must not raise and must not
        leave a non-zero accumulator — a no-op, not a crash.
        """
        empty_baseline = SeasonalBaselineModel({})
        detector = CUSUMDetector(
            CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5),
            empty_baseline,
        )
        anomalies = detector.update(_event(0.0, latency_ms=999.0))
        assert anomalies == []
        s_up, s_lo = detector.accumulator_state("src", "latency_ms")
        assert s_up == 0.0
        assert s_lo == 0.0
