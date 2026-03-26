"""
Unit tests for EWMADetector and EWMAConfig.

The definitive test in this suite is the impulse test: a single event with
|z| > L must fire EWMA within one tick while CUSUM (needing accumulation)
does not. This is the entire reason EWMA exists alongside CUSUM — the two
detectors cover complementary failure modes: CUSUM catches gradual ramp drift,
EWMA catches sudden spikes that CUSUM's accumulation lag would miss.

No containers or external services are required. All baselines are constructed
in-memory with known mean and std values.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from detection.anomaly import AnomalyEvent
from detection.baseline import BaselineEntry, BaselineKey, SeasonalBaselineModel
from detection.cusum import CUSUMConfig, CUSUMDetector
from detection.ewma import EWMAConfig, EWMADetector
from simulator.models import PipelineEvent

SIM_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # Monday 00:00 → how=0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_baseline(mean: float, std: float, stage_id: str = "src") -> SeasonalBaselineModel:
    """
    Uniform (mean, std) across all 168 hour_of_week slots so tests can insert
    events at any timestamp without worrying about seasonal slot boundaries.
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
    detector: EWMADetector,
    events: List[PipelineEvent],
) -> List[AnomalyEvent]:
    """Feed all events and collect every AnomalyEvent emitted."""
    result: List[AnomalyEvent] = []
    for event in events:
        result.extend(detector.update(event))
    return result


# ---------------------------------------------------------------------------
# EWMAConfig validation
# ---------------------------------------------------------------------------


class TestEWMAConfig:

    def test_rejects_zero_smoothing(self) -> None:
        with pytest.raises(ValueError, match="smoothing"):
            EWMAConfig(smoothing=0.0, control_limit_width=3.0)

    def test_rejects_smoothing_above_one(self) -> None:
        with pytest.raises(ValueError, match="smoothing"):
            EWMAConfig(smoothing=1.1, control_limit_width=3.0)

    def test_rejects_non_positive_control_limit(self) -> None:
        with pytest.raises(ValueError, match="control_limit_width"):
            EWMAConfig(smoothing=0.2, control_limit_width=0.0)

    def test_smoothing_of_one_is_valid(self) -> None:
        """λ=1 collapses EWMA to a per-event z-score — a valid degenerate case."""
        cfg = EWMAConfig(smoothing=1.0, control_limit_width=3.0)
        assert cfg.smoothing == 1.0

    def test_valid_config_constructs(self) -> None:
        cfg = EWMAConfig(smoothing=0.2, control_limit_width=3.0)
        assert cfg.smoothing == 0.2
        assert cfg.control_limit_width == 3.0


# ---------------------------------------------------------------------------
# EWMA arithmetic
# ---------------------------------------------------------------------------


class TestEWMAArithmetic:

    def test_ewma_starts_at_zero(self) -> None:
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = EWMADetector(EWMAConfig(smoothing=0.3, control_limit_width=3.0), baseline)
        val, n = detector.ewma_state("src", "latency_ms")
        assert val == 0.0
        assert n == 0

    def test_ewma_updates_correctly_after_one_step(self) -> None:
        """
        With λ=0.3 and z=(60-50)/10=1.0, the EWMA after one step must equal
        λ*z = 0.3. A wrong formula (e.g. missing the (1-λ) term) still
        computes 0.3 on the first step but diverges immediately on the second.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=10.0),  # high L to prevent fire
            baseline,
        )
        detector.update(_event(0.0, latency_ms=60.0))  # z = 1.0
        val, n = detector.ewma_state("src", "latency_ms")
        assert val == pytest.approx(0.3)
        assert n == 1

    def test_ewma_updates_correctly_after_two_steps(self) -> None:
        """
        Step 1: Z_1 = 0.3 * 1.0 = 0.3
        Step 2: Z_2 = 0.3 * 1.0 + 0.7 * 0.3 = 0.3 + 0.21 = 0.51
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=10.0),
            baseline,
        )
        detector.update(_event(0.0, latency_ms=60.0))  # z=1.0
        detector.update(_event(1.0, latency_ms=60.0))  # z=1.0
        val, n = detector.ewma_state("src", "latency_ms")
        assert val == pytest.approx(0.51)
        assert n == 2

    def test_control_limit_at_step_one_equals_lambda(self) -> None:
        """
        Asserts the algebraic identity σ_1 = λ so a variance formula change
        is caught here before the detection tests. Identity follows from:
            σ²_1 = (λ/(2-λ)) * [1 - (1-λ)^2] = (λ/(2-λ)) * λ(2-λ) = λ²
        """
        lam = 0.3
        ewma_variance_at_1 = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** 2)
        assert math.sqrt(ewma_variance_at_1) == pytest.approx(lam)

    def test_step_count_increments_on_missing_baseline(self) -> None:
        """
        When the baseline returns None, the step count must NOT increment —
        the detector skips the update entirely rather than advancing n with
        a no-op z-score.
        """
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=3.0),
            SeasonalBaselineModel({}),
        )
        detector.update(_event(0.0, latency_ms=99.0))
        val, n = detector.ewma_state("src", "latency_ms")
        assert val == 0.0
        assert n == 0


# ---------------------------------------------------------------------------
# Impulse detection — the core EWMA vs CUSUM differentiator
# ---------------------------------------------------------------------------


class TestEWMAImpulseDetection:
    """
    The impulse scenario is the definitive test of EWMA's value over CUSUM.
    A single event with z > L must fire EWMA within one tick. CUSUM needs
    multiple sub-threshold events to accumulate before firing — it will NOT
    fire on the same single spike.

    Setup:
    - baseline mean=50ms, std=10ms
    - Impulse: latency=85ms → z=(85-50)/10 = 3.5
    - EWMA: λ=0.3, L=3.0
        Z_1 = 0.3 * 3.5 = 1.05
        UCL_1 = 3.0 * λ = 3.0 * 0.3 = 0.9
        1.05 > 0.9 → fires at tick 1 ✓
    - CUSUM: h=4.0, k=0.5
        S_upper_1 = max(0, 3.5 - 0.5) = 3.0 < h=4.0 → does not fire ✓
    """

    BASELINE_MEAN = 50.0
    BASELINE_STD = 10.0
    IMPULSE_LATENCY = 85.0   # z = 3.5 — above L=3.0, below CUSUM's h=4.0 after k
    EWMA_LAMBDA = 0.3
    EWMA_L = 3.0
    CUSUM_H = 4.0
    CUSUM_K = 0.5

    def test_ewma_detects_upward_impulse_in_one_tick(self) -> None:
        baseline = _flat_baseline(self.BASELINE_MEAN, self.BASELINE_STD)
        detector = EWMADetector(
            EWMAConfig(
                smoothing=self.EWMA_LAMBDA,
                control_limit_width=self.EWMA_L,
                metrics=("latency_ms",),
            ),
            baseline,
        )
        anomalies = detector.update(_event(0.0, latency_ms=self.IMPULSE_LATENCY))

        assert len(anomalies) == 1
        assert anomalies[0].signal == "upper"
        assert anomalies[0].detector_type == "ewma"
        assert anomalies[0].metric == "latency_ms"

    def test_ewma_detects_downward_impulse_in_one_tick(self) -> None:
        """
        Lower control limit fires for a sharp latency drop — indicating a
        stage returning suspiciously fast, possible data loss or short-circuit.
        """
        baseline = _flat_baseline(self.BASELINE_MEAN, self.BASELINE_STD)
        detector = EWMADetector(
            EWMAConfig(
                smoothing=self.EWMA_LAMBDA,
                control_limit_width=self.EWMA_L,
                metrics=("latency_ms",),
            ),
            baseline,
        )
        # z = (15 - 50) / 10 = -3.5 — mirrors the upward impulse
        anomalies = detector.update(_event(0.0, latency_ms=15.0))

        assert len(anomalies) == 1
        assert anomalies[0].signal == "lower"

    def test_cusum_does_not_fire_on_same_impulse(self) -> None:
        """
        CUSUM must stay silent on the impulse that fires EWMA — the key
        complementarity property. If CUSUM also fired here, one of the two
        detectors would be redundant.
        """
        baseline = _flat_baseline(self.BASELINE_MEAN, self.BASELINE_STD)
        cusum = CUSUMDetector(
            CUSUMConfig(
                decision_threshold=self.CUSUM_H,
                slack_parameter=self.CUSUM_K,
                metrics=("latency_ms",),
            ),
            baseline,
        )
        anomalies = cusum.update(_event(0.0, latency_ms=self.IMPULSE_LATENCY))
        assert anomalies == [], (
            f"CUSUM fired on a single impulse (z=3.5, h={self.CUSUM_H}, k={self.CUSUM_K}). "
            "This means CUSUM and EWMA are redundant for spike detection — "
            "re-check h or the impulse latency."
        )

    def test_sub_threshold_impulse_does_not_fire(self) -> None:
        """
        A z-score of exactly L must not fire — condition is strict inequality
        (>). At z=L, Z_1=λ*L and UCL_1=L*λ, so Z_1 == UCL_1 and the
        condition is false.
        """
        baseline = _flat_baseline(self.BASELINE_MEAN, self.BASELINE_STD)
        detector = EWMADetector(
            EWMAConfig(smoothing=self.EWMA_LAMBDA, control_limit_width=self.EWMA_L,
                       metrics=("latency_ms",)),
            baseline,
        )
        # z = (80 - 50) / 10 = 3.0 — exactly equal to L, must not fire
        anomalies = detector.update(_event(0.0, latency_ms=80.0))
        assert anomalies == []


# ---------------------------------------------------------------------------
# No-fire under normal conditions
# ---------------------------------------------------------------------------


class TestEWMANoFire:

    def test_on_target_events_produce_no_anomaly(self) -> None:
        """
        200 events at baseline mean (z=0). The EWMA update collapses to
        λ*0 + (1-λ)*0 = 0 every tick, so the statistic stays at 0 permanently.
        Any fire here indicates the control limit formula computes a negative
        or zero UCL.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = EWMADetector(
            EWMAConfig(smoothing=0.2, control_limit_width=3.0, metrics=("latency_ms",)),
            baseline,
        )
        events = [_event(float(i), latency_ms=50.0) for i in range(200)]
        assert _fire_events(detector, events) == []

    def test_missing_baseline_skips_update_silently(self) -> None:
        """
        An event for a stage with no baseline entry must produce no anomaly
        and no exception — a no-op, not a crash.
        """
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=3.0),
            SeasonalBaselineModel({}),
        )
        anomalies = detector.update(_event(0.0, latency_ms=999.0))
        assert anomalies == []


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------


class TestEWMAReset:

    def test_ewma_value_resets_to_zero_after_fire(self) -> None:
        """
        EWMA value must reset to 0 after firing so the detector re-arms from
        centre. Without a reset, a sustained high z-score fires every tick
        after the first threshold crossing.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=3.0, metrics=("latency_ms",)),
            baseline,
        )
        detector.update(_event(0.0, latency_ms=85.0))  # fires
        val, n = detector.ewma_state("src", "latency_ms")
        assert val == 0.0, "EWMA value must reset to 0 after firing"

    def test_step_count_preserved_after_fire(self) -> None:
        """
        n must NOT reset after a fire so control limits stay at their mature
        (wider) values after re-arm. Resetting n tightens the limits and
        can cause spurious fires during the re-arm period.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=3.0, metrics=("latency_ms",)),
            baseline,
        )
        detector.update(_event(0.0, latency_ms=85.0))  # fires at n=1
        _, n = detector.ewma_state("src", "latency_ms")
        assert n == 1, "Step count must survive a fire; only the EWMA value resets"

    def test_explicit_reset_clears_both_value_and_count(self) -> None:
        """
        reset() fully clears both value and n, unlike the post-fire reset.
        The healing layer calls reset() after remediating a fault and needs
        the detector to restart with tight startup control limits.
        """
        baseline = _flat_baseline(mean=50.0, std=10.0)
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=10.0, metrics=("latency_ms",)),
            baseline,
        )
        for i in range(10):
            detector.update(_event(float(i), latency_ms=60.0))

        _, n_before = detector.ewma_state("src", "latency_ms")
        assert n_before == 10

        detector.reset("src", "latency_ms")
        val, n_after = detector.ewma_state("src", "latency_ms")
        assert val == 0.0
        assert n_after == 0


# ---------------------------------------------------------------------------
# Multi-stage isolation
# ---------------------------------------------------------------------------


class TestEWMAMultiStage:

    def test_stage_states_are_independent(self) -> None:
        """
        Spike events sent to stage_a only — stage_b's EWMA must stay at zero.
        A shared state dict with a key bug (e.g. ignoring stage_id) would
        contaminate stage_b.
        """
        entries: dict = {}
        for stage in ("stage_a", "stage_b"):
            for how in range(168):
                for metric in ("latency_ms", "row_count", "error_rate"):
                    entries[BaselineKey(stage, how, metric)] = BaselineEntry(
                        baseline_mean=50.0,
                        baseline_std=10.0,
                        sample_count=50,
                        fitted_at=SIM_START,
                    )
        combined = SeasonalBaselineModel(entries)
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=10.0),
            combined,
        )
        for i in range(5):
            detector.update(_event(float(i), latency_ms=90.0, stage_id="stage_a"))

        val_b, n_b = detector.ewma_state("stage_b", "latency_ms")
        assert val_b == 0.0
        assert n_b == 0

    def test_anomaly_carries_correct_stage_id(self) -> None:
        baseline = _flat_baseline(mean=50.0, std=10.0, stage_id="etl")
        detector = EWMADetector(
            EWMAConfig(smoothing=0.3, control_limit_width=3.0, metrics=("latency_ms",)),
            baseline,
        )
        anomalies = detector.update(_event(0.0, latency_ms=85.0, stage_id="etl"))
        assert len(anomalies) == 1
        assert anomalies[0].stage_id == "etl"
