from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from detection.anomaly import AnomalyEvent, extract_metric
from detection.baseline import SeasonalBaselineModel
from simulator.models import PipelineEvent


@dataclass(frozen=True)
class CUSUMConfig:
    """
    I'm exposing both decision_threshold (h) and slack_parameter (k) as
    first-class fields rather than deriving one from the other because the
    optimal values depend on the false-positive budget, not on each other.

    The trade-off:
    - Increasing h reduces false-positive rate but increases detection lag
      (more observations needed before S_upper or S_lower exceeds h).
    - Increasing k makes the detector ignore small deviations (slack), which
      reduces false positives from minor seasonal noise but increases the lag
      for detecting a shift of exactly k standard deviations.

    Typical starting values for z-score input:
    - k = 0.5 σ  (target shift to detect: 1σ above baseline)
    - h = 4.0    (in control ARL ≈ 500 for standard normal input)

    A lower h (e.g. h = 2.0) makes the detector more sensitive but at the
    cost of roughly one false alarm per 100 observations in a healthy pipeline.
    """

    decision_threshold: float    # h — S must exceed this to fire
    slack_parameter: float       # k — allowance subtracted from each z-score
    metrics: Tuple[str, ...] = ("latency_ms", "row_count", "error_rate")

    def __post_init__(self) -> None:
        if self.decision_threshold <= 0:
            raise ValueError(
                f"decision_threshold must be > 0, got {self.decision_threshold}"
            )
        if self.slack_parameter < 0:
            raise ValueError(
                f"slack_parameter must be >= 0, got {self.slack_parameter}"
            )


class CUSUMDetector:
    """
    I'm implementing the tabular (one-sided) CUSUM applied in both directions
    (upper and lower) rather than the FIR (fast initial response) variant.
    FIR starts accumulators at h/2 to speed up detection after a reset, but
    it also increases false-positive rate immediately after every reset —
    including the frequent legitimate resets that happen when a fault clears.
    Tabular CUSUM starting from 0 after each reset is slower to re-arm but
    has a stable false-positive rate throughout its operating window.

    The update formula for z-score input (μ = 0 under baseline):
        S_upper[t] = max(0, S_upper[t-1] + z[t] - k)
        S_lower[t] = max(0, S_lower[t-1] - z[t] - k)

    Fire when S_upper > h or S_lower > h. Reset the fired accumulator to 0.

    I'm resetting only the accumulator that fired rather than both. If a
    latency spike causes S_upper to fire, S_lower may have been accumulating
    negative drift simultaneously — resetting it too would discard information
    about a concurrent downward shift in a different metric.
    """

    def __init__(
        self,
        config: CUSUMConfig,
        baseline_model: SeasonalBaselineModel,
    ) -> None:
        self._config = config
        self._baseline = baseline_model
        # Keyed by (stage_id, metric). Initialised to 0.0 on first access.
        self._s_upper: Dict[Tuple[str, str], float] = {}
        self._s_lower: Dict[Tuple[str, str], float] = {}

    def update(self, event: PipelineEvent) -> List[AnomalyEvent]:
        """
        I'm processing all configured metrics in a single update() call rather
        than requiring the caller to call update() once per metric. The caller
        (the detection loop) sees one event at a time; splitting the per-metric
        logic out would force it to loop over metrics externally and keep track
        of which detector instance owns which metric, complicating the loop.

        Returns a list of AnomalyEvents (typically empty). A single event can
        fire at most one AnomalyEvent per metric (upper or lower, not both),
        but can fire for multiple metrics simultaneously.
        """
        anomalies: List[AnomalyEvent] = []
        k = self._config.slack_parameter
        h = self._config.decision_threshold

        for metric in self._config.metrics:
            value = extract_metric(event, metric)
            z = self._baseline.z_score(
                stage_id=event.stage_id,
                event_time=event.event_time,
                metric=metric,
                value=value,
            )
            # I'm skipping the accumulator update entirely when the baseline
            # has no entry for this (stage, hour_of_week, metric) slot rather
            # than treating z=None as z=0. Accumulating z=0 would be harmless
            # numerically but would make it impossible to distinguish "no drift"
            # from "no baseline" when debugging accumulator traces.
            if z is None:
                continue

            key = (event.stage_id, metric)
            s_up = self._s_upper.get(key, 0.0)
            s_lo = self._s_lower.get(key, 0.0)

            s_up = max(0.0, s_up + z - k)
            s_lo = max(0.0, s_lo - z - k)

            if s_up > h:
                anomalies.append(AnomalyEvent(
                    detector_type="cusum",
                    stage_id=event.stage_id,
                    metric=metric,
                    signal="upper",
                    detector_value=s_up,
                    threshold=h,
                    z_score=z,
                    detected_at=event.event_time,
                    fault_label=event.fault_label,
                ))
                s_up = 0.0

            if s_lo > h:
                anomalies.append(AnomalyEvent(
                    detector_type="cusum",
                    stage_id=event.stage_id,
                    metric=metric,
                    signal="lower",
                    detector_value=s_lo,
                    threshold=h,
                    z_score=z,
                    detected_at=event.event_time,
                    fault_label=event.fault_label,
                ))
                s_lo = 0.0

            self._s_upper[key] = s_up
            self._s_lower[key] = s_lo

        return anomalies

    def accumulator_state(self, stage_id: str, metric: str) -> Tuple[float, float]:
        """Return (S_upper, S_lower) for inspection in tests and diagnostics."""
        key = (stage_id, metric)
        return self._s_upper.get(key, 0.0), self._s_lower.get(key, 0.0)

    def reset(self, stage_id: str, metric: str) -> None:
        """
        I'm exposing an explicit reset so the healing layer can clear a
        detector's state after a confirmed fault is remediated. Without a
        reset, a detector that fired on a real fault would carry a residual
        accumulator value into the recovery period, potentially firing again
        on the first legitimate event after healing — a false alarm at exactly
        the worst moment.
        """
        key = (stage_id, metric)
        self._s_upper.pop(key, None)
        self._s_lower.pop(key, None)

    def reset_all(self) -> None:
        """Clear all accumulator state — used on baseline model refresh."""
        self._s_upper.clear()
        self._s_lower.clear()
