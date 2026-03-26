from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from detection.anomaly import AnomalyEvent, extract_metric
from detection.baseline import SeasonalBaselineModel
from simulator.models import PipelineEvent


@dataclass(frozen=True)
class CUSUMConfig:
    """
    Both decision_threshold (h) and slack_parameter (k) are first-class fields
    because their optimal values depend on the false-positive budget, not on
    each other.

    Trade-offs:
    - Increasing h reduces FPR but increases detection lag.
    - Increasing k ignores small deviations, reducing seasonal noise FPs but
      increasing lag for shifts of exactly k standard deviations.

    Typical starting values for z-score input:
    - k = 0.5 σ  (target shift to detect: 1σ above baseline)
    - h = 4.0    (in-control ARL ≈ 500 for standard normal input)
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
    Tabular (one-sided) CUSUM applied in both directions rather than the FIR
    (fast initial response) variant. FIR starts accumulators at h/2 to speed
    up detection after a reset, but raises FPR immediately after every reset —
    including legitimate resets when a fault clears. Tabular CUSUM from 0 is
    slower to re-arm but has a stable FPR throughout its operating window.

    Update formula for z-score input (μ = 0 under baseline):
        S_upper[t] = max(0, S_upper[t-1] + z[t] - k)
        S_lower[t] = max(0, S_lower[t-1] - z[t] - k)

    Fire when S_upper > h or S_lower > h. Reset only the fired accumulator —
    if a spike causes S_upper to fire, S_lower may have been accumulating
    negative drift simultaneously and resetting it discards that signal.
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
        Processes all configured metrics in a single call rather than requiring
        one call per metric. The caller sees one event at a time; per-metric
        splitting would force external metric loops and complicate ownership.

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
            # Skip when no baseline rather than treating z=None as z=0.
            # Accumulating z=0 is harmless numerically but makes it impossible
            # to distinguish "no drift" from "no baseline" in accumulator traces.
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
        Explicit reset for the healing layer after a confirmed fault is
        remediated. Without a reset, a residual accumulator value carried into
        the recovery period could fire again on the first legitimate event
        after healing — a false alarm at the worst possible moment.
        """
        key = (stage_id, metric)
        self._s_upper.pop(key, None)
        self._s_lower.pop(key, None)

    def reset_all(self) -> None:
        """Clear all accumulator state — used on baseline model refresh."""
        self._s_upper.clear()
        self._s_lower.clear()
