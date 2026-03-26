from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from detection.anomaly import AnomalyEvent, extract_metric
from detection.baseline import SeasonalBaselineModel
from simulator.models import PipelineEvent


@dataclass(frozen=True)
class EWMAConfig:
    """
    Two separate parameters rather than a single 'sensitivity' knob because
    smoothing (λ) and control limit width (L) control independent things:
    λ governs reaction speed; L governs the anomaly threshold. Merging them
    makes it impossible to independently tune false-positive rate and detection
    lag.

    Trade-offs:
    - High λ (→ 1.0): collapses to a per-event z-score — maximum reactivity,
      minimum smoothing. Good for sudden spike detection.
    - Low λ (→ 0.0): heavy smoothing; impulses attenuated but sustained shifts
      accumulate, approaching CUSUM behaviour at the limit.
    - L = 3.0 is the standard Shewhart 3σ boundary; lower L increases
      sensitivity at the cost of more false positives.

    Recommended starting point for pipeline latency:
      smoothing=0.2, control_limit_width=3.0
    """

    smoothing: float           # λ — must be in (0, 1]
    control_limit_width: float # L — multiplier on EWMA standard deviation
    metrics: Tuple[str, ...] = ("latency_ms", "row_count", "error_rate")

    def __post_init__(self) -> None:
        if not (0.0 < self.smoothing <= 1.0):
            raise ValueError(
                f"smoothing must be in (0, 1], got {self.smoothing}"
            )
        if self.control_limit_width <= 0.0:
            raise ValueError(
                f"control_limit_width must be > 0, got {self.control_limit_width}"
            )


class EWMADetector:
    """
    Applies the Lucas & Saccucci (1990) EWMA control chart to seasonal z-scores
    rather than raw metric values. Z-score input means L has a consistent
    interpretation across all metrics: L=3 always means "3 EWMA standard
    deviations from seasonal expected value" for both latency_ms and row_count.

    Update formula (applied to z-score input):
        Z_t = λ * z_t + (1 - λ) * Z_{t-1},   Z_0 = 0

    Exact-variance control limit (Lucas & Saccucci 1990, eq. 4):
        σ²_t = (λ / (2 - λ)) * [1 - (1 - λ)^(2t)]
        UCL_t = L * σ_t,   LCL_t = -L * σ_t

    At t=1: σ_1 = λ (algebraically exact), so a single event with |z| > L fires
    immediately — EWMA is equivalent to a z-score alarm at step 1, then smooths
    transients as t grows without the startup false-positive rate of a fixed-limit
    chart.

    CUSUM fires on sustained drift; EWMA reacts within one tick to a sharp spike
    but will not fire on a gradual ramp whose per-event z scores are all below L.
    Running both in parallel covers the full fault taxonomy.

    Z is reset to 0 after a fire (n intact) so the detector re-arms from centre
    but retains mature control limits. Resetting n too would collapse limits back
    to tight startup values, causing spurious fires during the re-arm period.
    """

    def __init__(
        self,
        config: EWMAConfig,
        baseline_model: SeasonalBaselineModel,
    ) -> None:
        self._config = config
        self._baseline = baseline_model
        # (stage_id, metric) → (ewma_value, step_count)
        # step_count drives the exact-variance control limit formula.
        self._state: Dict[Tuple[str, str], Tuple[float, int]] = {}

    def update(self, event: PipelineEvent) -> List[AnomalyEvent]:
        """
        Processes all configured metrics per call so the caller loop doesn't
        track metric ownership per detector instance.

        Returns a list of AnomalyEvents — empty under normal conditions.
        A single event can fire at most one AnomalyEvent per metric (upper or
        lower) but may fire for multiple metrics simultaneously.
        """
        lam = self._config.smoothing
        L = self._config.control_limit_width
        fired: List[AnomalyEvent] = []

        for metric in self._config.metrics:
            value = extract_metric(event, metric)
            z = self._baseline.z_score(
                stage_id=event.stage_id,
                event_time=event.event_time,
                metric=metric,
                value=value,
            )
            # Skip when no baseline — accumulating z=None as 0 would produce a
            # misleading EWMA trace that drifts toward 0 rather than staying at
            # its last valid value, causing spurious lower-signal fires after a
            # baseline refresh.
            if z is None:
                continue

            key = (event.stage_id, metric)
            ewma_val, n = self._state.get(key, (0.0, 0))

            ewma_val = lam * z + (1.0 - lam) * ewma_val
            n += 1

            # Exact-variance formula. At n=1 this reduces to σ=λ (see class
            # docstring), so the first tick is equivalent to a z-score test.
            ewma_variance = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * n))
            ewma_std = math.sqrt(max(0.0, ewma_variance))
            control_limit = L * ewma_std

            # Store updated EWMA before checking — we want n to reflect the
            # processed event count even if no anomaly fires.
            self._state[key] = (ewma_val, n)

            if control_limit == 0.0:
                # This only happens at n=0 before any update, which can't
                # occur here. Guard kept for numerical safety.
                continue

            if ewma_val > control_limit:
                fired.append(AnomalyEvent(
                    detector_type="ewma",
                    stage_id=event.stage_id,
                    metric=metric,
                    signal="upper",
                    detector_value=ewma_val,
                    threshold=control_limit,
                    z_score=z,
                    detected_at=event.event_time,
                    fault_label=event.fault_label,
                ))
                self._state[key] = (0.0, n)
            elif ewma_val < -control_limit:
                fired.append(AnomalyEvent(
                    detector_type="ewma",
                    stage_id=event.stage_id,
                    metric=metric,
                    signal="lower",
                    detector_value=ewma_val,
                    threshold=control_limit,
                    z_score=z,
                    detected_at=event.event_time,
                    fault_label=event.fault_label,
                ))
                self._state[key] = (0.0, n)

        return fired

    def ewma_state(self, stage_id: str, metric: str) -> Tuple[float, int]:
        """Return (ewma_value, step_count) for inspection in tests and diagnostics."""
        return self._state.get((stage_id, metric), (0.0, 0))

    def reset(self, stage_id: str, metric: str) -> None:
        """
        Explicit reset for the healing layer to clear state after a confirmed
        fault is remediated — same contract as CUSUMDetector.reset(). Unlike
        the post-fire reset (which keeps n), this also resets n so control
        limits restart from tight startup values on the next event.
        """
        self._state.pop((stage_id, metric), None)

    def reset_all(self) -> None:
        """Clear all EWMA state — used on baseline model refresh."""
        self._state.clear()
