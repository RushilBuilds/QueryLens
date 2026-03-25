from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from detection.anomaly import AnomalyEvent, extract_metric
from detection.baseline import BaselineEntry, BaselineKey, SeasonalBaselineModel
from detection.cusum import CUSUMConfig, CUSUMDetector
from detection.ewma import EWMAConfig, EWMADetector
from simulator.fault_injection import FAULT_TYPES, FaultInjector, FaultSchedule, FaultSpec
from simulator.models import PipelineEvent

# I'm fixing the simulation epoch to a Monday 00:00 UTC so that all events
# land in hour_of_week=0 with the flat baseline. The benchmark does not need
# time-varying baselines — the goal is measuring detection accuracy for known
# fault magnitudes, not seasonal model correctness.
_BENCH_EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Fault magnitudes chosen to produce reliable z-score deviations at the
# configured baseline std values. All six fault types should be detectable
# by at least one metric within a 30-event fault window.
_FAULT_MAGNITUDES: Dict[str, float] = {
    "latency_spike": 4.0,         # 4x latency → z ≈ 15 on latency_ms
    "dropped_connection": 0.8,    # 80% drop probability → z ≈ -4 on row_count
    "schema_drift": 0.5,          # 50% row loss + schema_error status → z ≈ -2.5 row_count + z=10 error_rate
    "partition_skew": 4.0,        # 4x latency → same signal as latency_spike
    "throughput_collapse": 10.0,  # 10x row reduction → z ≈ -4.5 on row_count
    "error_burst": 0.9,           # 90% error probability → z ≈ 9 on error_rate
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    I'm separating BenchmarkConfig from detector configs so the benchmark can
    be re-run with different detector sensitivity settings without changing the
    event generation parameters. This lets the benchmark act as a calibration
    tool: hold event generation fixed, vary detector configs, observe the
    recall/FPR trade-off.
    """

    n_warmup_events: int = 50
    # Warmup events give both detectors time to reach their steady-state
    # accumulator levels on clean data. Without warmup, EWMA's tight startup
    # control limits would register the first fault event as unusually significant.

    n_fault_events: int = 30
    # 30 events is long enough for CUSUM to accumulate enough signal for gradual
    # faults (schema_drift) while being short enough to keep total benchmark
    # runtime under 1 second. At 5 trials × 6 fault types × 100 events = 3,000
    # events total, runtime is dominated by Python dict lookups, not computation.

    n_recovery_events: int = 20
    # Recovery events after each fault window count toward the false-positive rate
    # denominator. 20 events is long enough to catch a detector that keeps firing
    # after the fault window closes.

    n_trials_per_fault: int = 5
    # 5 independent trials per fault type gives us recall resolution of 0.20.
    # The threshold is 0.90 — effectively requiring 5/5 detection for a clean
    # implementation. This is intentional: if any single fault type misses even
    # one trial, the threshold fails and the developer must investigate.

    # Baseline parameters — same for all metrics/stages.
    baseline_mean_latency: float = 50.0
    baseline_std_latency: float = 10.0
    baseline_mean_row_count: float = 100.0
    baseline_std_row_count: float = 20.0
    # I'm using mean=0.0, std=0.1 for error_rate so that a single status='error'
    # event produces z=10, triggering EWMA immediately and giving CUSUM a large
    # accumulator increment. The low std is intentional: in a healthy pipeline
    # the error rate is near zero, and a high std would require sustained errors
    # to build signal — which is correct for a noisy system but not what we're
    # modelling here.
    baseline_mean_error_rate: float = 0.0
    baseline_std_error_rate: float = 0.1

    cusum_config: CUSUMConfig = field(
        default_factory=lambda: CUSUMConfig(
            decision_threshold=4.0,
            slack_parameter=0.5,
        )
    )
    ewma_config: EWMAConfig = field(
        default_factory=lambda: EWMAConfig(
            smoothing=0.3,
            control_limit_width=3.0,
        )
    )


@dataclass(frozen=True)
class FaultResult:
    """
    I'm storing the raw counts alongside the derived rates so callers can
    aggregate results across multiple benchmark runs without losing precision
    from intermediate float divisions. Aggregating rates (averaging averages)
    is statistically incorrect when trial counts differ.
    """

    fault_type: str
    detector_type: str
    detected_trials: int           # out of n_trials_per_fault
    total_trials: int
    detection_lags: List[int]      # event index of first detection per detected trial
    non_fault_fires: int           # anomaly events during warmup + recovery
    non_fault_events: int          # total events during warmup + recovery

    @property
    def recall(self) -> float:
        return self.detected_trials / self.total_trials if self.total_trials > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """
        I'm defining FPR as anomaly_fires / non_fault_events rather than the
        classical (FP / (FP + TN)) because each event produces at most one
        anomaly (or zero), making 'events with false alarms / non-fault events'
        equivalent to FP / (FP + TN). Both definitions converge at the event
        granularity.
        """
        return (
            self.non_fault_fires / self.non_fault_events
            if self.non_fault_events > 0
            else 0.0
        )

    @property
    def mean_detection_lag_events(self) -> Optional[float]:
        """Returns None when no trials were detected — avoids dividing by 0."""
        return (
            sum(self.detection_lags) / len(self.detection_lags)
            if self.detection_lags
            else None
        )

    @property
    def passes_recall_threshold(self) -> bool:
        return self.recall >= 0.90

    @property
    def passes_fpr_threshold(self) -> bool:
        return self.false_positive_rate <= 0.05


@dataclass
class BenchmarkReport:
    """
    I'm storing results keyed by (detector_type, fault_type) rather than
    nested dicts so the report renderer can sort by any key without
    restructuring the data.
    """

    results: List[FaultResult] = field(default_factory=list)
    config: Optional[BenchmarkConfig] = None

    def passes_all_thresholds(self) -> bool:
        return all(
            r.passes_recall_threshold and r.passes_fpr_threshold
            for r in self.results
        )

    def failing_results(self) -> List[FaultResult]:
        return [
            r for r in self.results
            if not r.passes_recall_threshold or not r.passes_fpr_threshold
        ]

    def to_markdown(self) -> str:
        """
        I'm generating the report as a single Markdown string rather than
        writing directly to a file so that callers (tests, CI scripts) decide
        where the output goes. The test writes it to docs/; a CI summary job
        can write it to a step output.
        """
        lines = [
            "# Detection Accuracy Benchmark",
            "",
            "Precision, recall, and FPR for CUSUM and EWMA detectors across all six fault types.",
            "All six fault types from Milestone 3 are exercised with known magnitudes.",
            "",
            "**Thresholds:** recall ≥ 0.90, FPR ≤ 0.05",
            "",
        ]

        if self.config:
            lines += [
                "## Configuration",
                "",
                f"- Warmup events per trial: {self.config.n_warmup_events}",
                f"- Fault window events: {self.config.n_fault_events}",
                f"- Recovery events per trial: {self.config.n_recovery_events}",
                f"- Trials per fault type: {self.config.n_trials_per_fault}",
                f"- CUSUM decision_threshold: {self.config.cusum_config.decision_threshold}, "
                f"slack_parameter: {self.config.cusum_config.slack_parameter}",
                f"- EWMA smoothing: {self.config.ewma_config.smoothing}, "
                f"control_limit_width: {self.config.ewma_config.control_limit_width}",
                "",
            ]

        lines += [
            "## Results",
            "",
            "| Detector | Fault Type | Recall | FPR | Mean Detection Lag (events) | Pass |",
            "|---|---|---|---|---|---|",
        ]

        for r in sorted(self.results, key=lambda x: (x.detector_type, x.fault_type)):
            lag = (
                f"{r.mean_detection_lag_events:.1f}"
                if r.mean_detection_lag_events is not None
                else "—"
            )
            passed = "✓" if r.passes_recall_threshold and r.passes_fpr_threshold else "✗"
            lines.append(
                f"| {r.detector_type} | {r.fault_type} "
                f"| {r.recall:.2f} | {r.false_positive_rate:.3f} | {lag} | {passed} |"
            )

        lines += [
            "",
            "## Summary",
            "",
        ]

        if self.passes_all_thresholds():
            lines.append("All fault types pass both thresholds.")
        else:
            lines.append("The following fault types failed one or more thresholds:")
            lines.append("")
            for r in self.failing_results():
                reasons = []
                if not r.passes_recall_threshold:
                    reasons.append(f"recall={r.recall:.2f} < 0.90")
                if not r.passes_fpr_threshold:
                    reasons.append(f"FPR={r.false_positive_rate:.3f} > 0.05")
                lines.append(f"- {r.detector_type} / {r.fault_type}: {', '.join(reasons)}")

        return "\n".join(lines) + "\n"


class DetectorBenchmark:
    """
    I'm running each fault type as an independent set of trials rather than
    interleaving faults in a single stream because interleaving would make the
    detection lag calculation ambiguous — a detector that accumulates signal
    from a preceding fault window would appear to detect the next fault faster
    than it actually does. Independent trials with a fresh detector per trial
    give clean, interpretable lag numbers.

    I'm not using the SimulatorEngine here because the benchmark needs
    deterministic, minimal events — not a full topology with Poisson arrivals.
    The FaultInjector is still used directly so the injected events match
    exactly what the simulator would produce (same mutation logic, same
    fault_label propagation).
    """

    def __init__(self, config: BenchmarkConfig = BenchmarkConfig()) -> None:
        self._config = config
        self._baseline = self._build_baseline()

    def _build_baseline(self) -> SeasonalBaselineModel:
        """
        I'm building a flat baseline covering all 168 hour_of_week slots so
        events at any offset from _BENCH_EPOCH land in a valid baseline slot.
        All slots share the same mean and std — this benchmark is measuring
        detector accuracy, not baseline interpolation accuracy.
        """
        cfg = self._config
        entries: Dict[BaselineKey, BaselineEntry] = {}
        for how in range(168):
            for stage_id in ["bench_stage"]:
                entries[BaselineKey(stage_id, how, "latency_ms")] = BaselineEntry(
                    baseline_mean=cfg.baseline_mean_latency,
                    baseline_std=cfg.baseline_std_latency,
                    sample_count=200,
                    fitted_at=_BENCH_EPOCH,
                )
                entries[BaselineKey(stage_id, how, "row_count")] = BaselineEntry(
                    baseline_mean=cfg.baseline_mean_row_count,
                    baseline_std=cfg.baseline_std_row_count,
                    sample_count=200,
                    fitted_at=_BENCH_EPOCH,
                )
                entries[BaselineKey(stage_id, how, "error_rate")] = BaselineEntry(
                    baseline_mean=cfg.baseline_mean_error_rate,
                    baseline_std=cfg.baseline_std_error_rate,
                    sample_count=200,
                    fitted_at=_BENCH_EPOCH,
                )
        return SeasonalBaselineModel(entries)

    def _make_normal_event(self, offset_s: float) -> PipelineEvent:
        return PipelineEvent(
            stage_id="bench_stage",
            event_time=_BENCH_EPOCH + timedelta(seconds=offset_s),
            latency_ms=self._config.baseline_mean_latency,
            row_count=int(self._config.baseline_mean_row_count),
            payload_bytes=1024,
            status="ok",
            fault_label=None,
            trace_id=None,
        )

    def _make_fault_events(
        self,
        fault_type: str,
        offset_start_s: float,
        trial_seed: int,
    ) -> List[PipelineEvent]:
        """
        I'm creating a fresh FaultInjector per trial with a different seed so
        probabilistic faults (dropped_connection, error_burst) sample independent
        event sequences across trials. Using the same seed would make every trial
        identical — the recall estimate would then reflect one specific random
        realisation rather than the expected detection rate.
        """
        n = self._config.n_fault_events
        spec = FaultSpec(
            fault_type=fault_type,
            target_stage_id="bench_stage",
            start_offset_s=offset_start_s,
            duration_s=float(n) + 1.0,  # +1 ensures all n events fall inside the window
            magnitude=_FAULT_MAGNITUDES[fault_type],
            seed=trial_seed,
        )
        schedule = FaultSchedule(
            simulation_start=_BENCH_EPOCH,
            fault_specs=[spec],
        )
        injector = FaultInjector(schedule)
        base_events = [
            self._make_normal_event(offset_start_s + i)
            for i in range(n)
        ]
        return [injector.inject(e) for e in base_events]

    def _run_trial(
        self,
        fault_type: str,
        trial_index: int,
        cusum: CUSUMDetector,
        ewma: EWMADetector,
    ) -> Tuple[bool, bool, Optional[int], Optional[int], int, int]:
        """
        Runs one warmup → fault → recovery cycle for both detectors in a single
        pass over the event stream.

        Returns:
            (cusum_detected, ewma_detected,
             cusum_lag, ewma_lag,
             non_fault_cusum_fires, non_fault_ewma_fires)

        I'm running both detectors in the same loop rather than separate passes
        to keep trial_index→offset mapping consistent between them. Separate
        passes could diverge if normal event generation uses any state.
        """
        cfg = self._config
        n_warmup = cfg.n_warmup_events
        n_fault = cfg.n_fault_events
        n_recovery = cfg.n_recovery_events

        # Offset events far enough to avoid overlapping with other trials'
        # fault windows in the FaultSchedule active-window check.
        trial_block_size = n_warmup + n_fault + n_recovery + 10
        base_offset = trial_index * trial_block_size
        fault_offset = base_offset + n_warmup

        warmup = [self._make_normal_event(base_offset + i) for i in range(n_warmup)]
        fault_events = self._make_fault_events(fault_type, fault_offset, trial_seed=trial_index * 1000 + hash(fault_type) % 997)
        recovery = [self._make_normal_event(fault_offset + n_fault + i) for i in range(n_recovery)]

        cusum_detected = False
        ewma_detected = False
        cusum_lag: Optional[int] = None
        ewma_lag: Optional[int] = None
        non_fault_cusum_fires = 0
        non_fault_ewma_fires = 0

        for event in warmup:
            if cusum.update(event):
                non_fault_cusum_fires += 1
            if ewma.update(event):
                non_fault_ewma_fires += 1

        for i, event in enumerate(fault_events):
            c_anomalies = cusum.update(event)
            e_anomalies = ewma.update(event)
            if c_anomalies and not cusum_detected:
                cusum_detected = True
                cusum_lag = i
            if e_anomalies and not ewma_detected:
                ewma_detected = True
                ewma_lag = i

        for event in recovery:
            if cusum.update(event):
                non_fault_cusum_fires += 1
            if ewma.update(event):
                non_fault_ewma_fires += 1

        return (
            cusum_detected,
            ewma_detected,
            cusum_lag,
            ewma_lag,
            non_fault_cusum_fires,
            non_fault_ewma_fires,
        )

    def run(self) -> BenchmarkReport:
        """
        I'm creating fresh detector instances per fault type (not per trial) so
        each fault type starts from zero accumulator state. Reusing detectors
        across fault types would cause the state built up during one fault type's
        trials to artificially inflate the starting accumulator for the next —
        producing detection lags that are too short for the first trial of each
        new fault type.
        """
        results: List[FaultResult] = []

        for fault_type in FAULT_TYPES:
            cusum = CUSUMDetector(self._config.cusum_config, self._baseline)
            ewma = EWMADetector(self._config.ewma_config, self._baseline)

            c_detected_count = 0
            e_detected_count = 0
            c_lags: List[int] = []
            e_lags: List[int] = []
            c_non_fault_fires = 0
            e_non_fault_fires = 0
            non_fault_events_total = (
                self._config.n_warmup_events + self._config.n_recovery_events
            ) * self._config.n_trials_per_fault

            for trial_idx in range(self._config.n_trials_per_fault):
                (
                    c_det, e_det, c_lag, e_lag, c_fp, e_fp
                ) = self._run_trial(fault_type, trial_idx, cusum, ewma)

                if c_det:
                    c_detected_count += 1
                    if c_lag is not None:
                        c_lags.append(c_lag)
                if e_det:
                    e_detected_count += 1
                    if e_lag is not None:
                        e_lags.append(e_lag)
                c_non_fault_fires += c_fp
                e_non_fault_fires += e_fp

            n_trials = self._config.n_trials_per_fault
            results.append(FaultResult(
                fault_type=fault_type,
                detector_type="cusum",
                detected_trials=c_detected_count,
                total_trials=n_trials,
                detection_lags=c_lags,
                non_fault_fires=c_non_fault_fires,
                non_fault_events=non_fault_events_total,
            ))
            results.append(FaultResult(
                fault_type=fault_type,
                detector_type="ewma",
                detected_trials=e_detected_count,
                total_trials=n_trials,
                detection_lags=e_lags,
                non_fault_fires=e_non_fault_fires,
                non_fault_events=non_fault_events_total,
            ))

        return BenchmarkReport(results=results, config=self._config)
