"""
Detection accuracy benchmark for CUSUM and EWMA detectors.

I'm running this as a pytest module rather than a standalone script so it
integrates with CI naturally — the same command that runs unit tests also
gates on detection quality. The trade-off is slightly slower test collection
for the unit test suite; I'm accepting it because the benchmark runs in under
2 seconds and does not require containers.

The benchmark exercises all six fault types from Milestone 3 with known
magnitudes, then asserts recall ≥ 0.90 and FPR ≤ 0.05 for each combination
of (detector, fault_type). As a side effect it writes the full results table
to docs/detection_benchmark.md so the output is inspectable without re-running
pytest.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from detection.benchmark import BenchmarkConfig, DetectorBenchmark
from simulator.fault_injection import FAULT_TYPES

DOCS_DIR = Path(__file__).parent.parent / "docs"
REPORT_PATH = DOCS_DIR / "detection_benchmark.md"

# I'm using the default BenchmarkConfig here rather than parameterising the
# test, because the goal is asserting a fixed quality bar, not exploring the
# sensitivity surface. If the defaults ever need changing (e.g. fault magnitude
# proves too easy or too hard to detect), the config change is the breaking
# signal that prompts a code review.
_BENCHMARK_CONFIG = BenchmarkConfig()


@pytest.fixture(scope="module")
def benchmark_report():
    """
    I'm running the benchmark once per module and sharing the report across
    all tests to avoid re-running 3,000 events six times. Module scope is
    correct here because the benchmark is stateless — the same config always
    produces the same report.
    """
    bench = DetectorBenchmark(_BENCHMARK_CONFIG)
    report = bench.run()

    DOCS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report.to_markdown(), encoding="utf-8")

    return report


class TestBenchmarkThresholds:

    @pytest.mark.parametrize("fault_type", FAULT_TYPES)
    def test_cusum_recall(self, benchmark_report, fault_type: str) -> None:
        """
        I'm asserting CUSUM recall per fault type rather than a single aggregate
        because an aggregate ≥ 0.90 could mask a complete miss on one fault type
        (e.g. recall=1.0 on 5 types, 0.40 on one → aggregate=0.93 but the sixth
        type is broken). Per-type assertions catch exactly that failure mode.
        """
        result = next(
            r for r in benchmark_report.results
            if r.detector_type == "cusum" and r.fault_type == fault_type
        )
        assert result.recall >= 0.90, (
            f"CUSUM recall for '{fault_type}' is {result.recall:.2f}, "
            f"below the 0.90 threshold. "
            f"Detected {result.detected_trials}/{result.total_trials} trials. "
            "Check fault magnitude or decision_threshold configuration."
        )

    @pytest.mark.parametrize("fault_type", FAULT_TYPES)
    def test_ewma_recall(self, benchmark_report, fault_type: str) -> None:
        result = next(
            r for r in benchmark_report.results
            if r.detector_type == "ewma" and r.fault_type == fault_type
        )
        assert result.recall >= 0.90, (
            f"EWMA recall for '{fault_type}' is {result.recall:.2f}, "
            f"below the 0.90 threshold. "
            f"Detected {result.detected_trials}/{result.total_trials} trials."
        )

    @pytest.mark.parametrize("fault_type", FAULT_TYPES)
    def test_cusum_false_positive_rate(self, benchmark_report, fault_type: str) -> None:
        """
        I'm asserting FPR separately from recall because they capture different
        failure modes. A detector tuned to always fire (recall=1.0) trivially
        passes the recall test but fails FPR. Both gates must hold independently.
        """
        result = next(
            r for r in benchmark_report.results
            if r.detector_type == "cusum" and r.fault_type == fault_type
        )
        assert result.false_positive_rate <= 0.05, (
            f"CUSUM FPR for '{fault_type}' is {result.false_positive_rate:.3f}, "
            f"above the 0.05 threshold. "
            f"{result.non_fault_fires} false alarms in {result.non_fault_events} non-fault events. "
            "Consider increasing decision_threshold or slack_parameter."
        )

    @pytest.mark.parametrize("fault_type", FAULT_TYPES)
    def test_ewma_false_positive_rate(self, benchmark_report, fault_type: str) -> None:
        result = next(
            r for r in benchmark_report.results
            if r.detector_type == "ewma" and r.fault_type == fault_type
        )
        assert result.false_positive_rate <= 0.05, (
            f"EWMA FPR for '{fault_type}' is {result.false_positive_rate:.3f}, "
            f"above the 0.05 threshold. "
            f"{result.non_fault_fires} false alarms in {result.non_fault_events} non-fault events."
        )

    def test_report_written_to_docs(self, benchmark_report) -> None:
        """
        I'm asserting the report file exists as a smoke test that the side
        effect ran. A missing report file means CI would report passing tests
        but no audit record — operators would have no inspectable output from
        the benchmark job.
        """
        assert REPORT_PATH.exists(), f"Benchmark report not written to {REPORT_PATH}"
        content = REPORT_PATH.read_text(encoding="utf-8")
        assert "Detection Accuracy Benchmark" in content
        assert "recall" in content.lower()
