"""
Unit tests for HealingPolicyEngine, PolicyConfig, and PolicyRule.

All tests are pure in-memory — no containers required. The topology used is
a 3-stage linear chain: source_pg → transform_enrich → sink_warehouse, which
covers all three stage_type values in a single fixture.

Key scenarios tested:
  - Each of the 6 fault types routes to the expected action at high severity
  - Severity thresholds control which rule is selected (high vs non-high)
  - PAGE_OPERATOR fires when no specific rule matches
  - PAGE_OPERATOR fires when LocalizationResult has no candidates
  - PolicyRule.matches() wildcard semantics
  - PolicyConfig loads the default YAML and preserves priority order
  - dominant fault_type uses majority vote across evidence events
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import pytest

from causal.localization import LocalizationResult
from detection.anomaly import AnomalyEvent
from healing.engine import HealingPolicyEngine, _derive_severity, _dominant_fault_type
from healing.policy import HealingAction, HealingDecision, PolicyConfig, PolicyRule
from simulator.topology import PipelineStage, PipelineTopologyGraph

T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
DEFAULT_POLICY = Path(__file__).parent.parent / "config" / "healing_policy.yaml"


# ---------------------------------------------------------------------------
# Topology fixture
# ---------------------------------------------------------------------------


def _build_topology() -> PipelineTopologyGraph:
    return PipelineTopologyGraph([
        PipelineStage("source_pg",        "source",    [],                   0.0),
        PipelineStage("transform_enrich", "transform", ["source_pg"],       10.0),
        PipelineStage("sink_warehouse",   "sink",      ["transform_enrich"], 5.0),
    ])


def _anomaly(stage_id: str, fault_label: str) -> AnomalyEvent:
    return AnomalyEvent(
        detector_type="cusum",
        stage_id=stage_id,
        metric="latency_ms",
        signal="upper",
        detector_value=5.0,
        threshold=4.0,
        z_score=3.0,
        detected_at=T0,
        fault_label=fault_label,
    )


def _result(
    stage_id: str,
    posterior: float,
    fault_label: str,
    hypothesis_id: str = "hyp-001",
) -> LocalizationResult:
    return LocalizationResult(
        hypothesis_id=hypothesis_id,
        triggered_at=T0,
        evidence_events=(_anomaly(stage_id, fault_label),),
        ranked_candidates=((stage_id, posterior),),
    )


def _engine(topology: PipelineTopologyGraph) -> HealingPolicyEngine:
    return HealingPolicyEngine(PolicyConfig(DEFAULT_POLICY), topology)


# ---------------------------------------------------------------------------
# PolicyRule.matches() — wildcard semantics
# ---------------------------------------------------------------------------


class TestPolicyRuleMatches:

    def test_all_none_matches_anything(self) -> None:
        rule = PolicyRule(action=HealingAction.PAGE_OPERATOR, priority=999)
        assert rule.matches("latency_spike", "high", "source") is True

    def test_specific_fault_type_matches_exact(self) -> None:
        rule = PolicyRule(
            action=HealingAction.CIRCUIT_BREAK, priority=1,
            fault_type="latency_spike",
        )
        assert rule.matches("latency_spike", "high", "source") is True
        assert rule.matches("error_burst",   "high", "source") is False

    def test_specific_severity_matches_exact(self) -> None:
        rule = PolicyRule(
            action=HealingAction.CIRCUIT_BREAK, priority=1, severity="high"
        )
        assert rule.matches("latency_spike", "high",   "source") is True
        assert rule.matches("latency_spike", "medium", "source") is False

    def test_specific_stage_type_matches_exact(self) -> None:
        rule = PolicyRule(
            action=HealingAction.CIRCUIT_BREAK, priority=1, stage_type="source"
        )
        assert rule.matches("latency_spike", "high", "source")    is True
        assert rule.matches("latency_spike", "high", "transform") is False

    def test_combined_fields_all_must_match(self) -> None:
        rule = PolicyRule(
            action=HealingAction.CIRCUIT_BREAK, priority=1,
            fault_type="latency_spike", severity="high", stage_type="source",
        )
        assert rule.matches("latency_spike", "high",   "source")    is True
        assert rule.matches("latency_spike", "medium", "source")    is False
        assert rule.matches("latency_spike", "high",   "transform") is False
        assert rule.matches("error_burst",   "high",   "source")    is False


# ---------------------------------------------------------------------------
# _derive_severity helper
# ---------------------------------------------------------------------------


class TestDeriveSeverity:

    def test_high_at_threshold(self) -> None:
        assert _derive_severity(0.70) == "high"

    def test_high_above_threshold(self) -> None:
        assert _derive_severity(0.99) == "high"

    def test_medium_at_threshold(self) -> None:
        assert _derive_severity(0.40) == "medium"

    def test_medium_below_high(self) -> None:
        assert _derive_severity(0.55) == "medium"

    def test_low_below_medium(self) -> None:
        assert _derive_severity(0.10) == "low"

    def test_low_at_zero(self) -> None:
        assert _derive_severity(0.0) == "low"


# ---------------------------------------------------------------------------
# _dominant_fault_type helper
# ---------------------------------------------------------------------------


class TestDominantFaultType:

    def test_single_label(self) -> None:
        result = _result("source_pg", 0.9, "latency_spike")
        assert _dominant_fault_type(result) == "latency_spike"

    def test_majority_wins(self) -> None:
        events = (
            _anomaly("source_pg", "latency_spike"),
            _anomaly("source_pg", "latency_spike"),
            _anomaly("source_pg", "error_burst"),
        )
        result = LocalizationResult(
            hypothesis_id="h",
            triggered_at=T0,
            evidence_events=events,
            ranked_candidates=(("source_pg", 0.9),),
        )
        assert _dominant_fault_type(result) == "latency_spike"

    def test_all_none_labels_returns_none(self) -> None:
        event = AnomalyEvent(
            detector_type="cusum", stage_id="src", metric="latency_ms",
            signal="upper", detector_value=5.0, threshold=4.0, z_score=3.0,
            detected_at=T0, fault_label=None,
        )
        result = LocalizationResult(
            hypothesis_id="h", triggered_at=T0,
            evidence_events=(event,), ranked_candidates=(("src", 0.9),),
        )
        assert _dominant_fault_type(result) is None


# ---------------------------------------------------------------------------
# PolicyConfig — load and priority ordering
# ---------------------------------------------------------------------------


class TestPolicyConfig:

    def test_loads_default_yaml_without_error(self) -> None:
        cfg = PolicyConfig(DEFAULT_POLICY)
        assert len(cfg.rules) > 0

    def test_rules_sorted_by_priority_ascending(self) -> None:
        cfg = PolicyConfig(DEFAULT_POLICY)
        priorities = [r.priority for r in cfg.rules]
        assert priorities == sorted(priorities)

    def test_catch_all_page_operator_present(self) -> None:
        cfg = PolicyConfig(DEFAULT_POLICY)
        last = cfg.rules[-1]
        assert last.action == HealingAction.PAGE_OPERATOR
        assert last.fault_type is None
        assert last.severity is None
        assert last.stage_type is None

    def test_first_match_no_match_raises_if_no_catchall(self) -> None:
        # A config with only a specific rule and no catch-all should raise.
        from io import StringIO
        import yaml as _yaml
        raw = {"rules": [{"priority": 1, "fault_type": "latency_spike", "action": "circuit_break"}]}
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _yaml.dump(raw, f)
            tmp = f.name
        try:
            cfg = PolicyConfig(Path(tmp))
            with pytest.raises(RuntimeError, match="catch-all"):
                cfg.first_match("error_burst", "high", "source")
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# HealingPolicyEngine — per fault type routing
# ---------------------------------------------------------------------------


class TestHealingPolicyEngineFaultTypes:

    @pytest.fixture(scope="class")
    def engine(self) -> HealingPolicyEngine:
        return _engine(_build_topology())

    def test_latency_spike_high_on_source_circuit_breaks(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.85, "latency_spike")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.CIRCUIT_BREAK

    def test_latency_spike_low_severity_rate_limits(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.20, "latency_spike")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.RATE_LIMIT

    def test_dropped_connection_high_circuit_breaks(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.75, "dropped_connection")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.CIRCUIT_BREAK

    def test_throughput_collapse_scales_consumer(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.80, "throughput_collapse")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.SCALE_CONSUMER

    def test_partition_skew_reroutes_traffic(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.72, "partition_skew")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.REROUTE_TRAFFIC

    def test_error_burst_high_circuit_breaks(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.80, "error_burst")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.CIRCUIT_BREAK

    def test_error_burst_low_rate_limits(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.25, "error_burst")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.RATE_LIMIT

    def test_schema_drift_pages_operator(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.90, "schema_drift")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.PAGE_OPERATOR

    def test_unknown_fault_type_falls_back_to_page_operator(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.50, "completely_unknown_fault")
        decision = engine.select_action(result)
        assert decision.action == HealingAction.PAGE_OPERATOR


# ---------------------------------------------------------------------------
# HealingPolicyEngine — decision fields
# ---------------------------------------------------------------------------


class TestHealingDecisionFields:

    @pytest.fixture(scope="class")
    def engine(self) -> HealingPolicyEngine:
        return _engine(_build_topology())

    def test_decision_carries_correct_stage_id(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.85, "latency_spike", "hyp-fields")
        decision = engine.select_action(result)
        assert decision.target_stage_id == "source_pg"

    def test_decision_carries_fault_type(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.85, "latency_spike", "hyp-ftype")
        decision = engine.select_action(result)
        assert decision.fault_type == "latency_spike"

    def test_decision_carries_severity(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.85, "latency_spike", "hyp-sev")
        decision = engine.select_action(result)
        assert decision.severity == "high"

    def test_decision_carries_hypothesis_id(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.85, "latency_spike", "hyp-xyz")
        decision = engine.select_action(result)
        assert decision.hypothesis_id == "hyp-xyz"

    def test_decision_carries_non_empty_rule_matched(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = _result("source_pg", 0.85, "latency_spike", "hyp-rule")
        decision = engine.select_action(result)
        assert len(decision.rule_matched) > 0

    def test_no_candidates_returns_page_operator(
        self, engine: HealingPolicyEngine
    ) -> None:
        result = LocalizationResult(
            hypothesis_id="hyp-empty",
            triggered_at=T0,
            evidence_events=(_anomaly("source_pg", "latency_spike"),),
            ranked_candidates=(),
        )
        decision = engine.select_action(result)
        assert decision.action == HealingAction.PAGE_OPERATOR
        assert decision.target_stage_id == "unknown"
