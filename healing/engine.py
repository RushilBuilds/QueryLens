from __future__ import annotations

from collections import Counter
from typing import Optional

from causal.localization import LocalizationResult
from healing.policy import HealingAction, HealingDecision, PolicyConfig
from simulator.topology import PipelineTopologyGraph


# Posterior probability thresholds for severity classification.
# Tuned against M14 benchmark results: top candidates above 0.70 correspond
# to scenarios where the engine correctly isolated the root cause with high
# confidence; 0.40–0.70 is ambiguous and warrants a less disruptive action.
_HIGH_SEVERITY_THRESHOLD = 0.70
_MEDIUM_SEVERITY_THRESHOLD = 0.40


def _derive_severity(posterior: float) -> str:
    if posterior >= _HIGH_SEVERITY_THRESHOLD:
        return "high"
    if posterior >= _MEDIUM_SEVERITY_THRESHOLD:
        return "medium"
    return "low"


def _dominant_fault_type(result: LocalizationResult) -> Optional[str]:
    """
    Returns the most frequent non-None fault_label across all evidence events.
    Uses majority vote rather than the first event because a mixed-fault window
    (e.g. a latency spike that triggers a downstream error burst) should route
    to the primary fault type, not whichever anomaly arrived first.
    """
    labels = [e.fault_label for e in result.evidence_events if e.fault_label]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


class HealingPolicyEngine:
    """
    Decision layer between causal analysis and remediation execution.
    Takes a LocalizationResult, derives fault_type/severity/stage_type, and
    selects the highest-priority matching rule from PolicyConfig.

    Kept separate from execution so the policy can be unit-tested without
    triggering any real circuit breaker or scaling action — the engine only
    returns a HealingDecision; the caller decides whether to act on it.
    """

    def __init__(
        self,
        config: PolicyConfig,
        topology: PipelineTopologyGraph,
    ) -> None:
        self._config = config
        self._topology = topology

    def select_action(self, result: LocalizationResult) -> HealingDecision:
        """
        Selects the highest-priority matching action for the given LocalizationResult.
        Always returns a HealingDecision — falls back to PAGE_OPERATOR when no
        specific rule matches (guaranteed by the catch-all in the default policy).
        """
        top = result.top_candidate
        if top is None:
            # No candidates — cannot target a stage; page the operator.
            return HealingDecision(
                action=HealingAction.PAGE_OPERATOR,
                target_stage_id="unknown",
                fault_type=_dominant_fault_type(result),
                severity="low",
                rule_matched="fallback:no_candidates",
                hypothesis_id=result.hypothesis_id,
            )

        target_stage_id, posterior = top
        severity = _derive_severity(posterior)
        fault_type = _dominant_fault_type(result)
        stage_type = self._stage_type(target_stage_id)

        rule = self._config.first_match(fault_type, severity, stage_type)

        return HealingDecision(
            action=rule.action,
            target_stage_id=target_stage_id,
            fault_type=fault_type,
            severity=severity,
            rule_matched=rule.description,
            hypothesis_id=result.hypothesis_id,
        )

    def _stage_type(self, stage_id: str) -> Optional[str]:
        """Returns the stage_type string for stage_id, or None if unknown."""
        try:
            return self._topology.get_stage(stage_id).stage_type
        except KeyError:
            return None
