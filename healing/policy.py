from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml


class HealingAction(Enum):
    CIRCUIT_BREAK    = "circuit_break"
    RATE_LIMIT       = "rate_limit"
    REPLAY_RANGE     = "replay_range"
    REROUTE_TRAFFIC  = "reroute_traffic"
    SCALE_CONSUMER   = "scale_consumer"
    PAGE_OPERATOR    = "page_operator"


@dataclass(frozen=True)
class PolicyRule:
    """
    A single match arm in the policy table. None on any field is a wildcard —
    a rule with fault_type=None and severity=None matches every input, so a
    catch-all PAGE_OPERATOR rule at the bottom of the priority list acts as
    the fallback without requiring a separate code path.

    priority is an integer where lower values are evaluated first. Rules with
    the same priority are evaluated in load order; the first match wins.
    """

    action: HealingAction
    priority: int
    fault_type: Optional[str] = None   # None matches any fault_type
    severity: Optional[str] = None     # None matches any: "high"/"medium"/"low"
    stage_type: Optional[str] = None   # None matches any: "source"/"transform"/"sink"
    description: str = ""

    def matches(
        self,
        fault_type: Optional[str],
        severity: str,
        stage_type: Optional[str],
    ) -> bool:
        """Returns True if all specified fields match the given inputs."""
        if self.fault_type is not None and self.fault_type != fault_type:
            return False
        if self.severity is not None and self.severity != severity:
            return False
        if self.stage_type is not None and self.stage_type != stage_type:
            return False
        return True


@dataclass(frozen=True)
class HealingDecision:
    """
    Output of the HealingPolicyEngine. Carries enough context for the
    HealingAuditLog (M22) to record what was decided and why without
    re-querying the localization or policy tables.
    """

    action: HealingAction
    target_stage_id: str
    fault_type: Optional[str]
    severity: str
    rule_matched: str    # rule description, or "fallback:page_operator" if no rule matched
    hypothesis_id: str


_DEFAULT_POLICY_PATH = (
    Path(__file__).parent.parent / "config" / "healing_policy.yaml"
)


class PolicyConfig:
    """
    Loads rules from YAML rather than hardcoding them so the policy table can be
    updated without a code change or redeploy. Rules are sorted by priority at
    load time; the engine iterates once per decision, never re-sorts.

    The YAML schema mirrors the PolicyRule fields: action and priority are required;
    fault_type, severity, stage_type, and description are optional.
    """

    def __init__(self, path: Path = _DEFAULT_POLICY_PATH) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        rules: List[PolicyRule] = []
        for entry in raw["rules"]:
            rules.append(
                PolicyRule(
                    action=HealingAction(entry["action"]),
                    priority=int(entry["priority"]),
                    fault_type=entry.get("fault_type"),
                    severity=entry.get("severity"),
                    stage_type=entry.get("stage_type"),
                    description=entry.get("description", ""),
                )
            )

        # Stable sort: rules with equal priority preserve YAML declaration order.
        self._rules: List[PolicyRule] = sorted(rules, key=lambda r: r.priority)

    @property
    def rules(self) -> List[PolicyRule]:
        return list(self._rules)

    def first_match(
        self,
        fault_type: Optional[str],
        severity: str,
        stage_type: Optional[str],
    ) -> PolicyRule:
        """
        Returns the highest-priority rule that matches the given inputs.
        Always returns a rule — the default YAML guarantees a catch-all PAGE_OPERATOR
        at priority=999. Raises RuntimeError only if the YAML was stripped of the fallback.
        """
        for rule in self._rules:
            if rule.matches(fault_type, severity, stage_type):
                return rule
        raise RuntimeError(
            "No matching rule found and no catch-all PAGE_OPERATOR rule present. "
            "Add a rule with no fault_type/severity/stage_type filter to healing_policy.yaml."
        )
