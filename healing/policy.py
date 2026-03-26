from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
