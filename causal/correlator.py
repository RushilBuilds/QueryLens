from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

from causal.dag import CausalDAG
from detection.anomaly import AnomalyEvent


@dataclass(frozen=True)
class CorrelationPolicy:
    """
    window_duration_s sets the dedup horizon: anomalies within this span of the first
    event in a group are candidates for correlation. min_co_occurrence filters noise —
    a lone anomaly on an isolated stage does not generate an alert on its own.
    """

    window_duration_s: float = 60.0
    min_co_occurrence: int = 2

    def __post_init__(self) -> None:
        if self.window_duration_s <= 0.0:
            raise ValueError(
                f"window_duration_s must be > 0, got {self.window_duration_s}"
            )
        if self.min_co_occurrence < 1:
            raise ValueError(
                f"min_co_occurrence must be >= 1, got {self.min_co_occurrence}"
            )


@dataclass(frozen=True)
class CorrelatedAlert:
    """
    Collapsed representation of a group of causally related AnomalyEvents.
    Carrying evidence_events directly avoids a join back to the AnomalyEventBus
    when the HealingPolicyEngine selects a remediation action.
    """

    alert_id: str
    triggered_at: datetime                     # detected_at of the first constituent event
    closed_at: datetime                        # detected_at of the last constituent event
    affected_stage_ids: Tuple[str, ...]        # unique stages involved, sorted
    evidence_events: Tuple[AnomalyEvent, ...]  # constituent events ordered by detected_at


class _CorrelationGroup:
    """
    Mutable accumulator for one open correlation window. Not exposed publicly —
    AlertCorrelator emits CorrelatedAlert instances when groups close.
    """

    def __init__(self, first_event: AnomalyEvent, ancestor_ids: Set[str]) -> None:
        self.triggered_at: datetime = first_event.detected_at
        self.events: List[AnomalyEvent] = [first_event]
        # stage_ids: stages already in this group.
        # ancestor_ids: union of all ancestors for every stage in this group, plus
        # the stages themselves. Used for O(1) causal-relatedness checks on incoming events.
        self.stage_ids: Set[str] = {first_event.stage_id}
        self.ancestor_ids: Set[str] = ancestor_ids | {first_event.stage_id}

    def add(self, event: AnomalyEvent, ancestor_ids: Set[str]) -> None:
        self.events.append(event)
        self.stage_ids.add(event.stage_id)
        self.ancestor_ids |= ancestor_ids | {event.stage_id}

    def to_alert(self) -> CorrelatedAlert:
        sorted_events = sorted(self.events, key=lambda e: e.detected_at)
        return CorrelatedAlert(
            alert_id=str(uuid.uuid4()),
            triggered_at=self.triggered_at,
            closed_at=sorted_events[-1].detected_at,
            affected_stage_ids=tuple(sorted(self.stage_ids)),
            evidence_events=tuple(sorted_events),
        )


class AlertCorrelator:
    """
    Sliding deduplication window with causal ancestor grouping. Without this layer,
    a single fault propagating through a 10-stage pipeline generates 10 independent
    alerts — the healing layer (and operators) would receive 10 pages for one root cause.

    Grouping logic: two anomalies are correlated if (a) both fall within window_duration_s
    of the first event in an open group, AND (b) their stages share a causal ancestor,
    or one stage is an ancestor of the other.

    Groups close when a new event arrives after the window has expired. Groups below
    min_co_occurrence are discarded as noise rather than emitted as alerts.
    """

    def __init__(self, dag: CausalDAG, policy: CorrelationPolicy) -> None:
        self._dag = dag
        self._policy = policy
        self._window = timedelta(seconds=policy.window_duration_s)
        self._open_groups: List[_CorrelationGroup] = []

    def add(self, anomaly: AnomalyEvent) -> List[CorrelatedAlert]:
        """
        Processes one AnomalyEvent. Returns any CorrelatedAlerts emitted by closing
        windows that have expired relative to this event's timestamp.
        """
        emitted: List[CorrelatedAlert] = []

        # Close expired groups before evaluating the new event.
        still_open: List[_CorrelationGroup] = []
        for group in self._open_groups:
            if anomaly.detected_at - group.triggered_at > self._window:
                if len(group.events) >= self._policy.min_co_occurrence:
                    emitted.append(group.to_alert())
            else:
                still_open.append(group)
        self._open_groups = still_open

        ancestor_ids = self._ancestor_ids(anomaly.stage_id)

        # Add to the first open group that shares causal ancestry.
        merged = False
        for group in self._open_groups:
            if self._causally_related(ancestor_ids, anomaly.stage_id, group):
                group.add(anomaly, ancestor_ids)
                merged = True
                break

        if not merged:
            self._open_groups.append(_CorrelationGroup(anomaly, ancestor_ids))

        return emitted

    def flush(self) -> List[CorrelatedAlert]:
        """
        Force-emit all open groups. Call at end-of-stream to avoid silently dropping
        the final batch of correlated events.
        """
        emitted = [
            group.to_alert()
            for group in self._open_groups
            if len(group.events) >= self._policy.min_co_occurrence
        ]
        self._open_groups.clear()
        return emitted

    @property
    def open_group_count(self) -> int:
        """Number of groups currently accumulating events."""
        return len(self._open_groups)

    def _ancestor_ids(self, stage_id: str) -> Set[str]:
        try:
            return {a.stage.stage_id for a in self._dag.causal_ancestors(stage_id)}
        except KeyError:
            return set()

    def _causally_related(
        self,
        incoming_ancestors: Set[str],
        incoming_stage_id: str,
        group: _CorrelationGroup,
    ) -> bool:
        """
        Returns True if the incoming stage is causally related to any stage already in
        the group. Three cases cover all relationships: incoming is downstream of a group
        member, a group member is downstream of incoming, or they share a common ancestor.
        """
        if incoming_stage_id in group.ancestor_ids:
            return True
        if incoming_ancestors & group.stage_ids:
            return True
        if incoming_ancestors & group.ancestor_ids:
            return True
        return False
