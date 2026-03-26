from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from causal.dag import CausalDAG
from detection.anomaly import AnomalyEvent


@dataclass(frozen=True)
class FaultHypothesis:
    """
    Frozen dataclass keeps evidence immutable once emitted — a mutable list would allow
    a race if a late-arriving anomaly is added while scoring is in progress.

    hypothesis_id is a UUID string rather than an integer sequence so that hypotheses
    from parallel collector instances do not collide without a shared counter.
    """

    hypothesis_id: str
    triggered_at: datetime          # detected_at of the first anomaly in the window
    closed_at: datetime             # detected_at of the last anomaly in the window
    evidence_events: Tuple[AnomalyEvent, ...]   # frozen tuple, not list


@dataclass(frozen=True)
class LocalizationResult:
    """
    Includes evidence_events so the HealingAuditLog (M22) can record exactly which
    anomalies drove the decision without re-querying the AnomalyEventBus. Denormalising
    the evidence trades storage for query simplicity.

    ranked_candidates is ordered by posterior_probability descending — index 0 is the
    most probable root cause. Use ranked_candidates[0] for automated remediation and
    ranked_candidates[:2] for operator alerts.
    """

    hypothesis_id: str
    triggered_at: datetime
    evidence_events: Tuple[AnomalyEvent, ...]
    ranked_candidates: Tuple[Tuple[str, float], ...]  # (stage_id, posterior_probability)

    @property
    def top_candidate(self) -> Optional[Tuple[str, float]]:
        return self.ranked_candidates[0] if self.ranked_candidates else None

    def candidate_in_top_n(self, stage_id: str, n: int = 2) -> bool:
        """Returns True if stage_id appears in the top-n ranked candidates."""
        return any(
            stage_id == cid
            for cid, _ in self.ranked_candidates[:n]
        )


class AnomalyWindowCollector:
    """
    Gap-based correlation window rather than a fixed sliding window: a fixed window
    would close prematurely during a fault storm, grouping the first burst separately
    from the second. Gap-based grouping keeps all anomalies from a single fault event
    in one hypothesis regardless of intra-burst arrival rate.

    Trade-off: emission is delayed until the first anomaly that exceeds the gap threshold,
    adding one gap_duration_s of latency before the engine sees the hypothesis. At
    gap_duration_s=30.0 this is acceptable for post-fault analysis; for real-time healing,
    use a smaller gap (5–10s) at the cost of splitting some multi-stage fault bursts.
    """

    def __init__(
        self,
        gap_duration_s: float = 30.0,
        min_events: int = 1,
    ) -> None:
        if gap_duration_s <= 0.0:
            raise ValueError(f"gap_duration_s must be > 0, got {gap_duration_s}")
        if min_events < 1:
            raise ValueError(f"min_events must be >= 1, got {min_events}")

        self._gap = timedelta(seconds=gap_duration_s)
        self._min_events = min_events
        self._buffer: List[AnomalyEvent] = []
        self._window_start: Optional[datetime] = None
        self._last_event_time: Optional[datetime] = None

    def add(self, anomaly: AnomalyEvent) -> Optional[FaultHypothesis]:
        """
        Compares detected_at against the last event time (not the window start) because
        the gap criterion is about inactivity between events, not total window duration.
        A window with events at T=0, T=5, T=35 closes between T=5 and T=35, not at T=30.

        Returns a completed FaultHypothesis if the incoming anomaly starts a new window
        (gap exceeded), otherwise returns None.
        """
        emitted: Optional[FaultHypothesis] = None

        if self._last_event_time is not None:
            gap = anomaly.detected_at - self._last_event_time
            if gap > self._gap:
                emitted = self._emit()

        if not self._buffer:
            self._window_start = anomaly.detected_at

        self._buffer.append(anomaly)
        self._last_event_time = anomaly.detected_at

        return emitted

    def flush(self) -> Optional[FaultHypothesis]:
        """
        Force-emit the current window. Call at shutdown or end-of-stream to
        ensure the final window is not silently dropped.

        Returns None if the buffer is empty or has fewer events than min_events.
        """
        return self._emit()

    def _emit(self) -> Optional[FaultHypothesis]:
        if len(self._buffer) < self._min_events:
            self._buffer.clear()
            self._window_start = None
            self._last_event_time = None
            return None

        hypothesis = FaultHypothesis(
            hypothesis_id=str(uuid.uuid4()),
            triggered_at=self._window_start,  # type: ignore[arg-type]
            closed_at=self._last_event_time,  # type: ignore[arg-type]
            evidence_events=tuple(self._buffer),
        )
        self._buffer.clear()
        self._window_start = None
        self._last_event_time = None
        return hypothesis

    @property
    def pending_count(self) -> int:
        """Number of anomalies buffered but not yet emitted."""
        return len(self._buffer)


class FaultLocalizationEngine:
    """
    Bayesian posterior update rather than a simple vote count: vote counting treats all
    evidence equally and cannot distinguish a stage that appeared at the expected
    propagation delay from one with timing inconsistent with causality. The Bayesian
    approach weights each evidence event by how well its timing matches the propagation
    delay model for each candidate.

    Scoring formula for candidate C given evidence event E at symptomatic stage S:
        contribution(C, E) = [C is ancestor of S]
                           × (1 / (cumulative_delay(C, S) + ε))
                           × timing_factor(C, S, E)

    timing_factor: if C also appears in the evidence, checks whether C's anomaly
    precedes S's anomaly by approximately cumulative_delay(C, S). Correct ordering
    doubles the contribution; wrong ordering halves it. No candidate anomaly → neutral 1.0.

    After summing contributions, scores are normalised to sum to 1.0 (posterior probability).

    ε=1.0ms in the delay denominator prevents infinite scores for zero-delay (co-located)
    edges. A delay of 0ms with ε=1.0 gives contribution=1.0, correctly prioritising
    co-located root causes.
    """

    _DELAY_EPSILON_MS = 1.0
    _TIMING_MATCH_BONUS = 2.0      # multiplier when anomaly ordering matches delay model
    _TIMING_MISMATCH_PENALTY = 0.5  # multiplier when ordering contradicts delay model
    # ±50% of expected delay as timing match tolerance. Tighter rejects valid root
    # causes during high-load; looser reduces the propagation-timing signal.
    _TIMING_TOLERANCE_FRACTION = 0.5

    def __init__(self, dag: CausalDAG) -> None:
        self._dag = dag

    def localize(self, hypothesis: FaultHypothesis) -> Optional[LocalizationResult]:
        """
        Returns a LocalizationResult ranking all candidate root-cause stages by
        posterior probability, or None if no candidates exist (e.g. only source
        stages are symptomatic and have no ancestors).
        """
        candidates = self._gather_candidates(hypothesis)
        if not candidates:
            return None

        scores = self._score_candidates(candidates, hypothesis)
        if not scores:
            return None

        total = sum(scores.values())
        if total == 0.0:
            posterior = {c: 1.0 / len(scores) for c in scores}
        else:
            posterior = {c: s / total for c, s in scores.items()}

        ranked = sorted(
            posterior.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

        return LocalizationResult(
            hypothesis_id=hypothesis.hypothesis_id,
            triggered_at=hypothesis.triggered_at,
            evidence_events=hypothesis.evidence_events,
            ranked_candidates=tuple(ranked),
        )

    def _gather_candidates(self, hypothesis: FaultHypothesis) -> List[str]:
        """
        Collects all stages that are ancestors of at least one symptomatic stage.
        Symptomatic stages themselves are included as candidates — a stage that fires
        an anomaly could be both root cause and symptom if the fault is internal to it.
        """
        symptomatic_ids = {e.stage_id for e in hypothesis.evidence_events}
        candidate_ids: set = set()

        for stage_id in symptomatic_ids:
            try:
                ancestors = self._dag.causal_ancestors(stage_id)
            except KeyError:
                continue
            for a in ancestors:
                candidate_ids.add(a.stage.stage_id)
            # Include the symptomatic stage itself as a candidate.
            candidate_ids.add(stage_id)

        return list(candidate_ids)

    def _score_candidates(
        self,
        candidates: List[str],
        hypothesis: FaultHypothesis,
    ) -> Dict[str, float]:
        """
        Scores sum over all (candidate, evidence_event) pairs rather than unique stages
        because multiple anomalies for the same stage (CUSUM + EWMA both firing on
        latency_ms) carry independent evidence. Collapsing to unique stages would discard
        the stronger signal that two detectors agreed on the same symptom.
        """
        # Build lookup: stage_id → list of detected_at times for that stage in evidence.
        evidence_times: Dict[str, List[datetime]] = {}
        for e in hypothesis.evidence_events:
            evidence_times.setdefault(e.stage_id, []).append(e.detected_at)

        scores: Dict[str, float] = {c: 0.0 for c in candidates}

        for candidate_id in candidates:
            for evidence_event in hypothesis.evidence_events:
                symptomatic_id = evidence_event.stage_id
                if symptomatic_id == candidate_id:
                    # Self-contribution: candidate is also symptomatic. Assign a
                    # baseline score to ensure it's ranked above non-symptomatic
                    # stages with zero coverage.
                    scores[candidate_id] += 1.0 / self._DELAY_EPSILON_MS
                    continue

                try:
                    ancestors = self._dag.causal_ancestors(symptomatic_id)
                except KeyError:
                    continue

                ancestor_entry = next(
                    (a for a in ancestors if a.stage.stage_id == candidate_id),
                    None,
                )
                if ancestor_entry is None:
                    continue

                delay_ms = ancestor_entry.cumulative_delay_ms
                base = 1.0 / (delay_ms + self._DELAY_EPSILON_MS)

                # Timing adjustment using anomaly ordering.
                timing = self._timing_factor(
                    candidate_id,
                    symptomatic_id,
                    delay_ms,
                    evidence_times,
                    evidence_event.detected_at,
                )
                scores[candidate_id] += base * timing

        return scores

    def _timing_factor(
        self,
        candidate_id: str,
        symptomatic_id: str,
        expected_delay_ms: float,
        evidence_times: Dict[str, List[datetime]],
        symptom_time: datetime,
    ) -> float:
        """
        If the candidate itself appeared in the evidence, compare its anomaly
        time against the symptom time to assess whether the ordering is consistent
        with fault propagation physics. Returns the timing multiplier.
        """
        candidate_times = evidence_times.get(candidate_id)
        if not candidate_times:
            return 1.0  # no candidate anomaly time — neutral

        # Use the earliest anomaly time for the candidate (first sign of fault).
        earliest_candidate = min(candidate_times)
        observed_lag_ms = (
            symptom_time - earliest_candidate
        ).total_seconds() * 1000.0

        if observed_lag_ms < 0.0:
            # Symptom appeared BEFORE the candidate anomaly — contradicts causality.
            return self._TIMING_MISMATCH_PENALTY

        tolerance_ms = max(expected_delay_ms * self._TIMING_TOLERANCE_FRACTION, 100.0)
        if abs(observed_lag_ms - expected_delay_ms) <= tolerance_ms:
            return self._TIMING_MATCH_BONUS

        return 1.0  # ordering correct but outside tolerance — neutral
