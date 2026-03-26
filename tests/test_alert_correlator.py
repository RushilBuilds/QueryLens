"""
Unit tests for AlertCorrelator, CorrelationPolicy, and CorrelatedAlert.

All tests use in-memory topologies — no containers or external services required.

Two topologies are used:
  (A) Linear 5-stage chain: src → t1 → t2 → t3 → snk
      All stages share src as a common ancestor; any pair is causally related.

  (B) Disjoint pair: independent_src → independent_snk
      These stages share no ancestors with topology A.

The key scenario from the roadmap (10 downstream anomalies from one root cause →
exactly one CorrelatedAlert) is tested in TestSingleFaultCollapse.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from causal.correlator import AlertCorrelator, CorrelatedAlert, CorrelationPolicy
from causal.dag import CausalDAG
from detection.anomaly import AnomalyEvent
from simulator.topology import PipelineStage, PipelineTopologyGraph


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _linear_topology() -> PipelineTopologyGraph:
    """src → t1 → t2 → t3 → snk (all causally related via src)."""
    return PipelineTopologyGraph([
        PipelineStage("src",  "source",    [],          0.0),
        PipelineStage("t1",   "transform", ["src"],     10.0),
        PipelineStage("t2",   "transform", ["t1"],      15.0),
        PipelineStage("t3",   "transform", ["t2"],       5.0),
        PipelineStage("snk",  "sink",      ["t3"],      20.0),
    ])


def _disjoint_topology() -> PipelineTopologyGraph:
    """isrc → isnk — no shared ancestors with the linear topology."""
    return PipelineTopologyGraph([
        PipelineStage("src",  "source",    [],          0.0),
        PipelineStage("t1",   "transform", ["src"],     10.0),
        PipelineStage("t2",   "transform", ["t1"],      15.0),
        PipelineStage("t3",   "transform", ["t2"],       5.0),
        PipelineStage("snk",  "sink",      ["t3"],      20.0),
        PipelineStage("isrc", "source",    [],           0.0),
        PipelineStage("isnk", "sink",      ["isrc"],    10.0),
    ])


def _anomaly(
    stage_id: str,
    offset_s: float = 0.0,
    metric: str = "latency_ms",
    fault_label: str = "latency_spike",
) -> AnomalyEvent:
    return AnomalyEvent(
        detector_type="cusum",
        stage_id=stage_id,
        metric=metric,
        signal="upper",
        detector_value=5.0,
        threshold=4.0,
        z_score=3.0,
        detected_at=T0 + timedelta(seconds=offset_s),
        fault_label=fault_label,
    )


@pytest.fixture(scope="module")
def linear_dag() -> CausalDAG:
    return CausalDAG(_linear_topology())


@pytest.fixture(scope="module")
def disjoint_dag() -> CausalDAG:
    return CausalDAG(_disjoint_topology())


# ---------------------------------------------------------------------------
# CorrelationPolicy — validation
# ---------------------------------------------------------------------------


class TestCorrelationPolicy:

    def test_default_values(self) -> None:
        policy = CorrelationPolicy()
        assert policy.window_duration_s == 60.0
        assert policy.min_co_occurrence == 2

    def test_rejects_zero_window(self) -> None:
        with pytest.raises(ValueError, match="window_duration_s"):
            CorrelationPolicy(window_duration_s=0.0)

    def test_rejects_negative_window(self) -> None:
        with pytest.raises(ValueError, match="window_duration_s"):
            CorrelationPolicy(window_duration_s=-10.0)

    def test_rejects_zero_min_co_occurrence(self) -> None:
        with pytest.raises(ValueError, match="min_co_occurrence"):
            CorrelationPolicy(min_co_occurrence=0)

    def test_min_co_occurrence_one_is_valid(self) -> None:
        policy = CorrelationPolicy(min_co_occurrence=1)
        assert policy.min_co_occurrence == 1


# ---------------------------------------------------------------------------
# AlertCorrelator — basic grouping
# ---------------------------------------------------------------------------


class TestAlertCorrelatorGrouping:

    def test_two_causally_related_events_within_window_form_one_group(
        self, linear_dag: CausalDAG
    ) -> None:
        correlator = AlertCorrelator(linear_dag, CorrelationPolicy(window_duration_s=60.0))
        correlator.add(_anomaly("src", offset_s=0.0))
        correlator.add(_anomaly("snk", offset_s=5.0))
        assert correlator.open_group_count == 1
        alerts = correlator.flush()
        assert len(alerts) == 1
        assert set(alerts[0].affected_stage_ids) == {"src", "snk"}

    def test_two_unrelated_stages_form_separate_groups(
        self, disjoint_dag: CausalDAG
    ) -> None:
        correlator = AlertCorrelator(disjoint_dag, CorrelationPolicy(window_duration_s=60.0))
        correlator.add(_anomaly("src",  offset_s=0.0))
        correlator.add(_anomaly("isrc", offset_s=5.0))
        # isrc shares no ancestors with src — two separate groups.
        assert correlator.open_group_count == 2
        alerts = correlator.flush()
        # Each group has only 1 event; min_co_occurrence=2 → both discarded.
        assert len(alerts) == 0

    def test_event_outside_window_closes_first_group(
        self, linear_dag: CausalDAG
    ) -> None:
        policy = CorrelationPolicy(window_duration_s=30.0, min_co_occurrence=2)
        correlator = AlertCorrelator(linear_dag, policy)
        correlator.add(_anomaly("src", offset_s=0.0))
        correlator.add(_anomaly("t1",  offset_s=10.0))
        # Third event arrives 40s after the first — exceeds the 30s window.
        closed = correlator.add(_anomaly("t2", offset_s=40.0))
        assert len(closed) == 1
        assert set(closed[0].affected_stage_ids) == {"src", "t1"}
        # The third event opens a new group.
        assert correlator.open_group_count == 1

    def test_flush_returns_qualifying_groups(self, linear_dag: CausalDAG) -> None:
        policy = CorrelationPolicy(window_duration_s=60.0, min_co_occurrence=2)
        correlator = AlertCorrelator(linear_dag, policy)
        correlator.add(_anomaly("t2", offset_s=0.0))
        correlator.add(_anomaly("t3", offset_s=5.0))
        alerts = correlator.flush()
        assert len(alerts) == 1

    def test_flush_discards_sub_threshold_groups(self, linear_dag: CausalDAG) -> None:
        policy = CorrelationPolicy(window_duration_s=60.0, min_co_occurrence=3)
        correlator = AlertCorrelator(linear_dag, policy)
        correlator.add(_anomaly("src", offset_s=0.0))
        correlator.add(_anomaly("t1",  offset_s=5.0))
        # Only 2 events; min_co_occurrence=3 → discarded.
        alerts = correlator.flush()
        assert len(alerts) == 0

    def test_flush_clears_all_open_groups(self, linear_dag: CausalDAG) -> None:
        correlator = AlertCorrelator(linear_dag, CorrelationPolicy())
        correlator.add(_anomaly("src", offset_s=0.0))
        correlator.flush()
        assert correlator.open_group_count == 0

    def test_same_stage_multiple_metrics_stays_in_one_group(
        self, linear_dag: CausalDAG
    ) -> None:
        correlator = AlertCorrelator(linear_dag, CorrelationPolicy(min_co_occurrence=1))
        correlator.add(_anomaly("src", offset_s=0.0, metric="latency_ms"))
        correlator.add(_anomaly("src", offset_s=1.0, metric="error_rate"))
        assert correlator.open_group_count == 1
        alerts = correlator.flush()
        assert len(alerts[0].evidence_events) == 2

    def test_min_co_occurrence_one_emits_single_event_alert(
        self, linear_dag: CausalDAG
    ) -> None:
        policy = CorrelationPolicy(window_duration_s=60.0, min_co_occurrence=1)
        correlator = AlertCorrelator(linear_dag, policy)
        correlator.add(_anomaly("src", offset_s=0.0))
        alerts = correlator.flush()
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# TestSingleFaultCollapse — the key M17 scenario
# ---------------------------------------------------------------------------


class TestSingleFaultCollapse:

    def test_ten_downstream_anomalies_produce_one_alert(
        self, linear_dag: CausalDAG
    ) -> None:
        """
        A single fault propagating through all 5 stages fires 2 anomalies per stage
        (CUSUM + EWMA, different metrics). All 10 events share src as a common ancestor
        and fall within the correlation window — exactly one CorrelatedAlert must be emitted.
        """
        policy = CorrelationPolicy(window_duration_s=120.0, min_co_occurrence=2)
        correlator = AlertCorrelator(linear_dag, policy)

        stages = ["src", "t1", "t2", "t3", "snk"]
        anomalies: List[AnomalyEvent] = []
        for i, stage in enumerate(stages):
            anomalies.append(_anomaly(stage, offset_s=float(i * 2),     metric="latency_ms"))
            anomalies.append(_anomaly(stage, offset_s=float(i * 2 + 1), metric="error_rate"))

        assert len(anomalies) == 10

        emitted_during: List[CorrelatedAlert] = []
        for a in anomalies:
            emitted_during.extend(correlator.add(a))

        # No groups should close mid-stream — all within 120s window.
        assert len(emitted_during) == 0

        alerts = correlator.flush()
        assert len(alerts) == 1, (
            f"Expected 1 CorrelatedAlert, got {len(alerts)}"
        )
        alert = alerts[0]
        assert len(alert.evidence_events) == 10
        assert set(alert.affected_stage_ids) == set(stages)

    def test_two_sequential_faults_produce_two_alerts(
        self, linear_dag: CausalDAG
    ) -> None:
        """
        Two separate fault bursts separated by more than the correlation window
        must produce two distinct CorrelatedAlerts.
        """
        policy = CorrelationPolicy(window_duration_s=30.0, min_co_occurrence=2)
        correlator = AlertCorrelator(linear_dag, policy)

        # First burst at t=0–10s.
        first_burst = [
            _anomaly("src", offset_s=0.0),
            _anomaly("t1",  offset_s=5.0),
            _anomaly("snk", offset_s=10.0),
        ]
        # Second burst at t=60–70s (well outside the 30s window).
        second_burst = [
            _anomaly("src", offset_s=60.0),
            _anomaly("t1",  offset_s=65.0),
            _anomaly("snk", offset_s=70.0),
        ]

        closed: List[CorrelatedAlert] = []
        for a in first_burst + second_burst:
            closed.extend(correlator.add(a))

        final = correlator.flush()
        all_alerts = closed + final
        assert len(all_alerts) == 2

    def test_unrelated_concurrent_faults_produce_separate_alerts(
        self, disjoint_dag: CausalDAG
    ) -> None:
        """
        Two faults occurring simultaneously on disconnected pipeline branches must
        produce separate alerts, not be merged into one.
        """
        policy = CorrelationPolicy(window_duration_s=60.0, min_co_occurrence=2)
        correlator = AlertCorrelator(disjoint_dag, policy)

        # Branch 1: src → t1 fault.
        correlator.add(_anomaly("src",  offset_s=0.0))
        correlator.add(_anomaly("t1",   offset_s=1.0))
        # Branch 2: isrc → isnk fault (unrelated).
        correlator.add(_anomaly("isrc", offset_s=2.0))
        correlator.add(_anomaly("isnk", offset_s=3.0))

        alerts = correlator.flush()
        assert len(alerts) == 2
        stage_sets = [set(a.affected_stage_ids) for a in alerts]
        assert {"src", "t1"} in stage_sets
        assert {"isrc", "isnk"} in stage_sets


# ---------------------------------------------------------------------------
# CorrelatedAlert — field correctness
# ---------------------------------------------------------------------------


class TestCorrelatedAlertFields:

    def test_affected_stage_ids_are_sorted(self, linear_dag: CausalDAG) -> None:
        policy = CorrelationPolicy(min_co_occurrence=1)
        correlator = AlertCorrelator(linear_dag, policy)
        correlator.add(_anomaly("snk", offset_s=0.0))
        correlator.add(_anomaly("src", offset_s=1.0))
        alerts = correlator.flush()
        ids = list(alerts[0].affected_stage_ids)
        assert ids == sorted(ids)

    def test_evidence_events_ordered_by_detected_at(self, linear_dag: CausalDAG) -> None:
        policy = CorrelationPolicy(min_co_occurrence=1)
        correlator = AlertCorrelator(linear_dag, policy)
        # Add events out of chronological order by stage.
        correlator.add(_anomaly("snk", offset_s=10.0))
        correlator.add(_anomaly("src", offset_s=0.0))
        correlator.add(_anomaly("t2",  offset_s=5.0))
        alerts = correlator.flush()
        times = [e.detected_at for e in alerts[0].evidence_events]
        assert times == sorted(times)

    def test_triggered_at_matches_first_event(self, linear_dag: CausalDAG) -> None:
        policy = CorrelationPolicy(min_co_occurrence=1)
        correlator = AlertCorrelator(linear_dag, policy)
        correlator.add(_anomaly("src", offset_s=0.0))
        correlator.add(_anomaly("t1",  offset_s=5.0))
        alerts = correlator.flush()
        assert alerts[0].triggered_at == T0

    def test_closed_at_matches_last_event(self, linear_dag: CausalDAG) -> None:
        policy = CorrelationPolicy(min_co_occurrence=1)
        correlator = AlertCorrelator(linear_dag, policy)
        correlator.add(_anomaly("src", offset_s=0.0))
        correlator.add(_anomaly("t1",  offset_s=5.0))
        alerts = correlator.flush()
        assert alerts[0].closed_at == T0 + timedelta(seconds=5.0)

    def test_alert_id_is_unique_per_alert(self, linear_dag: CausalDAG) -> None:
        policy = CorrelationPolicy(window_duration_s=10.0, min_co_occurrence=2)
        correlator = AlertCorrelator(linear_dag, policy)

        correlator.add(_anomaly("src", offset_s=0.0))
        correlator.add(_anomaly("t1",  offset_s=5.0))
        first_closed = correlator.add(_anomaly("t2", offset_s=20.0))

        correlator.add(_anomaly("t3",  offset_s=20.0))
        second_alerts = correlator.flush()

        all_ids = [a.alert_id for a in first_closed + second_alerts]
        assert len(all_ids) == len(set(all_ids)), "alert_id values must be unique"
