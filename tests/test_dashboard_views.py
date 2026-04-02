"""
Smoke tests for AnomalyTimelineView and CausalGraphView (M28).

Verifies that view functions:
  - Accept valid data without raising exceptions
  - Handle empty input gracefully
  - Produce correct Plotly figure structures
  - Highlight root cause stage in the causal graph
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# AnomalyTimelineView
# ---------------------------------------------------------------------------


class TestAnomalyTimeline:

    @patch("dashboard.views.timeline.st")
    def test_renders_without_error(self, mock_st: MagicMock) -> None:
        from dashboard.views.timeline import render_anomaly_timeline

        anomalies = [
            {
                "stage_id": "ingest",
                "detected_at": "2024-06-01T12:00:00",
                "detector_type": "cusum",
                "metric": "latency_ms",
                "signal": "upper",
                "detector_value": 5.2,
                "threshold": 4.0,
            },
            {
                "stage_id": "transform",
                "detected_at": "2024-06-01T12:01:00",
                "detector_type": "ewma",
                "metric": "latency_ms",
                "signal": "upper",
                "detector_value": 3.8,
                "threshold": 3.0,
            },
        ]
        render_anomaly_timeline(anomalies)
        mock_st.plotly_chart.assert_called_once()

    @patch("dashboard.views.timeline.st")
    def test_empty_anomalies_shows_info(self, mock_st: MagicMock) -> None:
        from dashboard.views.timeline import render_anomaly_timeline

        render_anomaly_timeline([])
        mock_st.info.assert_called_once()

    @patch("dashboard.views.timeline.st")
    def test_groups_by_detector_type(self, mock_st: MagicMock) -> None:
        from dashboard.views.timeline import render_anomaly_timeline

        anomalies = [
            {"stage_id": "s", "detected_at": "2024-01-01T00:00:00",
             "detector_type": "cusum", "metric": "m", "signal": "upper",
             "detector_value": 1.0, "threshold": 1.0},
            {"stage_id": "s", "detected_at": "2024-01-01T00:01:00",
             "detector_type": "ewma", "metric": "m", "signal": "upper",
             "detector_value": 1.0, "threshold": 1.0},
        ]
        render_anomaly_timeline(anomalies)

        fig = mock_st.plotly_chart.call_args[0][0]
        trace_names = [t.name for t in fig.data]
        assert "CUSUM" in trace_names
        assert "EWMA" in trace_names


# ---------------------------------------------------------------------------
# CausalGraphView
# ---------------------------------------------------------------------------


class TestCausalGraph:

    def _sample_stages(self):
        return [
            {"stage_id": "source", "p99_latency_ms": 20.0, "event_count": 500,
             "circuit_breaker": {"state": "closed", "trip_count": 0}},
            {"stage_id": "transform", "p99_latency_ms": 150.0, "event_count": 450,
             "circuit_breaker": {"state": "closed", "trip_count": 0}},
            {"stage_id": "sink", "p99_latency_ms": 30.0, "event_count": 400,
             "circuit_breaker": {"state": "open", "trip_count": 1}},
        ]

    @patch("dashboard.views.causal_graph.st")
    def test_renders_without_error(self, mock_st: MagicMock) -> None:
        from dashboard.views.causal_graph import render_causal_graph

        render_causal_graph(self._sample_stages(), [])
        mock_st.plotly_chart.assert_called_once()

    @patch("dashboard.views.causal_graph.st")
    def test_empty_stages_shows_info(self, mock_st: MagicMock) -> None:
        from dashboard.views.causal_graph import render_causal_graph

        render_causal_graph([], [])
        mock_st.info.assert_called_once()

    @patch("dashboard.views.causal_graph.st")
    def test_root_cause_highlighted_in_red(self, mock_st: MagicMock) -> None:
        from dashboard.views.causal_graph import render_causal_graph

        localizations = [{"root_cause_stage_id": "source", "hypothesis_id": "h1"}]
        render_causal_graph(self._sample_stages(), localizations)

        fig = mock_st.plotly_chart.call_args[0][0]
        # Find the node trace (the one with markers)
        node_trace = [t for t in fig.data if hasattr(t, "marker") and t.marker.size]
        assert len(node_trace) == 1
        colors = node_trace[0].marker.color
        # source is index 0, should be red (#e74c3c)
        assert colors[0] == "#e74c3c"

    @patch("dashboard.views.causal_graph.st")
    def test_open_breaker_is_red(self, mock_st: MagicMock) -> None:
        from dashboard.views.causal_graph import render_causal_graph

        render_causal_graph(self._sample_stages(), [])

        fig = mock_st.plotly_chart.call_args[0][0]
        node_trace = [t for t in fig.data if hasattr(t, "marker") and t.marker.size]
        colors = node_trace[0].marker.color
        # sink (index 2) has open breaker — should be red
        assert colors[2] == "#e74c3c"

    @patch("dashboard.views.causal_graph.st")
    def test_elevated_latency_is_amber(self, mock_st: MagicMock) -> None:
        from dashboard.views.causal_graph import render_causal_graph

        render_causal_graph(self._sample_stages(), [])

        fig = mock_st.plotly_chart.call_args[0][0]
        node_trace = [t for t in fig.data if hasattr(t, "marker") and t.marker.size]
        colors = node_trace[0].marker.color
        # transform (index 1) has p99=150ms — amber
        assert colors[1] == "#f39c12"


# ---------------------------------------------------------------------------
# Layered positions
# ---------------------------------------------------------------------------


class TestLayeredPositions:

    def test_single_node(self) -> None:
        from dashboard.views.causal_graph import _layered_positions

        pos = _layered_positions(["a"])
        assert pos["a"] == (0.0, 0.5)

    def test_two_nodes_span_zero_to_one(self) -> None:
        from dashboard.views.causal_graph import _layered_positions

        pos = _layered_positions(["a", "b"])
        assert pos["a"][0] == 0.0
        assert pos["b"][0] == 1.0

    def test_preserves_order(self) -> None:
        from dashboard.views.causal_graph import _layered_positions

        pos = _layered_positions(["c", "a", "b"])
        assert pos["c"][0] < pos["a"][0] < pos["b"][0]
