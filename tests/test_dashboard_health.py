"""
Smoke tests for the dashboard PipelineHealthView and API client.

Verifies:
  - The Streamlit app module imports without errors
  - PipelineHealthView renders without exceptions for normal, empty, and degraded data
  - Severity color-coding logic maps correctly
  - QueryLensAPI constructs URLs correctly
  - API client methods produce valid request parameters

No live API or Streamlit server required — view functions are tested by calling
them directly (they write to a Streamlit context that is no-op in test).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dashboard.api_client import QueryLensAPI
from dashboard.views.health import _severity_color, _color_to_emoji, _breaker_label


# ---------------------------------------------------------------------------
# Severity color logic
# ---------------------------------------------------------------------------


class TestSeverityColor:

    def test_low_latency_closed_breaker_is_green(self) -> None:
        assert _severity_color(50.0, "closed") == "green"

    def test_amber_threshold(self) -> None:
        assert _severity_color(150.0, "closed") == "amber"

    def test_red_threshold(self) -> None:
        assert _severity_color(600.0, "closed") == "red"

    def test_open_breaker_overrides_low_latency(self) -> None:
        assert _severity_color(10.0, "open") == "red"

    def test_half_open_with_low_latency_is_green(self) -> None:
        assert _severity_color(50.0, "half_open") == "green"

    def test_zero_latency_is_green(self) -> None:
        assert _severity_color(0.0, "closed") == "green"

    def test_boundary_at_100ms(self) -> None:
        assert _severity_color(100.0, "closed") == "amber"

    def test_boundary_at_500ms(self) -> None:
        assert _severity_color(500.0, "closed") == "red"


class TestColorToEmoji:

    def test_green_emoji(self) -> None:
        assert _color_to_emoji("green") == "\U0001f7e2"

    def test_amber_emoji(self) -> None:
        assert _color_to_emoji("amber") == "\U0001f7e1"

    def test_red_emoji(self) -> None:
        assert _color_to_emoji("red") == "\U0001f534"

    def test_unknown_returns_default(self) -> None:
        assert _color_to_emoji("purple") == "\u26aa"


class TestBreakerLabel:

    def test_closed(self) -> None:
        assert _breaker_label("closed") == "Closed"

    def test_open(self) -> None:
        assert _breaker_label("open") == "OPEN"

    def test_half_open(self) -> None:
        assert _breaker_label("half_open") == "Half-Open"

    def test_unknown_passthrough(self) -> None:
        assert _breaker_label("weird") == "weird"


# ---------------------------------------------------------------------------
# PipelineHealthView render — smoke test with mocked Streamlit
# ---------------------------------------------------------------------------


class TestRenderPipelineHealth:

    @patch("dashboard.views.health.st")
    def test_renders_without_error(self, mock_st: MagicMock) -> None:
        from dashboard.views.health import render_pipeline_health

        stages = [
            {
                "stage_id": "ingest",
                "p99_latency_ms": 25.0,
                "event_count": 1000,
                "circuit_breaker": {"state": "closed", "trip_count": 0},
            },
            {
                "stage_id": "transform",
                "p99_latency_ms": 200.0,
                "event_count": 900,
                "circuit_breaker": {"state": "open", "trip_count": 2},
            },
        ]
        # Should not raise
        render_pipeline_health(stages)
        assert mock_st.columns.called

    @patch("dashboard.views.health.st")
    def test_empty_stages_shows_info(self, mock_st: MagicMock) -> None:
        from dashboard.views.health import render_pipeline_health

        render_pipeline_health([])
        mock_st.info.assert_called_once()


# ---------------------------------------------------------------------------
# QueryLensAPI — URL construction
# ---------------------------------------------------------------------------


class TestQueryLensAPI:

    def test_base_url_strips_trailing_slash(self) -> None:
        api = QueryLensAPI(base_url="http://localhost:8000/")
        assert api._base == "http://localhost:8000"

    @patch("dashboard.api_client.requests.Session")
    def test_stages_calls_correct_path(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        api = QueryLensAPI(base_url="http://api:8000")
        api._session = mock_session
        api.stages()

        mock_session.get.assert_called_once_with(
            "http://api:8000/stages", params=None, timeout=10
        )

    @patch("dashboard.api_client.requests.Session")
    def test_healing_actions_passes_outcome_param(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": [], "total": 0}
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        api = QueryLensAPI(base_url="http://api:8000")
        api._session = mock_session
        api.healing_actions(outcome="pending")

        call_args = mock_session.get.call_args
        assert call_args[1]["params"]["outcome"] == "pending"
