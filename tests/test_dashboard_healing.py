"""
Smoke tests for HealingActivityView, ManualOverridePanel, and HealingAuditTrail (M29).

All three views are tested with mocked Streamlit and API client.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dashboard.views.healing import (
    _outcome_badge,
    render_audit_trail,
    render_healing_activity,
    render_override_panel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_actions():
    return [
        {
            "id": 1,
            "hypothesis_id": "hyp-001",
            "stage_id": "ingest",
            "action": "circuit_break",
            "fault_type": "latency_spike",
            "severity": "high",
            "outcome": "pending",
            "triggered_at": "2024-06-01T12:00:00",
            "resolved_at": None,
            "notes": None,
        },
        {
            "id": 2,
            "hypothesis_id": "hyp-002",
            "stage_id": "transform",
            "action": "replay_range",
            "fault_type": "dropped_connection",
            "severity": "medium",
            "outcome": "success",
            "triggered_at": "2024-06-01T12:05:00",
            "resolved_at": "2024-06-01T12:10:00",
            "notes": "replayed 200 records",
        },
    ]


# ---------------------------------------------------------------------------
# Outcome badge
# ---------------------------------------------------------------------------


class TestOutcomeBadge:

    def test_pending(self) -> None:
        assert "Pending" in _outcome_badge("pending")

    def test_success(self) -> None:
        assert "Success" in _outcome_badge("success")

    def test_failed(self) -> None:
        assert "Failed" in _outcome_badge("failed")

    def test_cancelled(self) -> None:
        assert "Cancelled" in _outcome_badge("cancelled")

    def test_unknown_passthrough(self) -> None:
        assert _outcome_badge("weird") == "weird"


# ---------------------------------------------------------------------------
# HealingActivityView
# ---------------------------------------------------------------------------


class TestHealingActivityView:

    @patch("dashboard.views.healing.st")
    def test_renders_table(self, mock_st: MagicMock) -> None:
        render_healing_activity(_sample_actions())
        mock_st.dataframe.assert_called_once()

    @patch("dashboard.views.healing.st")
    def test_empty_shows_info(self, mock_st: MagicMock) -> None:
        render_healing_activity([])
        mock_st.info.assert_called_once()

    @patch("dashboard.views.healing.st")
    def test_row_count_matches_actions(self, mock_st: MagicMock) -> None:
        render_healing_activity(_sample_actions())
        rows = mock_st.dataframe.call_args[0][0]
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# ManualOverridePanel
# ---------------------------------------------------------------------------


class TestOverridePanel:

    @patch("dashboard.views.healing.st")
    def test_no_pending_shows_caption(self, mock_st: MagicMock) -> None:
        actions = [a for a in _sample_actions() if a["outcome"] != "pending"]
        mock_api = MagicMock()
        render_override_panel(actions, mock_api)
        mock_st.caption.assert_called_once()

    @patch("dashboard.views.healing.st")
    def test_pending_shows_selectbox(self, mock_st: MagicMock) -> None:
        mock_st.selectbox.return_value = "hyp-001"
        mock_st.text_input.return_value = ""
        mock_st.button.return_value = False
        mock_api = MagicMock()

        render_override_panel(_sample_actions(), mock_api)
        mock_st.selectbox.assert_called_once()


# ---------------------------------------------------------------------------
# HealingAuditTrail
# ---------------------------------------------------------------------------


class TestAuditTrail:

    @patch("dashboard.views.healing.st")
    def test_renders_expanders(self, mock_st: MagicMock) -> None:
        mock_api = MagicMock()
        mock_api.localization_detail.return_value = {
            "root_cause_stage_id": "ingest",
            "posterior_probability": 0.85,
            "evidence_count": 3,
            "ranked_candidates": [
                {"stage_id": "ingest", "posterior_probability": 0.85},
            ],
        }
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        render_audit_trail(_sample_actions(), mock_api)
        assert mock_st.expander.call_count == 2

    @patch("dashboard.views.healing.st")
    def test_empty_actions_no_expanders(self, mock_st: MagicMock) -> None:
        mock_api = MagicMock()
        render_audit_trail([], mock_api)
        mock_st.expander.assert_not_called()
