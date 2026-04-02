"""
PipelineHealthView — grid of stage cards showing current status at a glance.

Color-coding uses deviation from a static threshold rather than the seasonal
baseline (which requires a DB query per stage). Thresholds are intentionally
conservative: amber at p99 > 100ms, red at p99 > 500ms. Operators can adjust
via environment variables once seasonal baselines are wired into the API.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

# ---------------------------------------------------------------------------
# Severity thresholds — p99 latency in milliseconds
# ---------------------------------------------------------------------------

_AMBER_THRESHOLD_MS = 100.0
_RED_THRESHOLD_MS = 500.0


def _severity_color(p99_ms: float, breaker_state: str) -> str:
    """
    Circuit-open overrides latency-based coloring because an open breaker means
    the stage is already isolated — latency may look fine simply because no
    traffic is flowing through it.
    """
    if breaker_state == "open":
        return "red"
    if p99_ms >= _RED_THRESHOLD_MS:
        return "red"
    if p99_ms >= _AMBER_THRESHOLD_MS:
        return "amber"
    return "green"


def _color_to_emoji(color: str) -> str:
    """Maps severity to a unicode indicator for Streamlit markdown rendering."""
    return {"green": "\U0001f7e2", "amber": "\U0001f7e1", "red": "\U0001f534"}.get(color, "\u26aa")


def _breaker_label(state: str) -> str:
    return {"closed": "Closed", "open": "OPEN", "half_open": "Half-Open"}.get(state, state)


def render_pipeline_health(stages: List[Dict[str, Any]]) -> None:
    """
    Renders a responsive grid of stage cards. Three columns on wide screens
    collapse naturally via Streamlit's column layout. Each card shows stage
    name, severity indicator, p99 latency, event count, and breaker state.
    """
    if not stages:
        st.info("No stages found. Run the simulator to generate pipeline data.")
        return

    cols_per_row = 3
    for row_start in range(0, len(stages), cols_per_row):
        row_stages = stages[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, stage in zip(cols, row_stages):
            p99 = stage.get("p99_latency_ms", 0.0)
            breaker = stage.get("circuit_breaker", {})
            breaker_state = breaker.get("state", "unknown")
            trip_count = breaker.get("trip_count", 0)
            event_count = stage.get("event_count", 0)

            color = _severity_color(p99, breaker_state)
            emoji = _color_to_emoji(color)

            with col:
                st.markdown(
                    f"### {emoji} {stage['stage_id']}\n"
                    f"- **p99 latency**: {p99:.1f} ms\n"
                    f"- **events**: {event_count:,}\n"
                    f"- **breaker**: {_breaker_label(breaker_state)}"
                    + (f" ({trip_count} trips)" if trip_count > 0 else "")
                )
