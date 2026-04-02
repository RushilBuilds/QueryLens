"""
QueryLens Observatory Dashboard — pipeline health overview.

Streamlit entry point: `streamlit run dashboard/app.py`

Auto-refreshes every 10 seconds via streamlit-autorefresh so operators see
live status without manual reloads. Falls back gracefully when the API is
unreachable — the dashboard shows an error banner rather than crashing,
because the API going down is exactly when operators need the dashboard most.
"""
from __future__ import annotations

import os

import streamlit as st

from dashboard.api_client import QueryLensAPI
from dashboard.views.health import render_pipeline_health

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QueryLens Observatory",
    page_icon="\u2699",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10_000, limit=None, key="dashboard_autorefresh")
except ImportError:
    pass  # graceful fallback — manual refresh still works

# ---------------------------------------------------------------------------
# Sidebar — API connection
# ---------------------------------------------------------------------------

api_url = os.environ.get("QUERYLENS_API_URL", "http://localhost:8000")
api = QueryLensAPI(base_url=api_url)

st.sidebar.title("QueryLens")
st.sidebar.caption(f"API: {api_url}")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Pipeline Health Overview")

try:
    health = api.health()
    if health["status"] == "ok":
        st.success("System healthy")
    else:
        st.warning(f"System degraded — db: {health['db']}, redpanda: {health['redpanda']}")
except Exception as exc:
    st.error(f"API unreachable: {exc}")
    st.stop()

try:
    stages = api.stages()
    render_pipeline_health(stages)
except Exception as exc:
    st.error(f"Failed to load stages: {exc}")
