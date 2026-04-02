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
from dashboard.views.causal_graph import render_causal_graph
from dashboard.views.health import render_pipeline_health
from dashboard.views.healing import render_audit_trail, render_healing_activity, render_override_panel
from dashboard.views.timeline import render_anomaly_timeline

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
    stages = []

# ---------------------------------------------------------------------------
# Anomaly timeline
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Anomaly Timeline")

try:
    all_anomalies = []
    for stage in stages:
        resp = api.stage_anomalies(stage["stage_id"], page_size=50)
        for item in resp.get("items", []):
            item["stage_id"] = stage["stage_id"]
            all_anomalies.append(item)
    render_anomaly_timeline(all_anomalies)
except Exception as exc:
    st.error(f"Failed to load anomalies: {exc}")

# ---------------------------------------------------------------------------
# Causal graph
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Causal Graph")

try:
    localizations = api.localizations(page_size=1).get("items", [])
    render_causal_graph(stages, localizations)
except Exception as exc:
    st.error(f"Failed to render causal graph: {exc}")

# ---------------------------------------------------------------------------
# Healing activity
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Healing Activity")

try:
    healing_resp = api.healing_actions(page_size=50)
    healing_items = healing_resp.get("items", [])
    render_healing_activity(healing_items)
    render_override_panel(healing_items, api)
    render_audit_trail(healing_items, api)
except Exception as exc:
    st.error(f"Failed to load healing actions: {exc}")
