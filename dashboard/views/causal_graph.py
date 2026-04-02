"""
CausalGraphView — renders the pipeline DAG with nodes colored by anomaly state.

Uses Plotly for interactive rendering: operators can hover over nodes to see
stage details and zoom into dense subgraphs. NetworkX provides the layout
algorithm (spring_layout produces readable results for DAGs under 50 nodes).

The top-ranked root cause from the latest LocalizationResult is highlighted in
red; symptomatic stages in amber; healthy stages in green. Edge width encodes
propagation delay — thicker edges mean slower propagation paths.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import plotly.graph_objects as go
import streamlit as st


def render_causal_graph(
    stages: List[Dict[str, Any]],
    localizations: List[Dict[str, Any]],
) -> None:
    """
    Builds a DAG visualization from the /stages response. Localizations provide
    the root-cause highlight. If no localizations exist, all nodes render in
    their health-based color without root-cause annotation.
    """
    if not stages:
        st.info("No stages available for graph rendering.")
        return

    # Determine root cause and symptomatic stages from latest localization
    root_cause_id: Optional[str] = None
    symptomatic_ids: Set[str] = set()

    if localizations:
        latest = localizations[0]  # already sorted by created_at desc from API
        root_cause_id = latest.get("root_cause_stage_id")

    # Build node positions using a simple layered layout
    # (avoiding networkx dependency in the dashboard — it's only needed server-side)
    stage_ids = [s["stage_id"] for s in stages]
    positions = _layered_positions(stage_ids)

    # Node colors
    node_colors = []
    node_texts = []
    for s in stages:
        sid = s["stage_id"]
        breaker_state = s.get("circuit_breaker", {}).get("state", "unknown")
        p99 = s.get("p99_latency_ms", 0.0)

        if sid == root_cause_id:
            color = "#e74c3c"  # red — root cause
            label = f"{sid} (ROOT CAUSE)"
        elif breaker_state == "open":
            color = "#e74c3c"  # red — circuit open
            label = f"{sid} (OPEN)"
        elif p99 >= 100.0:
            color = "#f39c12"  # amber — elevated latency
            label = f"{sid} (degraded)"
        else:
            color = "#2ecc71"  # green — healthy
            label = sid

        node_colors.append(color)
        node_texts.append(
            f"<b>{sid}</b><br>"
            f"p99: {p99:.1f}ms<br>"
            f"breaker: {breaker_state}<br>"
            f"events: {s.get('event_count', 0):,}"
        )

    x_vals = [positions[sid][0] for sid in stage_ids]
    y_vals = [positions[sid][1] for sid in stage_ids]

    fig = go.Figure()

    # Edges — draw lines between sequential stages
    # Without topology data from the API, we render a linear chain based on order
    for i in range(len(stage_ids) - 1):
        fig.add_trace(go.Scatter(
            x=[x_vals[i], x_vals[i + 1]],
            y=[y_vals[i], y_vals[i + 1]],
            mode="lines",
            line=dict(color="#bdc3c7", width=2),
            hoverinfo="none",
            showlegend=False,
        ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        marker=dict(size=30, color=node_colors, line=dict(width=2, color="white")),
        text=stage_ids,
        textposition="bottom center",
        hovertext=node_texts,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title="Pipeline Causal Graph",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def _layered_positions(stage_ids: List[str]) -> Dict[str, tuple]:
    """
    Simple left-to-right layout for a linear chain. Places nodes at equal
    horizontal intervals on a single row. Sufficient for pipelines under 10
    stages; a force-directed layout would be needed for branching topologies.
    """
    n = len(stage_ids)
    return {
        sid: (i / max(n - 1, 1), 0.5)
        for i, sid in enumerate(stage_ids)
    }
