"""
AnomalyTimelineView — Plotly event timeline showing anomalies per stage.

Renders a horizontal bar chart where each bar represents an anomaly event.
X-axis is time, Y-axis is stage_id. Hovering shows detector type and signal
value. Color encodes detector type (cusum vs ewma) so operators can distinguish
gradual drift from sudden spikes at a glance.
"""
from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

_DETECTOR_COLORS = {
    "cusum": "#e74c3c",   # red for gradual drift
    "ewma": "#3498db",    # blue for sudden spikes
}
_DEFAULT_COLOR = "#95a5a6"


def render_anomaly_timeline(anomalies: List[Dict[str, Any]], title: str = "Anomaly Timeline") -> None:
    """
    Renders a scatter-style timeline of anomaly events. Each dot is one anomaly
    positioned by detected_at (x) and stage_id (y). Scatter chosen over Gantt
    because anomalies are point events, not intervals — a Gantt bar with zero
    duration collapses to invisible.
    """
    if not anomalies:
        st.info("No anomalies in the selected time range.")
        return

    times = [a["detected_at"] for a in anomalies]
    stages = [a.get("stage_id", "unknown") for a in anomalies]
    detectors = [a.get("detector_type", "unknown") for a in anomalies]
    colors = [_DETECTOR_COLORS.get(d, _DEFAULT_COLOR) for d in detectors]
    hover_texts = [
        f"detector: {a.get('detector_type')}<br>"
        f"metric: {a.get('metric')}<br>"
        f"signal: {a.get('signal')}<br>"
        f"value: {a.get('detector_value', 0):.2f}<br>"
        f"threshold: {a.get('threshold', 0):.2f}"
        for a in anomalies
    ]

    fig = go.Figure()

    # Group by detector type for legend entries
    for dtype, color in _DETECTOR_COLORS.items():
        mask_times = [t for t, d in zip(times, detectors) if d == dtype]
        mask_stages = [s for s, d in zip(stages, detectors) if d == dtype]
        mask_hovers = [h for h, d in zip(hover_texts, detectors) if d == dtype]

        if mask_times:
            fig.add_trace(go.Scatter(
                x=mask_times,
                y=mask_stages,
                mode="markers",
                name=dtype.upper(),
                marker=dict(size=10, color=color, opacity=0.8),
                hovertext=mask_hovers,
                hoverinfo="text",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Stage",
        height=max(300, len(set(stages)) * 60),
        showlegend=True,
        margin=dict(l=150),
    )

    st.plotly_chart(fig, use_container_width=True)
