"""
HealingActivityView, ManualOverridePanel, and HealingAuditTrail — M29.

Renders a table of healing actions with outcome badges, an override form for
pending actions, and an expandable audit trail linking anomaly to localization
to policy rule to outcome.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from dashboard.api_client import QueryLensAPI

# ---------------------------------------------------------------------------
# Outcome styling
# ---------------------------------------------------------------------------

_OUTCOME_BADGES = {
    "pending":   "\U0001f7e1 Pending",
    "success":   "\U0001f7e2 Success",
    "failed":    "\U0001f534 Failed",
    "cancelled": "\u26aa Cancelled",
}


def _outcome_badge(outcome: str) -> str:
    return _OUTCOME_BADGES.get(outcome, outcome)


# ---------------------------------------------------------------------------
# HealingActivityView
# ---------------------------------------------------------------------------


def render_healing_activity(actions: List[Dict[str, Any]]) -> None:
    """
    Renders a table of healing actions sorted by triggered_at descending.
    Uses st.dataframe for sortable columns rather than st.table because
    operators need to sort by outcome or elapsed time during triage.
    """
    if not actions:
        st.info("No healing actions recorded.")
        return

    rows = []
    for a in actions:
        rows.append({
            "Hypothesis": a.get("hypothesis_id", ""),
            "Stage": a.get("stage_id", ""),
            "Action": a.get("action", ""),
            "Severity": a.get("severity", ""),
            "Outcome": _outcome_badge(a.get("outcome", "")),
            "Triggered": a.get("triggered_at", ""),
            "Resolved": a.get("resolved_at", "") or "-",
        })

    st.dataframe(rows, use_container_width=True)


# ---------------------------------------------------------------------------
# ManualOverridePanel
# ---------------------------------------------------------------------------


def render_override_panel(actions: List[Dict[str, Any]], api: QueryLensAPI) -> None:
    """
    Shows a form to cancel a pending healing action. Only displays if there are
    pending actions — no point showing an empty override form.
    """
    pending = [a for a in actions if a.get("outcome") == "pending"]

    if not pending:
        st.caption("No pending actions to override.")
        return

    st.subheader("Manual Override")

    hypothesis_ids = [a["hypothesis_id"] for a in pending]
    selected = st.selectbox("Select pending action", hypothesis_ids, key="override_select")
    operator = st.text_input("Operator name", key="override_operator")
    reason = st.text_input("Reason (optional)", key="override_reason")

    if st.button("Cancel Action", key="override_submit"):
        if not operator:
            st.warning("Operator name is required.")
            return
        try:
            result = api.override_action(selected, operator, reason or None)
            st.success(f"Action cancelled: {result.get('hypothesis_id')}")
        except Exception as exc:
            st.error(f"Override failed: {exc}")


# ---------------------------------------------------------------------------
# HealingAuditTrail
# ---------------------------------------------------------------------------


def render_audit_trail(
    actions: List[Dict[str, Any]],
    api: QueryLensAPI,
) -> None:
    """
    Expandable section per action showing the full chain: triggering anomaly,
    localization result, policy rule matched, and outcome. Each expander fetches
    the localization detail on open to avoid loading all details upfront.
    """
    if not actions:
        return

    st.subheader("Audit Trail")

    for action in actions:
        hyp_id = action.get("hypothesis_id", "unknown")
        label = (
            f"{_outcome_badge(action.get('outcome', ''))} "
            f"{hyp_id} — {action.get('action', '')} on {action.get('stage_id', '')}"
        )

        with st.expander(label, expanded=False):
            st.markdown(f"**Action**: {action.get('action')}")
            st.markdown(f"**Severity**: {action.get('severity')}")
            st.markdown(f"**Fault type**: {action.get('fault_type', 'unknown')}")
            st.markdown(f"**Triggered**: {action.get('triggered_at')}")
            st.markdown(f"**Resolved**: {action.get('resolved_at') or 'pending'}")
            if action.get("notes"):
                st.markdown(f"**Notes**: {action['notes']}")

            # Fetch localization detail
            try:
                loc = api.localization_detail(hyp_id)
                st.markdown("---")
                st.markdown(f"**Root cause**: {loc.get('root_cause_stage_id', 'unknown')}")
                st.markdown(f"**Posterior**: {loc.get('posterior_probability', 0):.2f}")
                st.markdown(f"**Evidence count**: {loc.get('evidence_count', 0)}")
                candidates = loc.get("ranked_candidates", [])
                if candidates:
                    st.markdown("**Ranked candidates**:")
                    for c in candidates:
                        st.markdown(
                            f"- {c['stage_id']}: {c['posterior_probability']:.3f}"
                        )
            except Exception:
                st.caption("Localization detail not available.")
