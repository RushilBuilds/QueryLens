from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ingestion.models import HealingActionRow

router = APIRouter(prefix="/healing", tags=["healing"])


# ---------------------------------------------------------------------------
# GET /healing/actions — paginated list
# ---------------------------------------------------------------------------


@router.get("/actions")
def list_healing_actions(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    outcome: Optional[str] = Query(None, description="Filter by outcome: pending, success, failed, cancelled"),
) -> Dict[str, Any]:
    """
    Paginated healing action history ordered by triggered_at descending.
    Optional outcome filter lets operators quickly find pending or failed actions
    without scrolling through resolved ones.
    """
    engine: Engine = request.app.state.db_engine

    with Session(engine) as session:
        base = select(HealingActionRow)
        count_base = select(func.count()).select_from(HealingActionRow)

        if outcome:
            base = base.where(HealingActionRow.outcome == outcome)
            count_base = count_base.where(HealingActionRow.outcome == outcome)

        total = session.execute(count_base).scalar() or 0

        offset = (page - 1) * page_size
        query = base.order_by(HealingActionRow.triggered_at.desc()).offset(offset).limit(page_size)
        rows = session.execute(query).scalars().all()

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [_action_to_dict(row) for row in rows],
    }


# ---------------------------------------------------------------------------
# POST /healing/actions/{hypothesis_id}/override — cancel a pending action
# ---------------------------------------------------------------------------


class OverrideRequest(BaseModel):
    """
    Operator identity required so the audit log records who cancelled the action.
    A free-text operator field rather than auth-token extraction keeps the API
    usable before an auth layer is wired in.
    """
    operator: str
    reason: Optional[str] = None


@router.post("/actions/{hypothesis_id}/override")
def override_action(
    request: Request,
    hypothesis_id: str,
    body: OverrideRequest,
) -> Dict[str, Any]:
    """
    Marks a pending healing action as cancelled. Returns 404 if no action exists
    for the hypothesis_id, 409 if the action is already resolved — re-cancelling
    a completed action would overwrite the real outcome.
    """
    engine: Engine = request.app.state.db_engine

    with Session(engine) as session:
        row = session.execute(
            select(HealingActionRow).where(
                HealingActionRow.hypothesis_id == hypothesis_id
            ).order_by(HealingActionRow.triggered_at.desc())
        ).scalar_one_or_none()

        if row is None:
            return JSONResponse(
                status_code=404,
                content={"detail": f"no healing action for hypothesis {hypothesis_id}"},
            )

        if row.outcome != "pending":
            return JSONResponse(
                status_code=409,
                content={
                    "detail": f"action already resolved as '{row.outcome}'",
                    "hypothesis_id": hypothesis_id,
                },
            )

        row.outcome = "cancelled"
        row.resolved_at = datetime.now(tz=timezone.utc)
        row.notes = f"cancelled by {body.operator}: {body.reason or 'no reason given'}"
        session.commit()

        return {
            "hypothesis_id": hypothesis_id,
            "outcome": "cancelled",
            "operator": body.operator,
            "resolved_at": row.resolved_at.isoformat() if hasattr(row.resolved_at, "isoformat") else row.resolved_at,
        }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _action_to_dict(row: HealingActionRow) -> Dict[str, Any]:
    return {
        "id": row.id,
        "hypothesis_id": row.hypothesis_id,
        "stage_id": row.stage_id,
        "action": row.action,
        "fault_type": row.fault_type,
        "severity": row.severity,
        "outcome": row.outcome,
        "triggered_at": _safe_isoformat(row.triggered_at),
        "resolved_at": _safe_isoformat(row.resolved_at),
        "notes": row.notes,
    }


def _safe_isoformat(val) -> Optional[str]:
    if val is None:
        return None
    return val.isoformat() if hasattr(val, "isoformat") else val
