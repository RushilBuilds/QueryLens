from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ingestion.models import FaultLocalizationRow

router = APIRouter(prefix="/localizations", tags=["localizations"])


# ---------------------------------------------------------------------------
# GET /localizations — paginated list
# ---------------------------------------------------------------------------


@router.get("")
def list_localizations(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    """
    Paginated list ordered by created_at descending so operators see the most
    recent localization first without scrolling. Returns the top candidate and
    posterior probability inline — enough to triage without drilling into detail.
    """
    engine: Engine = request.app.state.db_engine

    with Session(engine) as session:
        count_query = select(func.count()).select_from(FaultLocalizationRow)
        total = session.execute(count_query).scalar() or 0

        offset = (page - 1) * page_size
        query = (
            select(FaultLocalizationRow)
            .order_by(FaultLocalizationRow.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        rows = session.execute(query).scalars().all()

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [_localization_summary(row) for row in rows],
    }


# ---------------------------------------------------------------------------
# GET /localizations/{hypothesis_id} — full detail
# ---------------------------------------------------------------------------


@router.get("/{hypothesis_id}")
def get_localization(
    request: Request,
    hypothesis_id: str,
) -> Dict[str, Any]:
    """
    Full detail view including all evidence event IDs and ranked candidates.
    Returns 404 if hypothesis_id is unknown rather than an empty body —
    explicit absence is easier to handle in dashboard error flows.
    """
    engine: Engine = request.app.state.db_engine

    with Session(engine) as session:
        row = session.execute(
            select(FaultLocalizationRow).where(
                FaultLocalizationRow.hypothesis_id == hypothesis_id
            )
        ).scalar_one_or_none()

    if row is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "hypothesis not found"})

    return _localization_detail(row)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _localization_summary(row: FaultLocalizationRow) -> Dict[str, Any]:
    return {
        "hypothesis_id": row.hypothesis_id,
        "triggered_at": _safe_isoformat(row.triggered_at),
        "root_cause_stage_id": row.root_cause_stage_id,
        "posterior_probability": row.posterior_probability,
        "evidence_count": row.evidence_count,
        "created_at": _safe_isoformat(row.created_at),
    }


def _localization_detail(row: FaultLocalizationRow) -> Dict[str, Any]:
    ranked = json.loads(row.ranked_candidates_json) if row.ranked_candidates_json else []
    evidence = json.loads(row.evidence_json) if row.evidence_json else []

    return {
        "hypothesis_id": row.hypothesis_id,
        "triggered_at": _safe_isoformat(row.triggered_at),
        "root_cause_stage_id": row.root_cause_stage_id,
        "posterior_probability": row.posterior_probability,
        "evidence_count": row.evidence_count,
        "true_label": row.true_label,
        "created_at": _safe_isoformat(row.created_at),
        "ranked_candidates": [
            {"stage_id": c[0], "posterior_probability": c[1]} for c in ranked
        ],
        "evidence_events": evidence,
    }


def _safe_isoformat(val) -> Optional[str]:
    """Handles both datetime objects (PostgreSQL) and strings (SQLite)."""
    if val is None:
        return None
    return val.isoformat() if hasattr(val, "isoformat") else val
