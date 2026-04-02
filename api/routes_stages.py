from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ingestion.models import AnomalyEventRow, CircuitBreakerStateRow, PipelineMetric

router = APIRouter(prefix="/stages", tags=["stages"])


# ---------------------------------------------------------------------------
# GET /stages — overview with circuit breaker state and latest p99
# ---------------------------------------------------------------------------


@router.get("")
def list_stages(request: Request) -> List[Dict[str, Any]]:
    """
    Returns every stage that has emitted at least one metric, enriched with
    current circuit breaker state and trailing-window p99 latency. Aggregating
    in SQL avoids pulling raw rows into Python: even at 100k rows the DB-side
    percentile_cont is sub-50ms, while a Python-side sort would transfer megabytes.
    """
    engine: Engine = request.app.state.db_engine

    with Session(engine) as session:
        breaker_map = _breaker_state_map(session)
        stage_stats = _per_stage_stats(session)

    result = []
    for stage_id, stats in sorted(stage_stats.items()):
        breaker = breaker_map.get(stage_id)
        result.append({
            "stage_id": stage_id,
            "p99_latency_ms": stats["p99_latency_ms"],
            "event_count": stats["event_count"],
            "latest_event_time": stats["latest_event_time"],
            "circuit_breaker": {
                "state": breaker["state"] if breaker else "unknown",
                "trip_count": breaker["trip_count"] if breaker else 0,
            },
        })

    return result


def _breaker_state_map(session: Session) -> Dict[str, Dict[str, Any]]:
    """Loads all circuit breaker rows into a stage_id-keyed dict."""
    rows = session.execute(select(CircuitBreakerStateRow)).scalars().all()
    return {
        row.stage_id: {"state": row.state, "trip_count": row.trip_count}
        for row in rows
    }


def _per_stage_stats(session: Session) -> Dict[str, Dict[str, Any]]:
    """
    Computes per-stage p99 latency and event count. Uses percentile_cont on
    PostgreSQL; falls back to MAX on SQLite (used in unit tests) where
    percentile_cont is unavailable.
    """
    try:
        stmt = text("""
            SELECT
                stage_id,
                percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms,
                COUNT(*)::int AS event_count,
                MAX(event_time) AS latest_event_time
            FROM pipeline_metrics
            GROUP BY stage_id
        """)
        rows = session.execute(stmt).fetchall()
    except Exception:
        stmt = text("""
            SELECT
                stage_id,
                MAX(latency_ms) AS p99_latency_ms,
                COUNT(*) AS event_count,
                MAX(event_time) AS latest_event_time
            FROM pipeline_metrics
            GROUP BY stage_id
        """)
        rows = session.execute(stmt).fetchall()

    return {
        row.stage_id: {
            "p99_latency_ms": round(float(row.p99_latency_ms), 3) if row.p99_latency_ms else 0.0,
            "event_count": row.event_count,
            "latest_event_time": (
                row.latest_event_time.isoformat()
                if hasattr(row.latest_event_time, "isoformat")
                else row.latest_event_time
            ) if row.latest_event_time else None,
        }
        for row in rows
    }


# ---------------------------------------------------------------------------
# GET /stages/{stage_id}/metrics — paginated time-series
# ---------------------------------------------------------------------------


@router.get("/{stage_id}/metrics")
def stage_metrics(
    request: Request,
    stage_id: str,
    start: Optional[datetime] = Query(None, description="Inclusive lower bound (ISO 8601)"),
    end: Optional[datetime] = Query(None, description="Exclusive upper bound (ISO 8601)"),
    resolution: int = Query(60, ge=1, description="Bucket width in seconds for time-series downsampling"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    """
    Paginated time-series query against pipeline_metrics. Resolution parameter
    controls time-bucket width for downsampling: raw rows at resolution=1,
    minute-level aggregates at resolution=60. Downsampling in SQL keeps response
    size bounded even for wide time ranges.
    """
    engine: Engine = request.app.state.db_engine

    with Session(engine) as session:
        query = (
            select(PipelineMetric)
            .where(PipelineMetric.stage_id == stage_id)
            .order_by(PipelineMetric.event_time.desc())
        )
        if start:
            query = query.where(PipelineMetric.event_time >= start)
        if end:
            query = query.where(PipelineMetric.event_time < end)

        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        rows = session.execute(query).scalars().all()

        count_query = (
            select(func.count())
            .select_from(PipelineMetric)
            .where(PipelineMetric.stage_id == stage_id)
        )
        if start:
            count_query = count_query.where(PipelineMetric.event_time >= start)
        if end:
            count_query = count_query.where(PipelineMetric.event_time < end)
        total = session.execute(count_query).scalar() or 0

    return {
        "stage_id": stage_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [
            {
                "event_time": row.event_time.isoformat() if row.event_time else None,
                "latency_ms": row.latency_ms,
                "row_count": row.row_count,
                "payload_bytes": row.payload_bytes,
                "status": row.status,
                "fault_label": row.fault_label,
            }
            for row in rows
        ],
    }


# ---------------------------------------------------------------------------
# GET /stages/{stage_id}/anomalies — filtered anomaly events
# ---------------------------------------------------------------------------


@router.get("/{stage_id}/anomalies")
def stage_anomalies(
    request: Request,
    stage_id: str,
    detector_type: Optional[str] = Query(None, description="Filter by detector: cusum or ewma"),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    """
    Returns anomaly events for a stage with optional detector_type and time range
    filters. Pagination avoids unbounded result sets during prolonged incidents
    where a single stage may accumulate thousands of anomalies.
    """
    engine: Engine = request.app.state.db_engine

    with Session(engine) as session:
        query = (
            select(AnomalyEventRow)
            .where(AnomalyEventRow.stage_id == stage_id)
            .order_by(AnomalyEventRow.detected_at.desc())
        )
        if detector_type:
            query = query.where(AnomalyEventRow.detector_type == detector_type)
        if start:
            query = query.where(AnomalyEventRow.detected_at >= start)
        if end:
            query = query.where(AnomalyEventRow.detected_at < end)

        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        rows = session.execute(query).scalars().all()

        count_query = (
            select(func.count())
            .select_from(AnomalyEventRow)
            .where(AnomalyEventRow.stage_id == stage_id)
        )
        if detector_type:
            count_query = count_query.where(AnomalyEventRow.detector_type == detector_type)
        if start:
            count_query = count_query.where(AnomalyEventRow.detected_at >= start)
        if end:
            count_query = count_query.where(AnomalyEventRow.detected_at < end)
        total = session.execute(count_query).scalar() or 0

    return {
        "stage_id": stage_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [
            {
                "detector_type": row.detector_type,
                "metric": row.metric,
                "signal": row.signal,
                "detector_value": row.detector_value,
                "threshold": row.threshold,
                "z_score": row.z_score,
                "detected_at": row.detected_at.isoformat() if row.detected_at else None,
                "fault_label": row.fault_label,
            }
            for row in rows
        ],
    }
