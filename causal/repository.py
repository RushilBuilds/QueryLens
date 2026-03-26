from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from causal.localization import LocalizationResult
from detection.anomaly import AnomalyEvent
from ingestion.models import FaultLocalizationRow

_log = structlog.get_logger(__name__)


def _anomaly_to_dict(event: AnomalyEvent) -> dict:
    return {
        "detector_type": event.detector_type,
        "stage_id": event.stage_id,
        "metric": event.metric,
        "signal": event.signal,
        "detector_value": event.detector_value,
        "threshold": event.threshold,
        "z_score": event.z_score,
        "detected_at": event.detected_at.isoformat(),
        "fault_label": event.fault_label,
    }


def _dict_to_anomaly(d: dict) -> AnomalyEvent:
    return AnomalyEvent(
        detector_type=d["detector_type"],
        stage_id=d["stage_id"],
        metric=d["metric"],
        signal=d["signal"],  # type: ignore[arg-type]
        detector_value=d["detector_value"],
        threshold=d["threshold"],
        z_score=d["z_score"],
        detected_at=datetime.fromisoformat(d["detected_at"]),
        fault_label=d.get("fault_label"),
    )


class LocalizationRepository:
    """
    Synchronous write path rather than async: localization happens at tens-per-minute
    rates where the round-trip to PostgreSQL is dominated by network latency, not
    concurrency. Async adds complexity without measurable throughput benefit here.

    evidence_json stores the full AnomalyEvent field set so get_by_hypothesis_id
    can reconstruct a complete LocalizationResult without joining back to
    anomaly_events. Denormalising the evidence trades storage for query simplicity —
    the causal layer reads one row, the HealingAuditLog never re-queries the bus.
    """

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url, pool_pre_ping=True)

    def write(self, result: LocalizationResult) -> int:
        """
        Inserts a LocalizationResult as a single row. Returns the database-assigned id.
        Raises sqlalchemy.exc.IntegrityError on duplicate hypothesis_id — callers should
        treat that as a no-op (idempotent re-delivery from the bus).
        """
        top = result.ranked_candidates[0] if result.ranked_candidates else None
        row = FaultLocalizationRow(
            hypothesis_id=result.hypothesis_id,
            triggered_at=result.triggered_at,
            root_cause_stage_id=top[0] if top else None,
            posterior_probability=top[1] if top else None,
            ranked_candidates_json=json.dumps(
                [[stage_id, prob] for stage_id, prob in result.ranked_candidates]
            ),
            evidence_json=json.dumps(
                [_anomaly_to_dict(e) for e in result.evidence_events]
            ),
            evidence_count=len(result.evidence_events),
            true_label=None,
            created_at=datetime.now(tz=timezone.utc),
        )
        with Session(self._engine) as session:
            session.add(row)
            session.flush()
            db_id: int = row.id  # type: ignore[assignment]
            session.commit()

        _log.info(
            "localization_persisted",
            hypothesis_id=result.hypothesis_id,
            db_id=db_id,
            top_candidate=top[0] if top else None,
            evidence_count=len(result.evidence_events),
        )
        return db_id

    def get_by_hypothesis_id(
        self, hypothesis_id: str
    ) -> Optional[LocalizationResult]:
        """
        Reconstructs a LocalizationResult from the persisted row. Returns None if
        no row exists for the given hypothesis_id.
        """
        with Session(self._engine) as session:
            row = session.execute(
                select(FaultLocalizationRow).where(
                    FaultLocalizationRow.hypothesis_id == hypothesis_id
                )
            ).scalar_one_or_none()

        if row is None:
            return None

        ranked = tuple(
            (entry[0], entry[1])
            for entry in json.loads(row.ranked_candidates_json)
        )
        evidence = tuple(
            _dict_to_anomaly(d) for d in json.loads(row.evidence_json)
        )
        return LocalizationResult(
            hypothesis_id=row.hypothesis_id,
            triggered_at=row.triggered_at,
            evidence_events=evidence,
            ranked_candidates=ranked,
        )

    def close(self) -> None:
        self._engine.dispose()
