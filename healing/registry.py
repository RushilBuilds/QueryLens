from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from healing.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
from ingestion.models import CircuitBreakerStateRow

_log = structlog.get_logger(__name__)


class CircuitBreakerRegistry:
    """
    Thread-safe registry of one CircuitBreaker per stage_id. Persisting state to
    PostgreSQL means a process restart (e.g. after an OOM kill) does not reset
    trip_count — without persistence, the backoff schedule would restart from
    base_backoff_s and allow a recovering-but-still-flapping stage to trip the
    breaker repeatedly at the minimum interval.

    All in-memory access is protected by a single lock. Persistence is a separate
    upsert; callers decide when to call persist() rather than auto-persisting on
    every state change to avoid synchronous DB writes on the hot path.
    """

    def __init__(
        self,
        database_url: str,
        config: CircuitBreakerConfig,
    ) -> None:
        self._config = config
        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, stage_id: str) -> CircuitBreaker:
        """
        Returns the CircuitBreaker for stage_id, creating and loading it from
        PostgreSQL if it does not exist in memory. Safe to call concurrently.
        """
        with self._lock:
            if stage_id not in self._breakers:
                self._breakers[stage_id] = self._load_or_create(stage_id)
            return self._breakers[stage_id]

    def persist(self, stage_id: str) -> None:
        """
        Upserts the current in-memory state for stage_id to PostgreSQL.
        Uses INSERT ... ON CONFLICT DO UPDATE so the first persist creates the row
        and subsequent calls update it — no separate create/update paths.
        """
        with self._lock:
            breaker = self._breakers.get(stage_id)
        if breaker is None:
            return

        now = datetime.now(tz=timezone.utc)
        stmt = pg_insert(CircuitBreakerStateRow).values(
            stage_id=breaker.stage_id,
            state=breaker.state.value,
            failure_count=breaker.failure_count,
            trip_count=breaker.trip_count,
            opened_at=breaker.opened_at,
            updated_at=now,
        ).on_conflict_do_update(
            index_elements=["stage_id"],
            set_={
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "trip_count": breaker.trip_count,
                "opened_at": breaker.opened_at,
                "updated_at": now,
            },
        )
        with Session(self._engine) as session:
            session.execute(stmt)
            session.commit()

        _log.info(
            "circuit_breaker_persisted",
            stage_id=stage_id,
            state=breaker.state.value,
            trip_count=breaker.trip_count,
        )

    def all_stage_ids(self) -> List[str]:
        """Returns stage_ids of all currently tracked breakers."""
        with self._lock:
            return list(self._breakers.keys())

    def close(self) -> None:
        self._engine.dispose()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_or_create(self, stage_id: str) -> CircuitBreaker:
        """
        Attempts to load persisted state from PostgreSQL. Falls back to a fresh
        CLOSED breaker if no row exists — first encounter of a stage is always healthy.
        Called with self._lock held.
        """
        with Session(self._engine) as session:
            row = session.execute(
                select(CircuitBreakerStateRow).where(
                    CircuitBreakerStateRow.stage_id == stage_id
                )
            ).scalar_one_or_none()

        breaker = CircuitBreaker(stage_id=stage_id, config=self._config)

        if row is not None:
            breaker.state = CircuitBreakerState(row.state)
            breaker.failure_count = row.failure_count
            breaker.trip_count = row.trip_count
            breaker.opened_at = row.opened_at

        return breaker
