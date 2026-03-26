from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRICS = ("latency_ms", "row_count", "error_rate")


def _hour_of_week(dt: datetime) -> int:
    """
    Uses datetime.weekday() (Monday=0) rather than isoweekday() (Monday=1) so
    hour_of_week runs 0–167 with no off-by-one correction. PostgreSQL's
    EXTRACT(DOW) uses Sunday=0, so the fitter converts it with (dow + 6) % 7
    to match this function exactly.
    """
    return dt.weekday() * 24 + dt.hour


# ---------------------------------------------------------------------------
# BaselineKey and BaselineEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineKey:
    """
    Frozen dataclass rather than a plain tuple so fields are named at every
    call site. A tuple like ("src", 8, "latency_ms") is indistinguishable from
    ("latency_ms", 8, "src") to the type checker — a mismatch silently produces
    a cache miss and returns None z-scores.
    """

    stage_id: str
    hour_of_week: int  # 0–167: Monday 00:00 = 0, Sunday 23:00 = 167
    metric: str        # one of 'latency_ms', 'row_count', 'error_rate'


@dataclass(frozen=True)
class BaselineEntry:
    baseline_mean: float
    baseline_std: float
    sample_count: int
    fitted_at: datetime


# ---------------------------------------------------------------------------
# SeasonalBaselineModel
# ---------------------------------------------------------------------------


class SeasonalBaselineModel:
    """
    Pure in-memory lookup with no DB dependency so detectors can call z_score()
    on every event without touching the database on the hot path. All DB
    interaction is in BaselineFitter; this class is the read-only result of a fit.

    z_score() returns None when baseline_std == 0.0 rather than raising or
    returning infinity. A zero std means either one sample or all-identical
    values — the z-score is not interpretable and callers should skip the update.
    """

    def __init__(self, entries: Dict[BaselineKey, BaselineEntry]) -> None:
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: BaselineKey) -> Optional[BaselineEntry]:
        return self._entries.get(key)

    def z_score(
        self,
        stage_id: str,
        event_time: datetime,
        metric: str,
        value: float,
    ) -> Optional[float]:
        """
        Computes hour_of_week from event_time rather than wall time because the
        detection layer processes historical replays as well as live events.
        Wall time would assign the wrong seasonal slot to replayed events.
        """
        key = BaselineKey(
            stage_id=stage_id,
            hour_of_week=_hour_of_week(event_time),
            metric=metric,
        )
        entry = self._entries.get(key)
        if entry is None or entry.baseline_std == 0.0:
            return None
        return (value - entry.baseline_mean) / entry.baseline_std


# ---------------------------------------------------------------------------
# BaselineFitter
# ---------------------------------------------------------------------------


class BaselineFitter:
    """
    Uses a Python-side cutoff datetime rather than `NOW() - INTERVAL '28 days'`
    in SQL so tests can inspect the exact window without depending on DB server
    time, and to avoid SQLAlchemy's fragile INTERVAL parameterisation across
    Postgres versions.

    Single GROUP BY pass rather than three per-metric queries — one round trip
    is faster than three, and the group-by cost is identical for one or three
    columns in the SELECT list.
    """

    _UPSERT_SQL = text("""
        INSERT INTO stage_baselines
            (stage_id, hour_of_week, metric, baseline_mean, baseline_std,
             sample_count, fitted_at)
        VALUES
            (:stage_id, :hour_of_week, :metric, :baseline_mean, :baseline_std,
             :sample_count, :fitted_at)
        ON CONFLICT (stage_id, hour_of_week, metric) DO UPDATE
            SET baseline_mean = EXCLUDED.baseline_mean,
                baseline_std  = EXCLUDED.baseline_std,
                sample_count  = EXCLUDED.sample_count,
                fitted_at     = EXCLUDED.fitted_at
    """)

    _AGGREGATE_SQL = text("""
        SELECT
            stage_id,
            EXTRACT(DOW  FROM event_time AT TIME ZONE 'UTC')::int  AS dow,
            EXTRACT(HOUR FROM event_time AT TIME ZONE 'UTC')::int  AS hod,
            AVG(latency_ms)                                          AS latency_mean,
            COALESCE(STDDEV(latency_ms), 0.0)                       AS latency_std,
            AVG(CAST(row_count AS DOUBLE PRECISION))                AS row_count_mean,
            COALESCE(STDDEV(CAST(row_count AS DOUBLE PRECISION)), 0.0) AS row_count_std,
            AVG(CASE WHEN status != 'ok' THEN 1.0 ELSE 0.0 END)    AS error_rate_mean,
            COALESCE(
                STDDEV(CASE WHEN status != 'ok' THEN 1.0 ELSE 0.0 END),
                0.0
            )                                                        AS error_rate_std,
            COUNT(*)                                                 AS sample_count
        FROM pipeline_metrics
        WHERE event_time >= :cutoff
        GROUP BY stage_id, dow, hod
    """)

    def __init__(self, db_url: str, lookback_days: int = 28) -> None:
        if lookback_days < 1:
            raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
        self._engine = create_engine(db_url, pool_pre_ping=True)
        self._lookback_days = lookback_days

    def fit_and_persist(self) -> SeasonalBaselineModel:
        """
        SELECT and INSERT are separate transactions rather than INSERT ... SELECT
        because the aggregation query is read-only and long-running — holding a
        write lock for its duration would block ingestion writes. The two-
        transaction window means rows arriving between SELECT and INSERT are
        excluded, which is acceptable: baseline drift of a few events per hour
        has no detectable effect on z-score accuracy.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._lookback_days)
        fitted_at = datetime.now(timezone.utc)

        with self._engine.connect() as conn:
            rows = conn.execute(self._AGGREGATE_SQL, {"cutoff": cutoff}).fetchall()

        entries: Dict[BaselineKey, BaselineEntry] = {}
        upsert_params: List[dict] = []

        for row in rows:
            # Convert PostgreSQL's Sunday=0 DOW to ISO Monday=0 in Python rather
            # than SQL — the modulo expression is easier to audit than a SQL CASE,
            # and this runs once per (stage, slot) during fitting, not on the hot path.
            iso_dow = (int(row.dow) + 6) % 7
            how = iso_dow * 24 + int(row.hod)
            n = int(row.sample_count)

            metric_rows = [
                ("latency_ms", float(row.latency_mean), float(row.latency_std)),
                ("row_count",  float(row.row_count_mean), float(row.row_count_std)),
                ("error_rate", float(row.error_rate_mean), float(row.error_rate_std)),
            ]
            for metric, mean, std in metric_rows:
                key = BaselineKey(
                    stage_id=row.stage_id,
                    hour_of_week=how,
                    metric=metric,
                )
                entries[key] = BaselineEntry(
                    baseline_mean=mean,
                    baseline_std=std,
                    sample_count=n,
                    fitted_at=fitted_at,
                )
                upsert_params.append({
                    "stage_id":      row.stage_id,
                    "hour_of_week":  how,
                    "metric":        metric,
                    "baseline_mean": mean,
                    "baseline_std":  std,
                    "sample_count":  n,
                    "fitted_at":     fitted_at,
                })

        if upsert_params:
            with self._engine.begin() as conn:
                conn.execute(self._UPSERT_SQL, upsert_params)

        return SeasonalBaselineModel(entries)

    def close(self) -> None:
        self._engine.dispose()


# ---------------------------------------------------------------------------
# BaselineStore
# ---------------------------------------------------------------------------


class BaselineStore:
    """
    Uses time.monotonic() for the TTL check rather than datetime.now() because
    monotonic time never goes backwards. Wall time can jump forward (NTP) or
    backward (DST), making the cache appear perpetually fresh and silently
    serving stale baselines until the next restart.

    get_model() is not thread-safe by design — the ingestion worker and
    detection loop both run in a single thread. A lock would be premature
    complexity for a component that will never be called from multiple threads.
    """

    def __init__(self, fitter: BaselineFitter, ttl_s: float = 3600.0) -> None:
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be > 0, got {ttl_s}")
        self._fitter = fitter
        self._ttl_s = ttl_s
        self._model: Optional[SeasonalBaselineModel] = None
        self._last_fitted_monotonic: float = 0.0

    def get_model(self) -> SeasonalBaselineModel:
        """
        Re-fits lazily (on first call and on TTL expiry) rather than on a
        background timer because the detection loop is synchronous. A timer
        would require a thread and a lock with no latency benefit — the fit
        takes <100ms and the detection tick is 1 second.
        """
        now = time.monotonic()
        if self._model is None or (now - self._last_fitted_monotonic) >= self._ttl_s:
            self._model = self._fitter.fit_and_persist()
            self._last_fitted_monotonic = now
        return self._model

    def force_refresh(self) -> SeasonalBaselineModel:
        """Force a re-fit regardless of TTL state — used on startup and after schema changes."""
        self._model = self._fitter.fit_and_persist()
        self._last_fitted_monotonic = time.monotonic()
        return self._model

    def is_fresh(self) -> bool:
        """True if a model is cached and the TTL has not expired."""
        if self._model is None:
            return False
        return (time.monotonic() - self._last_fitted_monotonic) < self._ttl_s
