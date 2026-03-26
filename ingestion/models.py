from __future__ import annotations

from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import BigInteger, DateTime, Float, Integer, SmallInteger, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PipelineMetric(Base):
    """
    Maps to the partitioned parent table pipeline_metrics rather than a specific
    monthly child table. SQLAlchemy routes writes through the parent; PostgreSQL
    partition routing sends each row to the correct child based on event_time.

    The composite primary key (id, event_time) is required by PostgreSQL: every
    unique or primary key on a range-partitioned table must include all partition
    columns. trace_id is String(32) rather than UUID because OpenTelemetry trace
    IDs are 128-bit hex strings — the UUID type would require dashes format,
    adding a serialization step for a field only ever filtered on, never computed.
    """

    __tablename__ = "pipeline_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stage_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, primary_key=True
    )
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    payload_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    fault_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)


class AnomalyEventRow(Base):
    """
    Named AnomalyEventRow rather than AnomalyEvent to avoid shadowing the frozen
    dataclass in detection.anomaly. Distinct names make the boundary between
    in-memory event and DB row explicit without requiring an alias at every import.

    created_at is the wall-clock time the persister wrote the row; detected_at
    is the event_time of the triggering PipelineEvent. The gap between the two
    is the end-to-end detection-to-persistence latency.
    """

    __tablename__ = "anomaly_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stage_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    detector_type: Mapped[str] = mapped_column(String(16), nullable=False)
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    signal: Mapped[str] = mapped_column(String(8), nullable=False)
    detector_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    z_score: Mapped[float] = mapped_column(Float, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    fault_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    schema_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class StageBaseline(Base):
    """
    Separate table rather than a materialised view or JSON blob because the
    fitter needs to UPSERT on each run. Materialised views are read-only; JSON
    blobs require deserialisation on every detection tick. A normalised row per
    (stage_id, hour_of_week, metric) loads a full stage vector with a single
    indexed scan and updates as row-level writes.
    """

    __tablename__ = "stage_baselines"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stage_id: Mapped[str] = mapped_column(String(64), nullable=False)
    hour_of_week: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    baseline_mean: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_std: Mapped[float] = mapped_column(Float, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FaultLocalizationRow(Base):
    """
    Named FaultLocalizationRow to avoid shadowing the LocalizationResult dataclass
    in causal.localization. Full schema replaces the M1 stub: hypothesis_id is the
    join key used by the HealingAuditLog; ranked_candidates_json and evidence_json
    are TEXT rather than JSONB to keep the model portable to SQLite in unit tests.
    """

    __tablename__ = "fault_localizations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    hypothesis_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    root_cause_stage_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    posterior_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ranked_candidates_json: Mapped[str] = mapped_column(sa.Text, nullable=False)
    evidence_json: Mapped[str] = mapped_column(sa.Text, nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False)
    true_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CircuitBreakerStateRow(Base):
    """
    Persists CircuitBreaker FSM state per stage so registry restarts do not reset
    trip_count — losing trip history would collapse the exponential backoff schedule
    back to base_backoff_s, letting a flapping stage flood downstream consumers again.

    opened_at is NULL when state=closed; the registry uses this to reconstruct the
    exact opened_at on reload rather than treating NULL as time.now().
    """

    __tablename__ = "circuit_breaker_states"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stage_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False)
    trip_count: Mapped[int] = mapped_column(Integer, nullable=False)
    opened_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
