from __future__ import annotations

from datetime import datetime
from typing import Optional

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


class FaultLocalization(Base):
    """
    Stub table — columns (root_cause_stage_id, confidence_score, causal_path,
    algorithm_version) depend on the Bayesian localization API finalized in
    Milestone 13. Committing to column names early would create a migration
    diff the moment the causal engine interface changes.
    """

    __tablename__ = "fault_localizations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stage_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
