from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Float, Integer, SmallInteger, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PipelineMetric(Base):
    """
    I'm mapping PipelineMetric to the partitioned parent table pipeline_metrics
    rather than to a specific monthly child table. SQLAlchemy routes writes through
    the parent and PostgreSQL's partition routing sends each row to the correct child
    based on event_time — the application layer never needs to know which partition
    a row lives in.

    The composite primary key (id, event_time) is a PostgreSQL partitioning
    constraint: every unique or primary key on a range-partitioned table must include
    all partition columns. Using id alone as PK would fail at DDL time. The id column
    uses GENERATED ALWAYS AS IDENTITY so the database owns sequencing rather than
    relying on application-side UUID generation, which avoids hot-spot contention at
    high insert rates.

    trace_id is stored as String(32) rather than UUID because OpenTelemetry trace IDs
    are 128-bit hex strings (e.g. "4bf92f3577b34da6a3ce929d0e0e4736") — forcing them
    through Postgres's uuid type would require the dashes format, adding a
    serialization/deserialization step that buys nothing for a field we only ever
    filter on, never do arithmetic with.
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
    I'm naming this AnomalyEventRow rather than AnomalyEvent to avoid shadowing
    the frozen dataclass of the same name in detection.anomaly. Both need to be
    importable in the same file (integration tests, AnomalyPersister) — a name
    collision would force an alias every time, whereas distinct names make the
    boundary between in-memory event and DB row explicit at the type level.

    created_at is the wall-clock time the persister wrote the row; detected_at
    is the event_time of the PipelineEvent that triggered the anomaly. The gap
    between the two is the end-to-end detection-to-persistence latency — useful
    for the M14 benchmark to assess pipeline lag.
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
    I'm storing baselines in their own table rather than as a materialised view
    or a JSON blob in pipeline_metrics because the fitter needs to UPSERT on
    each run (ON CONFLICT DO UPDATE). Materialised views are read-only and JSON
    blobs make the per-slot access pattern in the BaselineStore require
    deserialisation on every detection tick. A normalised row per
    (stage_id, hour_of_week, metric) lets the store load a full stage vector
    with a single indexed scan and updates become row-level writes, not full
    JSON document replacements.
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
    I'm keeping FaultLocalization as a stub for the same reason as AnomalyEvent —
    the columns (root_cause_stage_id, confidence_score, causal_path, algorithm_version)
    depend on the Bayesian localization API designed in Milestone 13. Committing to
    those column names now would create a migration diff the moment the causal engine
    interface is finalized.
    """

    __tablename__ = "fault_localizations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stage_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
