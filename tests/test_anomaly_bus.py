"""
Integration test for AnomalyEventBus and AnomalyPersister.

Runs against real Redpanda and PostgreSQL containers rather than mocking
either — the contract is end-to-end: an AnomalyEvent fired by a detector
must land in both Redpanda (for the causal layer) and PostgreSQL (for audit
and replay). Mocking verifies serialization glue but not delivery guarantees.

The fault_label alignment test is the correctness anchor: the label from the
PipelineEvent must survive serialization → Redpanda → deserialization →
PostgreSQL write intact.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from detection.anomaly import AnomalyEvent
from detection.baseline import BaselineEntry, BaselineKey, SeasonalBaselineModel
from detection.bus import AnomalyEventBus, AnomalyEventSerializer
from detection.cusum import CUSUMConfig, CUSUMDetector
from detection.ewma import EWMAConfig, EWMADetector
from detection.persister import AnomalyPersister
from ingestion.producer import ProducerHealthCheck
from simulator.models import PipelineEvent

try:
    import docker
    docker.from_env().ping()
    from testcontainers.kafka import KafkaContainer
    from testcontainers.postgres import PostgresContainer
    _CONTAINERS_AVAILABLE = True
except Exception:
    _CONTAINERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CONTAINERS_AVAILABLE,
    reason="Docker or testcontainers not available",
)

KAFKA_IMAGE = "confluentinc/cp-kafka:7.6.0"
POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"

SIM_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _flat_baseline(
    mean: float, std: float, stage_id: str = "src"
) -> SeasonalBaselineModel:
    """
    Uniform baseline across all 168 hour_of_week slots so test events at any
    timestamp produce a deterministic z-score. Exercises the bus and persister,
    not the baseline logic.
    """
    entries = {
        BaselineKey(stage_id, how, metric): BaselineEntry(
            baseline_mean=mean,
            baseline_std=std,
            sample_count=50,
            fitted_at=SIM_START,
        )
        for how in range(168)
        for metric in ("latency_ms", "row_count", "error_rate")
    }
    return SeasonalBaselineModel(entries)


def _faulted_event(offset_s: float, latency_ms: float, fault_label: str) -> PipelineEvent:
    return PipelineEvent(
        stage_id="src",
        event_time=SIM_START + timedelta(seconds=offset_s),
        latency_ms=latency_ms,
        row_count=100,
        payload_bytes=1024,
        status="ok",
        fault_label=fault_label,
        trace_id=None,
    )


def _run_detectors(
    events: List[PipelineEvent],
    baseline: SeasonalBaselineModel,
) -> List[AnomalyEvent]:
    """
    Runs both CUSUM and EWMA on the same event stream and collects all fired
    AnomalyEvents so the integration test confirms both detector outputs flow
    through the bus and persister identically.
    """
    cusum = CUSUMDetector(
        CUSUMConfig(decision_threshold=4.0, slack_parameter=0.5, metrics=("latency_ms",)),
        baseline,
    )
    ewma = EWMADetector(
        EWMAConfig(smoothing=0.3, control_limit_width=3.0, metrics=("latency_ms",)),
        baseline,
    )
    fired: List[AnomalyEvent] = []
    for event in events:
        fired.extend(cusum.update(event))
        fired.extend(ewma.update(event))
    return fired


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def infra():
    """
    Starts both containers once per module to avoid paying the 10–15s startup
    cost per test. The broker and Postgres instance are stateless between tests
    as long as each test uses a unique consumer group id.
    """
    with KafkaContainer(KAFKA_IMAGE).with_kraft() as kafka, \
         PostgresContainer(POSTGRES_IMAGE) as pg:

        db_url = pg.get_connection_url().replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
        _run_migrations(db_url)

        yield {
            "kafka_bootstrap": kafka.get_bootstrap_server(),
            "db_url": db_url,
        }


# ---------------------------------------------------------------------------
# Serialization unit tests (no containers needed)
# ---------------------------------------------------------------------------


class TestAnomalyEventSerializer:

    def test_round_trip_preserves_all_fields(self) -> None:
        """
        Full field round-trip rather than just checking for no exception — a
        serializer that silently drops fault_label would pass a no-exception
        test but break fault_label alignment in the persister.
        """
        anomaly = AnomalyEvent(
            detector_type="cusum",
            stage_id="etl",
            metric="latency_ms",
            signal="upper",
            detector_value=5.2,
            threshold=4.0,
            z_score=2.8,
            detected_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            fault_label="latency_spike",
        )
        data = AnomalyEventSerializer.serialize(anomaly)
        restored = AnomalyEventSerializer.deserialize(data)
        assert restored == anomaly

    def test_round_trip_with_null_fault_label(self) -> None:
        anomaly = AnomalyEvent(
            detector_type="ewma",
            stage_id="sink",
            metric="error_rate",
            signal="lower",
            detector_value=0.9,
            threshold=0.6,
            z_score=-3.1,
            detected_at=datetime(2024, 6, 15, 8, 30, 0, tzinfo=timezone.utc),
            fault_label=None,
        )
        data = AnomalyEventSerializer.serialize(anomaly)
        restored = AnomalyEventSerializer.deserialize(data)
        assert restored.fault_label is None

    def test_deserialize_rejects_wrong_schema_version(self) -> None:
        import json
        payload = json.dumps({"schema_version": 99, "detector_type": "cusum"})
        from detection.bus import AnomalySerializationError
        with pytest.raises(AnomalySerializationError, match="schema_version"):
            AnomalyEventSerializer.deserialize(payload.encode())

    def test_deserialize_rejects_malformed_json(self) -> None:
        from detection.bus import AnomalySerializationError
        with pytest.raises(AnomalySerializationError):
            AnomalyEventSerializer.deserialize(b"not-json")


# ---------------------------------------------------------------------------
# Integration: bus publishes, persister writes to Postgres
# ---------------------------------------------------------------------------


class TestAnomalyBusAndPersisterIntegration:

    def test_anomalies_land_in_redpanda_and_postgres(self, infra: dict) -> None:
        """
        Both destinations are asserted together because the two-destination
        guarantee is the point — mocking each in isolation would miss a persister
        that consumes but fails to write, or a bus that flushes but the broker
        drops the message.
        """
        bootstrap = infra["kafka_bootstrap"]
        db_url = infra["db_url"]

        # Faulted events: latency spike (z = 3.5 per event for EWMA, ramp for CUSUM)
        baseline = _flat_baseline(mean=50.0, std=10.0)
        events = [
            _faulted_event(float(i), latency_ms=85.0, fault_label="latency_spike")
            for i in range(10)
        ]
        anomalies = _run_detectors(events, baseline)
        assert len(anomalies) >= 1, "Expected at least one anomaly from the faulted stream"

        # Publish all anomalies to Redpanda.
        health = ProducerHealthCheck()
        bus = AnomalyEventBus(bootstrap_servers=bootstrap, health_check=health)
        for anomaly in anomalies:
            bus.publish(anomaly)
        remaining = bus.flush(timeout_s=15.0)
        assert remaining == 0, f"{remaining} anomaly messages not delivered"
        assert health.failed_delivery_count == 0

        # Persist from Redpanda to Postgres.
        # Unique group_id per run so re-running the module does not skip already-committed offsets.
        group_id = f"test-persister-{int(time.time())}"
        persister = AnomalyPersister(
            bootstrap_servers=bootstrap,
            database_url=db_url,
            group_id=group_id,
        )
        count = persister.consume_and_persist(timeout_s=10.0, max_messages=500)
        persister.close()
        bus.close()

        assert count == len(anomalies), (
            f"Expected {len(anomalies)} rows persisted, got {count}"
        )

        # Query Postgres and assert row contents.
        engine = create_engine(db_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT detector_type, stage_id, metric, signal, fault_label "
                     "FROM anomaly_events ORDER BY id")
            ).fetchall()

        assert len(rows) >= len(anomalies)

        # Spot-check the first anomaly's fields.
        first = rows[0]
        assert first.stage_id == "src"
        assert first.detector_type in ("cusum", "ewma")
        assert first.metric == "latency_ms"
        assert first.signal in ("upper", "lower")

        engine.dispose()

    def test_fault_label_alignment(self, infra: dict) -> None:
        """
        fault_label must survive the full path: PipelineEvent → detector →
        AnomalyEvent → serialization → Redpanda → deserialization →
        AnomalyEventRow → PostgreSQL. Any drop corrupts the benchmark's
        precision/recall computation.
        """
        bootstrap = infra["kafka_bootstrap"]
        db_url = infra["db_url"]

        baseline = _flat_baseline(mean=50.0, std=10.0)

        # Use a distinctive fault_label so we can isolate these rows in the DB.
        fault_label = "schema_drift_test"
        events = [
            _faulted_event(float(i), latency_ms=85.0, fault_label=fault_label)
            for i in range(5)
        ]
        anomalies = _run_detectors(events, baseline)
        assert all(a.fault_label == fault_label for a in anomalies), (
            "Detectors must propagate fault_label from PipelineEvent to AnomalyEvent"
        )

        health = ProducerHealthCheck()
        bus = AnomalyEventBus(bootstrap_servers=bootstrap, health_check=health)
        for anomaly in anomalies:
            bus.publish(anomaly)
        bus.flush(timeout_s=15.0)

        group_id = f"test-fault-label-{int(time.time())}"
        persister = AnomalyPersister(
            bootstrap_servers=bootstrap,
            database_url=db_url,
            group_id=group_id,
        )
        persister.consume_and_persist(timeout_s=10.0, max_messages=500)
        persister.close()
        bus.close()

        engine = create_engine(db_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT fault_label FROM anomaly_events "
                    "WHERE fault_label = :label"
                ),
                {"label": fault_label},
            ).fetchall()
        engine.dispose()

        assert len(rows) >= len(anomalies), (
            f"Expected at least {len(anomalies)} rows with fault_label='{fault_label}', "
            f"got {len(rows)}"
        )
        assert all(r.fault_label == fault_label for r in rows)
