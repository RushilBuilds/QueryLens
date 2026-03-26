"""
Integration test for LocalizationRepository.

Runs against a real PostgreSQL container: the JSON serialisation round-trip and
hypothesis_id uniqueness constraint only surface against a real engine — SQLite
would accept the same DDL but behave differently on constraint violations and
timezone-aware datetime storage.

The test verifies that a LocalizationResult written via write() is fully
reconstructed by get_by_hypothesis_id(), including all ranked candidates and
evidence events.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from causal.localization import LocalizationResult
from causal.repository import LocalizationRepository
from detection.anomaly import AnomalyEvent

try:
    from testcontainers.postgres import PostgresContainer
    _CONTAINERS_AVAILABLE = True
except ImportError:
    _CONTAINERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CONTAINERS_AVAILABLE,
    reason="testcontainers not installed",
)

POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _anomaly(stage_id: str, offset_s: float = 0.0) -> AnomalyEvent:
    from datetime import timedelta
    return AnomalyEvent(
        detector_type="cusum",
        stage_id=stage_id,
        metric="latency_ms",
        signal="upper",
        detector_value=5.2,
        threshold=4.0,
        z_score=3.1,
        detected_at=T0 + timedelta(seconds=offset_s),
        fault_label="latency_spike",
    )


def _make_result(hypothesis_id: str = "hyp-001") -> LocalizationResult:
    evidence = (
        _anomaly("source_postgres", offset_s=0.0),
        _anomaly("transform_enrich", offset_s=5.0),
        _anomaly("sink_warehouse", offset_s=12.0),
    )
    candidates = (
        ("source_postgres", 0.72),
        ("transform_enrich", 0.20),
        ("sink_warehouse", 0.08),
    )
    return LocalizationResult(
        hypothesis_id=hypothesis_id,
        triggered_at=T0,
        evidence_events=evidence,
        ranked_candidates=candidates,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_url():
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        url = pg.get_connection_url().replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
        _run_migrations(url)
        yield url


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLocalizationRepositoryRoundTrip:

    def test_write_returns_positive_db_id(self, db_url: str) -> None:
        repo = LocalizationRepository(db_url)
        db_id = repo.write(_make_result("hyp-write-id"))
        repo.close()
        assert isinstance(db_id, int)
        assert db_id > 0

    def test_get_by_hypothesis_id_returns_none_for_missing(
        self, db_url: str
    ) -> None:
        repo = LocalizationRepository(db_url)
        result = repo.get_by_hypothesis_id("does-not-exist")
        repo.close()
        assert result is None

    def test_round_trip_preserves_hypothesis_id(self, db_url: str) -> None:
        original = _make_result("hyp-round-trip")
        repo = LocalizationRepository(db_url)
        repo.write(original)
        restored = repo.get_by_hypothesis_id("hyp-round-trip")
        repo.close()
        assert restored is not None
        assert restored.hypothesis_id == original.hypothesis_id

    def test_round_trip_preserves_triggered_at(self, db_url: str) -> None:
        original = _make_result("hyp-triggered-at")
        repo = LocalizationRepository(db_url)
        repo.write(original)
        restored = repo.get_by_hypothesis_id("hyp-triggered-at")
        repo.close()
        assert restored is not None
        assert restored.triggered_at == original.triggered_at

    def test_round_trip_preserves_ranked_candidates(self, db_url: str) -> None:
        original = _make_result("hyp-candidates")
        repo = LocalizationRepository(db_url)
        repo.write(original)
        restored = repo.get_by_hypothesis_id("hyp-candidates")
        repo.close()
        assert restored is not None
        assert len(restored.ranked_candidates) == len(original.ranked_candidates)
        for (orig_stage, orig_prob), (rest_stage, rest_prob) in zip(
            original.ranked_candidates, restored.ranked_candidates
        ):
            assert orig_stage == rest_stage
            assert abs(orig_prob - rest_prob) < 1e-9

    def test_round_trip_preserves_evidence_events(self, db_url: str) -> None:
        original = _make_result("hyp-evidence")
        repo = LocalizationRepository(db_url)
        repo.write(original)
        restored = repo.get_by_hypothesis_id("hyp-evidence")
        repo.close()
        assert restored is not None
        assert len(restored.evidence_events) == len(original.evidence_events)
        for orig, rest in zip(original.evidence_events, restored.evidence_events):
            assert orig.stage_id == rest.stage_id
            assert orig.metric == rest.metric
            assert orig.detector_type == rest.detector_type
            assert orig.fault_label == rest.fault_label
            assert orig.detected_at == rest.detected_at

    def test_round_trip_preserves_top_candidate(self, db_url: str) -> None:
        original = _make_result("hyp-top-candidate")
        repo = LocalizationRepository(db_url)
        repo.write(original)
        restored = repo.get_by_hypothesis_id("hyp-top-candidate")
        repo.close()
        assert restored is not None
        assert restored.top_candidate is not None
        assert restored.top_candidate[0] == "source_postgres"
        assert abs(restored.top_candidate[1] - 0.72) < 1e-9

    def test_write_records_visible_in_postgres(self, db_url: str) -> None:
        original = _make_result("hyp-db-visible")
        repo = LocalizationRepository(db_url)
        repo.write(original)
        repo.close()

        engine = create_engine(db_url)
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT hypothesis_id, root_cause_stage_id, posterior_probability, "
                    "evidence_count FROM fault_localizations "
                    "WHERE hypothesis_id = :hid"
                ),
                {"hid": "hyp-db-visible"},
            ).fetchone()
        engine.dispose()

        assert row is not None
        assert row.hypothesis_id == "hyp-db-visible"
        assert row.root_cause_stage_id == "source_postgres"
        assert abs(row.posterior_probability - 0.72) < 1e-6
        assert row.evidence_count == 3

    def test_result_with_no_candidates_writes_null_fields(
        self, db_url: str
    ) -> None:
        no_candidate_result = LocalizationResult(
            hypothesis_id="hyp-no-candidates",
            triggered_at=T0,
            evidence_events=(_anomaly("source_postgres"),),
            ranked_candidates=(),
        )
        repo = LocalizationRepository(db_url)
        repo.write(no_candidate_result)
        restored = repo.get_by_hypothesis_id("hyp-no-candidates")
        repo.close()
        assert restored is not None
        assert restored.ranked_candidates == ()
        assert restored.top_candidate is None

    def test_duplicate_hypothesis_id_raises(self, db_url: str) -> None:
        from sqlalchemy.exc import IntegrityError
        repo = LocalizationRepository(db_url)
        repo.write(_make_result("hyp-duplicate"))
        with pytest.raises(IntegrityError):
            repo.write(_make_result("hyp-duplicate"))
        repo.close()
