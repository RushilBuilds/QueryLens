"""
Unit and integration tests for SeasonalBaselineModel, BaselineFitter, BaselineStore.

I'm splitting tests into two classes: pure in-memory tests (no DB) and
container-backed integration tests. The split keeps the fast unit tests runnable
without Docker and makes the integration boundary explicit — any test in
TestBaselineFitterIntegration that fails must be a DB or migration problem,
not a baseline arithmetic bug.
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
from testcontainers.postgres import PostgresContainer

from detection.baseline import (
    BaselineEntry,
    BaselineFitter,
    BaselineKey,
    BaselineStore,
    SeasonalBaselineModel,
    _hour_of_week,
)

POSTGRES_IMAGE = "postgres:16-alpine"
ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(entries: dict) -> SeasonalBaselineModel:
    return SeasonalBaselineModel(entries)


def _entry(mean: float, std: float, n: int = 50) -> BaselineEntry:
    return BaselineEntry(
        baseline_mean=mean,
        baseline_std=std,
        sample_count=n,
        fitted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


# ---------------------------------------------------------------------------
# _hour_of_week helper
# ---------------------------------------------------------------------------


class TestHourOfWeek:

    def test_monday_midnight_is_zero(self) -> None:
        # 2024-01-01 was a Monday
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert _hour_of_week(dt) == 0

    def test_monday_noon_is_12(self) -> None:
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert _hour_of_week(dt) == 12

    def test_tuesday_midnight_is_24(self) -> None:
        dt = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert _hour_of_week(dt) == 24

    def test_sunday_midnight_is_144(self) -> None:
        # Sunday = weekday 6, so 6 * 24 = 144
        dt = datetime(2024, 1, 7, 0, 0, 0, tzinfo=timezone.utc)
        assert _hour_of_week(dt) == 144

    def test_sunday_2300_is_167(self) -> None:
        dt = datetime(2024, 1, 7, 23, 0, 0, tzinfo=timezone.utc)
        assert _hour_of_week(dt) == 167


# ---------------------------------------------------------------------------
# SeasonalBaselineModel unit tests
# ---------------------------------------------------------------------------


class TestSeasonalBaselineModel:

    def test_get_returns_entry_for_known_key(self) -> None:
        key = BaselineKey("src", 8, "latency_ms")
        model = _make_model({key: _entry(50.0, 5.0)})
        entry = model.get(key)
        assert entry is not None
        assert entry.baseline_mean == 50.0

    def test_get_returns_none_for_missing_key(self) -> None:
        model = _make_model({})
        assert model.get(BaselineKey("src", 0, "latency_ms")) is None

    def test_len_reflects_number_of_entries(self) -> None:
        entries = {
            BaselineKey("src", i, "latency_ms"): _entry(float(i), 1.0)
            for i in range(10)
        }
        model = _make_model(entries)
        assert len(model) == 10

    def test_z_score_computed_correctly(self) -> None:
        """
        I'm testing with mean=50, std=10, value=60 to get an expected z-score
        of 1.0. This is the simplest possible check that the formula
        (value - mean) / std is applied in the correct order — swapping
        numerator and denominator or negating would produce -1.0.
        """
        # 2024-01-01 00:08:00 → Monday 8am → hour_of_week = 8
        event_time = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
        key = BaselineKey("src", 8, "latency_ms")
        model = _make_model({key: _entry(mean=50.0, std=10.0)})

        z = model.z_score("src", event_time, "latency_ms", 60.0)
        assert z == pytest.approx(1.0)

    def test_z_score_returns_none_when_std_is_zero(self) -> None:
        """
        I'm asserting None (not 0.0 or inf) when std=0 because returning
        either of those would propagate into CUSUM as a meaningful deviation.
        None forces the caller to skip the update — the correct behaviour when
        the baseline is degenerate.
        """
        event_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        key = BaselineKey("src", 0, "latency_ms")
        model = _make_model({key: _entry(mean=50.0, std=0.0)})

        assert model.z_score("src", event_time, "latency_ms", 60.0) is None

    def test_z_score_returns_none_for_missing_key(self) -> None:
        model = _make_model({})
        event_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert model.z_score("src", event_time, "latency_ms", 50.0) is None

    def test_z_score_uses_event_time_not_wall_time(self) -> None:
        """
        I'm verifying that a historical event_time (different hour_of_week than
        the current wall time) correctly looks up the historical slot. If the
        implementation used wall time, this test would fail unless the test
        happened to run at exactly Monday 8am.
        """
        # Monday 8am = hour_of_week 8
        monday_8am = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
        # Monday 10am = hour_of_week 10
        monday_10am = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        model = _make_model({
            BaselineKey("src", 8, "latency_ms"): _entry(mean=50.0, std=5.0),
            BaselineKey("src", 10, "latency_ms"): _entry(mean=200.0, std=20.0),
        })

        z_8am = model.z_score("src", monday_8am, "latency_ms", 55.0)
        z_10am = model.z_score("src", monday_10am, "latency_ms", 220.0)

        # 55 with baseline(50, 5) → z = 1.0
        assert z_8am == pytest.approx(1.0)
        # 220 with baseline(200, 20) → z = 1.0
        assert z_10am == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BaselineStore unit tests (no DB — uses a stub fitter)
# ---------------------------------------------------------------------------


class _StubFitter:
    """
    I'm using a hand-rolled stub rather than unittest.mock because the mock
    library's call_count is reset on assignment, which would require careful
    ordering in tests. The stub's call_count is an explicit integer that
    only increments on fit_and_persist() — easier to reason about in assertions.
    """

    def __init__(self, model: SeasonalBaselineModel) -> None:
        self._model = model
        self.call_count = 0

    def fit_and_persist(self) -> SeasonalBaselineModel:
        self.call_count += 1
        return self._model


class TestBaselineStore:

    def _empty_model(self) -> SeasonalBaselineModel:
        return _make_model({})

    def test_is_fresh_false_before_first_call(self) -> None:
        store = BaselineStore(_StubFitter(self._empty_model()), ttl_s=60.0)
        assert not store.is_fresh()

    def test_get_model_triggers_fit_on_first_call(self) -> None:
        stub = _StubFitter(self._empty_model())
        store = BaselineStore(stub, ttl_s=60.0)
        store.get_model()
        assert stub.call_count == 1
        assert store.is_fresh()

    def test_get_model_uses_cache_within_ttl(self) -> None:
        """
        I'm using a very long TTL (3600s) so the cache is guaranteed to be
        fresh on the second call regardless of how slowly the test runs.
        Testing with a short TTL would require sleeping, which is fragile in CI.
        """
        stub = _StubFitter(self._empty_model())
        store = BaselineStore(stub, ttl_s=3600.0)
        store.get_model()
        store.get_model()
        assert stub.call_count == 1, (
            "Second get_model() within TTL must use cached model, "
            f"got {stub.call_count} fitter calls"
        )

    def test_get_model_refits_after_ttl_expires(self) -> None:
        """
        I'm using ttl_s=0.05 (50ms) so the test doesn't need a long sleep.
        The sleep is 80ms to give a 30ms margin above the TTL before the
        second get_model() call — enough to be reliable without being slow.
        """
        stub = _StubFitter(self._empty_model())
        store = BaselineStore(stub, ttl_s=0.05)
        store.get_model()
        time.sleep(0.08)
        store.get_model()
        assert stub.call_count == 2, (
            f"Expected 2 fitter calls after TTL expired, got {stub.call_count}"
        )

    def test_force_refresh_bypasses_ttl(self) -> None:
        stub = _StubFitter(self._empty_model())
        store = BaselineStore(stub, ttl_s=3600.0)
        store.get_model()
        store.force_refresh()
        assert stub.call_count == 2

    def test_rejects_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError, match="ttl_s"):
            BaselineStore(_StubFitter(self._empty_model()), ttl_s=0.0)


# ---------------------------------------------------------------------------
# BaselineFitter integration tests (real Postgres via testcontainers)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_db():
    """
    I'm using a module-scoped fixture with a fresh Postgres container so the
    baseline tests are isolated from the ingestion worker tests. The ingestion
    worker tests insert events at 2024 timestamps; mixing those with the
    recent timestamps this test inserts would corrupt the hour-of-week
    groupings the fitter produces.
    """
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        db_url = pg.get_connection_url().replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
        _run_migrations(db_url)
        yield db_url


def _insert_events(
    db_url: str,
    stage_id: str,
    event_times: List[datetime],
    latency_ms: float,
    row_count: int = 100,
    status: str = "ok",
) -> None:
    """
    I'm inserting directly via raw SQL rather than ORM to avoid importing
    PipelineMetric and triggering the partitioning constraint. The events
    land in the DEFAULT partition (timestamps are recent/2026) which is
    correct — the baseline fitter doesn't care which partition holds the data.
    """
    engine = create_engine(db_url)
    rows = [
        {
            "stage_id":     stage_id,
            "event_time":   t,
            "latency_ms":   latency_ms,
            "row_count":    row_count,
            "payload_bytes": 1024,
            "status":       status,
            "fault_label":  None,
            "trace_id":     None,
        }
        for t in event_times
    ]
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO pipeline_metrics
                    (stage_id, event_time, latency_ms, row_count,
                     payload_bytes, status, fault_label, trace_id)
                VALUES
                    (:stage_id, :event_time, :latency_ms, :row_count,
                     :payload_bytes, :status, :fault_label, :trace_id)
            """),
            rows,
        )
    engine.dispose()


class TestBaselineFitterIntegration:
    """
    I'm inserting 50 events per (stage, hour_of_week) slot to get stable
    enough statistics for the 5% recovery assertion. With fewer events the
    sample mean can deviate significantly from the true mean by chance —
    50 events gives a standard error of latency_std / sqrt(50) ≈ 0.7ms for
    a 5ms std, which is well within 5% of a 50ms true mean.
    """

    N_PER_SLOT = 50

    def _slot_times(self, how: int, n: int) -> List[datetime]:
        """
        Generate N recent timestamps that all fall in the given hour_of_week
        slot. I'm setting the candidate's hour to target_hour first, then
        walking backwards by day until the weekday matches. Subtracting whole
        timedelta(days=N) without replacing the hour would keep the clock at
        the current hour, never reaching the target hour.
        """
        now = datetime.now(timezone.utc)
        target_weekday = how // 24  # 0=Monday
        target_hour = how % 24
        # Anchor today at target_hour, then walk back day by day.
        today_at_target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        for days_back in range(8):
            candidate = today_at_target - timedelta(days=days_back)
            if candidate.weekday() == target_weekday and candidate < now:
                return [candidate + timedelta(seconds=i) for i in range(n)]
        raise RuntimeError(f"Could not find a recent timestamp for hour_of_week={how}")

    def test_fit_recovers_injected_latency_means_within_5_percent(
        self, baseline_db: str
    ) -> None:
        """
        I'm using two hour_of_week slots with very different latency means
        (50ms vs 150ms) so any cross-slot contamination in the SQL GROUP BY
        would be immediately visible — the recovered mean would be the average
        of the two (100ms) rather than the injected value.
        """
        # Use Monday 2am (how=2) and Monday 6am (how=6) — unlikely to clash
        # with wall clock time so _slot_times always finds a past occurrence.
        slot_a, slot_b = 2, 6
        true_mean_a, true_mean_b = 50.0, 150.0

        _insert_events(baseline_db, "src_probe", self._slot_times(slot_a, self.N_PER_SLOT), latency_ms=true_mean_a)
        _insert_events(baseline_db, "src_probe", self._slot_times(slot_b, self.N_PER_SLOT), latency_ms=true_mean_b)

        fitter = BaselineFitter(baseline_db, lookback_days=28)
        model = fitter.fit_and_persist()
        fitter.close()

        for slot, true_mean in [(slot_a, true_mean_a), (slot_b, true_mean_b)]:
            key = BaselineKey("src_probe", slot, "latency_ms")
            entry = model.get(key)
            assert entry is not None, f"No baseline fitted for hour_of_week={slot}"
            rel_err = abs(entry.baseline_mean - true_mean) / true_mean
            assert rel_err <= 0.05, (
                f"hour_of_week={slot}: recovered mean {entry.baseline_mean:.2f} "
                f"deviates {rel_err*100:.1f}% from true mean {true_mean} "
                f"(threshold 5%)"
            )

    def test_fit_populates_stage_baselines_table(self, baseline_db: str) -> None:
        """
        I'm asserting DB row count rather than just checking the in-memory model
        because the model is built from the query result — if the UPSERT failed
        silently, the model would still be correct but the table would be empty,
        breaking every future process restart that reads from the table.
        """
        engine = create_engine(baseline_db)
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM stage_baselines WHERE stage_id = 'src_probe'")
            ).scalar()
        engine.dispose()

        # 2 slots × 3 metrics = 6 rows minimum (may be more from previous test runs)
        assert count >= 6, (
            f"Expected at least 6 rows in stage_baselines for src_probe, got {count}"
        )

    def test_fit_returns_empty_model_with_no_data(self, baseline_db: str) -> None:
        """
        I'm testing a stage_id that was never inserted to verify fit_and_persist()
        returns an empty model rather than raising. An empty model is the correct
        result — the detection layer should gate on model.get() returning None
        rather than expecting the fitter to raise for missing stages.
        """
        fitter = BaselineFitter(baseline_db, lookback_days=28)
        model = fitter.fit_and_persist()
        fitter.close()

        # "ghost_stage" was never inserted — its key must not appear in the model.
        missing_key = BaselineKey("ghost_stage", 0, "latency_ms")
        assert model.get(missing_key) is None

    def test_fit_upserts_on_repeated_calls(self, baseline_db: str) -> None:
        """
        I'm verifying that calling fit_and_persist() twice on the same data
        does not duplicate rows. The UNIQUE constraint + ON CONFLICT DO UPDATE
        must leave exactly one row per (stage_id, hour_of_week, metric) slot.
        """
        fitter = BaselineFitter(baseline_db, lookback_days=28)
        fitter.fit_and_persist()
        fitter.fit_and_persist()
        fitter.close()

        engine = create_engine(baseline_db)
        with engine.connect() as conn:
            dup_count = conn.execute(text("""
                SELECT COUNT(*)
                FROM (
                    SELECT stage_id, hour_of_week, metric, COUNT(*) AS n
                    FROM stage_baselines
                    GROUP BY stage_id, hour_of_week, metric
                    HAVING COUNT(*) > 1
                ) dups
            """)).scalar()
        engine.dispose()

        assert dup_count == 0, (
            f"Found {dup_count} duplicate (stage_id, hour_of_week, metric) rows — "
            "UPSERT must not insert duplicates"
        )
