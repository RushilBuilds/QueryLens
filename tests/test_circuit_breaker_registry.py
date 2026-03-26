"""
Integration tests for CircuitBreakerRegistry against a real PostgreSQL container.

Verifies that:
- get() returns a fresh CLOSED breaker for an unknown stage
- persist() upserts state so a new registry instance reloads it on get()
- trip_count survives a registry restart (the core invariant for exponential backoff)
- opened_at is preserved across restarts
- concurrent get() calls are safe (thread-safety smoke test)
- all_stage_ids() reflects loaded breakers

No Redpanda required — only PostgreSQL is needed for state persistence.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from healing.circuit_breaker import CircuitBreakerConfig, CircuitBreakerState
from healing.registry import CircuitBreakerRegistry

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
T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


def _default_config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_threshold=3,
        base_backoff_s=1.0,
        max_backoff_s=30.0,
        backoff_multiplier=2.0,
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


class TestCircuitBreakerRegistryBasic:

    def test_get_returns_closed_breaker_for_unknown_stage(
        self, db_url: str
    ) -> None:
        reg = CircuitBreakerRegistry(db_url, _default_config())
        cb = reg.get("new_stage")
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.trip_count == 0
        reg.close()

    def test_get_returns_same_instance_on_repeated_calls(
        self, db_url: str
    ) -> None:
        reg = CircuitBreakerRegistry(db_url, _default_config())
        cb1 = reg.get("stage_identity")
        cb2 = reg.get("stage_identity")
        assert cb1 is cb2
        reg.close()

    def test_all_stage_ids_reflects_loaded_breakers(self, db_url: str) -> None:
        reg = CircuitBreakerRegistry(db_url, _default_config())
        reg.get("stage_a")
        reg.get("stage_b")
        ids = reg.all_stage_ids()
        assert "stage_a" in ids
        assert "stage_b" in ids
        reg.close()

    def test_persist_does_not_raise_for_untracked_stage(
        self, db_url: str
    ) -> None:
        reg = CircuitBreakerRegistry(db_url, _default_config())
        reg.persist("nonexistent_stage")  # should silently no-op
        reg.close()


class TestCircuitBreakerRegistryPersistence:

    def test_persist_and_reload_preserves_state(self, db_url: str) -> None:
        reg1 = CircuitBreakerRegistry(db_url, _default_config())
        cb = reg1.get("persist_state")
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # trips to OPEN
        assert cb.state == CircuitBreakerState.OPEN
        reg1.persist("persist_state")
        reg1.close()

        reg2 = CircuitBreakerRegistry(db_url, _default_config())
        reloaded = reg2.get("persist_state")
        assert reloaded.state == CircuitBreakerState.OPEN
        reg2.close()

    def test_persist_and_reload_preserves_trip_count(self, db_url: str) -> None:
        reg1 = CircuitBreakerRegistry(db_url, _default_config())
        cb = reg1.get("persist_trip_count")
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.trip_count == 1
        reg1.persist("persist_trip_count")
        reg1.close()

        reg2 = CircuitBreakerRegistry(db_url, _default_config())
        reloaded = reg2.get("persist_trip_count")
        assert reloaded.trip_count == 1
        reg2.close()

    def test_persist_and_reload_preserves_opened_at(self, db_url: str) -> None:
        reg1 = CircuitBreakerRegistry(db_url, _default_config())
        cb = reg1.get("persist_opened_at")
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        original_opened_at = cb.opened_at
        assert original_opened_at is not None
        reg1.persist("persist_opened_at")
        reg1.close()

        reg2 = CircuitBreakerRegistry(db_url, _default_config())
        reloaded = reg2.get("persist_opened_at")
        assert reloaded.opened_at is not None
        # Compare to the second — timezone representation may differ.
        delta = abs(
            (reloaded.opened_at - original_opened_at).total_seconds()
        )
        assert delta < 1.0
        reg2.close()

    def test_persist_updates_existing_row_on_second_call(
        self, db_url: str
    ) -> None:
        reg = CircuitBreakerRegistry(db_url, _default_config())
        cb = reg.get("persist_update")
        cb.record_failure()
        reg.persist("persist_update")
        cb.record_failure()
        cb.record_failure()  # trips
        reg.persist("persist_update")  # should UPDATE, not INSERT a duplicate
        reg.close()

        reg2 = CircuitBreakerRegistry(db_url, _default_config())
        reloaded = reg2.get("persist_update")
        assert reloaded.state == CircuitBreakerState.OPEN
        reg2.close()

    def test_closed_state_reloads_with_no_opened_at(self, db_url: str) -> None:
        reg1 = CircuitBreakerRegistry(db_url, _default_config())
        cb = reg1.get("persist_closed")
        assert cb.state == CircuitBreakerState.CLOSED
        reg1.persist("persist_closed")
        reg1.close()

        reg2 = CircuitBreakerRegistry(db_url, _default_config())
        reloaded = reg2.get("persist_closed")
        assert reloaded.state == CircuitBreakerState.CLOSED
        assert reloaded.opened_at is None
        reg2.close()


class TestCircuitBreakerRegistryConcurrency:

    def test_concurrent_get_calls_are_safe(self, db_url: str) -> None:
        """
        50 threads each calling get() for the same stage_id must all receive
        the same instance and not corrupt internal state.
        """
        reg = CircuitBreakerRegistry(db_url, _default_config())
        results = []
        errors = []

        def worker():
            try:
                cb = reg.get("concurrent_stage")
                results.append(id(cb))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        reg.close()

        assert len(errors) == 0
        # All threads must have received the same instance.
        assert len(set(results)) == 1
