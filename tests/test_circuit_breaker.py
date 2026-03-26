"""
Unit tests for CircuitBreaker FSM and CircuitBreakerConfig.

All tests are pure in-memory — no containers or external services required.
Time is injected via the `now` parameter on check_probe() so every timing
assertion is deterministic without sleep.

State transition coverage:
  CLOSED → OPEN (threshold reached)
  OPEN → HALF_OPEN (backoff elapsed)
  HALF_OPEN → CLOSED (probe success)
  HALF_OPEN → OPEN (probe failure, trip_count incremented)
  OPEN stays OPEN while backoff window active
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from healing.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)

T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# CircuitBreakerConfig — validation and backoff schedule
# ---------------------------------------------------------------------------


class TestCircuitBreakerConfig:

    def test_default_values(self) -> None:
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 5
        assert cfg.base_backoff_s == 1.0
        assert cfg.max_backoff_s == 60.0
        assert cfg.backoff_multiplier == 2.0

    def test_rejects_zero_failure_threshold(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_rejects_zero_base_backoff(self) -> None:
        with pytest.raises(ValueError, match="base_backoff_s"):
            CircuitBreakerConfig(base_backoff_s=0.0)

    def test_rejects_max_below_base(self) -> None:
        with pytest.raises(ValueError, match="max_backoff_s"):
            CircuitBreakerConfig(base_backoff_s=10.0, max_backoff_s=5.0)

    def test_rejects_multiplier_not_greater_than_one(self) -> None:
        with pytest.raises(ValueError, match="backoff_multiplier"):
            CircuitBreakerConfig(backoff_multiplier=1.0)

    def test_backoff_trip_1_returns_base(self) -> None:
        cfg = CircuitBreakerConfig(base_backoff_s=2.0, backoff_multiplier=3.0)
        assert cfg.backoff_for_trip(1) == 2.0

    def test_backoff_doubles_each_trip(self) -> None:
        cfg = CircuitBreakerConfig(base_backoff_s=1.0, backoff_multiplier=2.0, max_backoff_s=100.0)
        assert cfg.backoff_for_trip(1) == 1.0
        assert cfg.backoff_for_trip(2) == 2.0
        assert cfg.backoff_for_trip(3) == 4.0
        assert cfg.backoff_for_trip(4) == 8.0

    def test_backoff_capped_at_max(self) -> None:
        cfg = CircuitBreakerConfig(base_backoff_s=1.0, backoff_multiplier=2.0, max_backoff_s=5.0)
        assert cfg.backoff_for_trip(10) == 5.0

    def test_backoff_trip_zero_returns_base(self) -> None:
        cfg = CircuitBreakerConfig(base_backoff_s=3.0)
        assert cfg.backoff_for_trip(0) == 3.0


# ---------------------------------------------------------------------------
# CircuitBreaker — initial state
# ---------------------------------------------------------------------------


class TestCircuitBreakerInitialState:

    def test_starts_closed(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig())
        assert cb.state == CircuitBreakerState.CLOSED

    def test_starts_with_zero_failures(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig())
        assert cb.failure_count == 0
        assert cb.trip_count == 0

    def test_starts_with_no_opened_at(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig())
        assert cb.opened_at is None


# ---------------------------------------------------------------------------
# CircuitBreaker — CLOSED → OPEN transition
# ---------------------------------------------------------------------------


class TestClosedToOpen:

    def test_trips_after_threshold_failures(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        tripped = cb.record_failure()
        assert tripped is True
        assert cb.state == CircuitBreakerState.OPEN

    def test_does_not_trip_before_threshold(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        result = cb.record_failure()
        assert result is False
        assert cb.state == CircuitBreakerState.CLOSED

    def test_trip_sets_opened_at(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.opened_at is not None

    def test_trip_increments_trip_count(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.trip_count == 1

    def test_trip_resets_failure_count(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=2))
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 0

    def test_success_resets_failure_count_in_closed(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    def test_failure_in_open_state_is_ignored(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        result = cb.record_failure()
        assert result is False
        assert cb.trip_count == 1  # not incremented again


# ---------------------------------------------------------------------------
# CircuitBreaker — OPEN → HALF_OPEN transition
# ---------------------------------------------------------------------------


class TestOpenToHalfOpen:

    def test_check_probe_returns_false_before_backoff_elapsed(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, base_backoff_s=30.0)
        cb = CircuitBreaker("src", cfg)
        cb.record_failure()
        cb.opened_at = T0
        result = cb.check_probe(now=T0 + timedelta(seconds=10.0))
        assert result is False
        assert cb.state == CircuitBreakerState.OPEN

    def test_check_probe_transitions_to_half_open_after_backoff(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, base_backoff_s=10.0)
        cb = CircuitBreaker("src", cfg)
        cb.record_failure()
        cb.opened_at = T0
        result = cb.check_probe(now=T0 + timedelta(seconds=10.0))
        assert result is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_check_probe_at_exactly_backoff_boundary(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, base_backoff_s=5.0)
        cb = CircuitBreaker("src", cfg)
        cb.record_failure()
        cb.opened_at = T0
        result = cb.check_probe(now=T0 + timedelta(seconds=5.0))
        assert result is True

    def test_check_probe_returns_false_when_closed(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig())
        assert cb.check_probe() is False

    def test_backoff_doubles_on_second_trip(self) -> None:
        cfg = CircuitBreakerConfig(
            failure_threshold=1, base_backoff_s=5.0, backoff_multiplier=2.0
        )
        cb = CircuitBreaker("src", cfg)
        # First trip: backoff = 5s
        cb.record_failure()
        cb.opened_at = T0
        cb.check_probe(now=T0 + timedelta(seconds=5.0))  # → HALF_OPEN
        cb.record_failure()                               # → OPEN again, trip_count=2
        cb.opened_at = T0
        # Second trip: backoff = 10s — 5s is not enough
        assert cb.check_probe(now=T0 + timedelta(seconds=5.0)) is False
        assert cb.check_probe(now=T0 + timedelta(seconds=10.0)) is True


# ---------------------------------------------------------------------------
# CircuitBreaker — HALF_OPEN transitions
# ---------------------------------------------------------------------------


class TestHalfOpenTransitions:

    def test_success_in_half_open_closes_breaker(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, base_backoff_s=0.001)
        cb = CircuitBreaker("src", cfg)
        cb.record_failure()
        cb.opened_at = T0
        cb.check_probe(now=T0 + timedelta(seconds=1.0))
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.opened_at is None

    def test_failure_in_half_open_re_trips(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, base_backoff_s=0.001)
        cb = CircuitBreaker("src", cfg)
        cb.record_failure()
        cb.opened_at = T0
        cb.check_probe(now=T0 + timedelta(seconds=1.0))
        assert cb.state == CircuitBreakerState.HALF_OPEN
        tripped = cb.record_failure()
        assert tripped is True
        assert cb.state == CircuitBreakerState.OPEN

    def test_failure_in_half_open_increments_trip_count(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, base_backoff_s=0.001)
        cb = CircuitBreaker("src", cfg)
        cb.record_failure()  # trip_count = 1
        cb.opened_at = T0
        cb.check_probe(now=T0 + timedelta(seconds=1.0))
        cb.record_failure()  # trip_count = 2
        assert cb.trip_count == 2


# ---------------------------------------------------------------------------
# CircuitBreaker — force reset
# ---------------------------------------------------------------------------


class TestForceReset:

    def test_reset_from_open_returns_to_closed(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        cb.reset()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_reset_clears_failure_count(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.reset()
        assert cb.failure_count == 0

    def test_reset_clears_opened_at(self) -> None:
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        cb.reset()
        assert cb.opened_at is None

    def test_reset_does_not_clear_trip_count(self) -> None:
        # trip_count intentionally survives reset: the backoff schedule is cumulative
        # across the lifetime of the breaker, not just the current open window.
        cb = CircuitBreaker("src", CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.trip_count == 1
        cb.reset()
        assert cb.trip_count == 1
