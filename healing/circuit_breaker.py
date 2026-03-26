from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional


class CircuitBreakerState(Enum):
    CLOSED = "closed"        # normal operation — requests pass through
    OPEN = "open"            # tripped — requests rejected, waiting for backoff
    HALF_OPEN = "half_open"  # probe — one test request allowed through


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """
    Separates the threshold (when to trip) from the backoff schedule (when to retry)
    because they are tuned independently: a high-throughput stage may need a low
    failure_threshold to protect downstream consumers, while the backoff schedule
    depends on how long the upstream service typically takes to recover.

    base_backoff_s doubles on every trip (exponential) up to max_backoff_s.
    A cap prevents the breaker from staying open indefinitely on a flapping stage.
    """

    failure_threshold: int = 5        # consecutive failures before tripping
    base_backoff_s: float = 1.0       # initial backoff after first trip
    max_backoff_s: float = 60.0       # ceiling on exponential backoff
    backoff_multiplier: float = 2.0   # factor applied per trip count

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError(
                f"failure_threshold must be >= 1, got {self.failure_threshold}"
            )
        if self.base_backoff_s <= 0.0:
            raise ValueError(
                f"base_backoff_s must be > 0, got {self.base_backoff_s}"
            )
        if self.max_backoff_s < self.base_backoff_s:
            raise ValueError(
                f"max_backoff_s ({self.max_backoff_s}) must be >= "
                f"base_backoff_s ({self.base_backoff_s})"
            )
        if self.backoff_multiplier <= 1.0:
            raise ValueError(
                f"backoff_multiplier must be > 1.0, got {self.backoff_multiplier}"
            )

    def backoff_for_trip(self, trip_count: int) -> float:
        """
        Returns the backoff duration (seconds) for the given trip count.
        Exponential: base * multiplier^(trip_count - 1), capped at max_backoff_s.
        trip_count=1 returns base_backoff_s; each subsequent trip multiplies by backoff_multiplier.
        """
        if trip_count < 1:
            return self.base_backoff_s
        raw = self.base_backoff_s * (self.backoff_multiplier ** (trip_count - 1))
        return min(raw, self.max_backoff_s)


class CircuitBreaker:
    """
    Three-state FSM: CLOSED → OPEN → HALF_OPEN → CLOSED (or back to OPEN on probe failure).

    failure_count tracks consecutive failures in CLOSED state only — a success resets it
    to zero. trip_count is never reset: it drives the exponential backoff schedule so a
    repeatedly flapping stage sees progressively longer open intervals.

    opened_at is set when the breaker trips and is used to determine when the backoff
    interval has elapsed. Callers drive time via the `now` parameter on check_probe() so
    the FSM is deterministic in tests without sleep.
    """

    def __init__(self, stage_id: str, config: CircuitBreakerConfig) -> None:
        self.stage_id = stage_id
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count: int = 0
        self.trip_count: int = 0
        self.opened_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def record_failure(self) -> bool:
        """
        Records one consecutive failure. Trips the breaker (CLOSED → OPEN) when
        failure_count reaches failure_threshold. Returns True if this call tripped it.

        In HALF_OPEN state a failure immediately re-trips (HALF_OPEN → OPEN) and
        increments trip_count, which lengthens the next backoff interval.
        """
        if self.state == CircuitBreakerState.OPEN:
            return False

        self.failure_count += 1

        if self.state == CircuitBreakerState.HALF_OPEN or (
            self.failure_count >= self.config.failure_threshold
        ):
            self._trip()
            return True

        return False

    def record_success(self) -> None:
        """
        Records a success. In HALF_OPEN state this closes the breaker (probe passed).
        In CLOSED state it resets the consecutive failure counter.
        OPEN state successes are ignored — no probe is in flight.
        """
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._reset()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def check_probe(self, now: Optional[datetime] = None) -> bool:
        """
        Returns True and transitions OPEN → HALF_OPEN if the backoff interval has
        elapsed, signalling that one probe request should be allowed through.
        Returns False if the breaker is still within the backoff window, or if the
        state is not OPEN.

        Callers should call this before routing a request when the breaker is OPEN.
        """
        if self.state != CircuitBreakerState.OPEN:
            return False
        if self.opened_at is None:
            return False

        _now = now or datetime.now(tz=timezone.utc)
        backoff = self.config.backoff_for_trip(self.trip_count)
        if (_now - self.opened_at).total_seconds() >= backoff:
            self.state = CircuitBreakerState.HALF_OPEN
            return True

        return False

    def reset(self) -> None:
        """Force-resets to CLOSED. Used by the HealingPolicyEngine after confirmed recovery."""
        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trip(self) -> None:
        self.state = CircuitBreakerState.OPEN
        self.trip_count += 1
        self.failure_count = 0
        self.opened_at = datetime.now(tz=timezone.utc)

    def _reset(self) -> None:
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.opened_at = None

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def current_backoff_s(self) -> float:
        """Backoff duration in seconds for the current trip_count."""
        return self.config.backoff_for_trip(self.trip_count)

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(stage_id={self.stage_id!r}, state={self.state.value}, "
            f"failures={self.failure_count}, trips={self.trip_count})"
        )
