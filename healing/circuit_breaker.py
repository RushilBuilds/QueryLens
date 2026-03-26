from __future__ import annotations

import math
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
