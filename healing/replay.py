from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReplayRequest:
    """
    Specifies a contiguous offset range on a single partition to replay.
    One request per partition rather than a multi-partition range because
    partition offsets are independent — a cross-partition range would require
    coordinating seek positions across consumer threads, adding concurrency
    complexity with no benefit for the single-partition replay path.

    replay_rate_limit_rps caps the replay throughput so a recovering stage
    is not re-flooded with the same message volume that caused the original
    failure. Lower values give the stage more time to recover; higher values
    minimise replay lag.
    """

    topic: str
    partition: int
    start_offset: int
    end_offset: int               # inclusive — replay includes this offset
    hypothesis_id: str            # links this replay to the triggering LocalizationResult
    replay_rate_limit_rps: float = 100.0

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError(
                f"start_offset must be >= 0, got {self.start_offset}"
            )
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be >= "
                f"start_offset ({self.start_offset})"
            )
        if self.replay_rate_limit_rps <= 0.0:
            raise ValueError(
                f"replay_rate_limit_rps must be > 0, got {self.replay_rate_limit_rps}"
            )

    @property
    def message_count(self) -> int:
        """Number of messages in the range (inclusive on both ends)."""
        return self.end_offset - self.start_offset + 1
