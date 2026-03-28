from __future__ import annotations

from enum import Enum


class HealingOutcome(Enum):
    PENDING   = "pending"    # action started, awaiting confirmation
    SUCCESS   = "success"    # stage recovered, action confirmed effective
    FAILED    = "failed"     # action executed but stage did not recover
    CANCELLED = "cancelled"  # operator overrode via API before execution
