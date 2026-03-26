from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PipelineEvent:
    """
    Flat dataclass rather than Pydantic model — validation belongs at the ingestion
    boundary, not inside a generator hot loop where 3x construction overhead matters.

    A single observable unit of work through one pipeline stage. fault_label is
    synthetic ground truth written by FaultInjector so detectors can measure accuracy
    without a separate label store.
    """

    stage_id: str
    event_time: datetime
    latency_ms: float
    row_count: int
    payload_bytes: int
    status: str
    fault_label: Optional[str]
    trace_id: Optional[str] = None
