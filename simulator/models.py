from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PipelineEvent:
    """
    I'm representing a pipeline event as a flat dataclass rather than a Pydantic model
    because these objects are created at high frequency inside the simulator's hot loop.
    Pydantic's field validation adds roughly 3x construction overhead at 100k events/s —
    validation belongs at the ingestion boundary where untrusted data enters the system,
    not inside a generator that we control entirely.

    A PipelineEvent is a single observable unit of work passing through one pipeline
    stage: one batch read, one transform execution, or one write flush. Every field maps
    to a metric a real data pipeline would emit. The fault_label is synthetic — the
    FaultInjector writes it so detectors and the causal engine can measure their accuracy
    against ground truth without needing a separate label store.
    """

    stage_id: str
    event_time: datetime
    latency_ms: float
    row_count: int
    payload_bytes: int
    status: str
    fault_label: Optional[str]
