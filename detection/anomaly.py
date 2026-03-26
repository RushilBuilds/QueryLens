from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from simulator.models import PipelineEvent


@dataclass(frozen=True)
class AnomalyEvent:
    """
    Single schema for both CUSUM and EWMA rather than separate dataclasses —
    the AnomalyEventBus publishes to one Redpanda topic regardless of which
    detector fired. Separate schemas would force the bus to union two types and
    the causal layer to do the same. A detector_type discriminator keeps both
    simple.

    detector_value is named generically rather than 'cusum_value' or
    'ewma_value' so callers that only compare against threshold work identically
    for both detectors without branching.

    fault_label propagates from the originating PipelineEvent so the benchmark
    can compute precision/recall against ground-truth labels without joining
    back to pipeline_metrics on every query.
    """

    detector_type: str                    # 'cusum' or 'ewma'
    stage_id: str
    metric: str                           # 'latency_ms', 'row_count', 'error_rate'
    signal: Literal["upper", "lower"]     # which accumulator / control limit fired
    detector_value: float                 # accumulator value or EWMA statistic
    threshold: float                      # the h or L value that was exceeded
    z_score: float                        # the normalised input that triggered it
    detected_at: datetime                 # event_time of the triggering event
    fault_label: Optional[str] = None     # ground-truth label propagated from PipelineEvent


def extract_metric(event: PipelineEvent, metric: str) -> float:
    """
    Central extraction point so adding a new metric only requires changing
    this function, not every detector class. Extraction logic duplicated in
    CUSUMDetector and EWMADetector would drift out of sync the moment someone
    adds a metric to one and forgets the other.
    """
    if metric == "latency_ms":
        return event.latency_ms
    if metric == "row_count":
        return float(event.row_count)
    if metric == "error_rate":
        return 0.0 if event.status == "ok" else 1.0
    raise ValueError(
        f"Unknown metric {metric!r}. Valid metrics: 'latency_ms', 'row_count', 'error_rate'"
    )
