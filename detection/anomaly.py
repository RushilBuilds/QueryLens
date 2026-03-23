from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from simulator.models import PipelineEvent


@dataclass(frozen=True)
class AnomalyEvent:
    """
    I'm using a single AnomalyEvent schema for both CUSUM and EWMA rather than
    separate dataclasses per detector. The AnomalyEventBus in Milestone 13
    publishes to a single Redpanda topic regardless of which detector fired —
    separate schemas would require the bus to handle two types and the
    downstream causal layer to union them. A shared schema with a
    detector_type discriminator keeps the bus and consumer code simple.

    detector_value carries the raw accumulator (CUSUM S_upper/S_lower) or the
    EWMA statistic depending on detector_type. The field is intentionally
    named 'detector_value' rather than 'cusum_value' or 'ewma_value' so that
    callers that only need to compare against threshold work identically for
    both detectors without branching on detector_type.

    fault_label propagates from the originating PipelineEvent so that the M14
    benchmark can compute precision and recall against ground-truth fault labels
    without joining back to pipeline_metrics on every query.
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
    I'm extracting metric values here rather than inside each detector so that
    adding a new metric (e.g. 'payload_bytes') only requires changing this
    function — not every detector class. The alternative is metric extraction
    logic duplicated in CUSUMDetector and EWMADetector, which would drift out
    of sync the moment someone adds a metric to one but forgets the other.
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
