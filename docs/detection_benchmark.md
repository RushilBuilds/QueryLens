# Detection Accuracy Benchmark

Precision, recall, and FPR for CUSUM and EWMA detectors across all six fault types.
All six fault types from Milestone 3 are exercised with known magnitudes.

**Thresholds:** recall ≥ 0.90, FPR ≤ 0.05

## Configuration

- Warmup events per trial: 50
- Fault window events: 30
- Recovery events per trial: 20
- Trials per fault type: 5
- CUSUM decision_threshold: 4.0, slack_parameter: 0.5
- EWMA smoothing: 0.3, control_limit_width: 3.0

## Results

| Detector | Fault Type | Recall | FPR | Mean Detection Lag (events) | Pass |
|---|---|---|---|---|---|
| cusum | dropped_connection | 1.00 | 0.000 | 0.6 | ✓ |
| cusum | error_burst | 1.00 | 0.000 | 0.2 | ✓ |
| cusum | latency_spike | 1.00 | 0.000 | 0.0 | ✓ |
| cusum | partition_skew | 1.00 | 0.000 | 0.0 | ✓ |
| cusum | schema_drift | 1.00 | 0.000 | 0.0 | ✓ |
| cusum | throughput_collapse | 1.00 | 0.000 | 1.0 | ✓ |
| ewma | dropped_connection | 1.00 | 0.000 | 0.6 | ✓ |
| ewma | error_burst | 1.00 | 0.000 | 0.2 | ✓ |
| ewma | latency_spike | 1.00 | 0.000 | 0.0 | ✓ |
| ewma | partition_skew | 1.00 | 0.000 | 0.0 | ✓ |
| ewma | schema_drift | 1.00 | 0.000 | 0.0 | ✓ |
| ewma | throughput_collapse | 1.00 | 0.000 | 0.0 | ✓ |

## Summary

All fault types pass both thresholds.
