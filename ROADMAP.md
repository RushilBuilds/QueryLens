# QueryLens Roadmap

A self-healing data pipeline observatory. Each milestone produces working, committable software — no placeholders.

---

## Milestone 1 — Fault Simulation & Synthetic Workload

Goal: generate realistic, controllable pipeline failures so every downstream component has something to observe.

- [ ] Implement `WorkloadProfile` — models query arrival rate, payload size distribution, and concurrency using configurable Poisson processes
- [ ] Build `FaultInjector` — injects latency spikes, dropped connections, schema drift, and partition skew with configurable probability and duration
- [ ] Implement `PipelineTopologyGraph` — defines the DAG of pipeline stages so fault propagation follows real causal paths
- [ ] Add `SimulationClock` — deterministic event replay for reproducible test scenarios
- [ ] Write integration tests that verify injected faults produce measurable signal in emitted metrics

---

## Milestone 2 — Metric Ingestion & Stream Processing

Goal: get pipeline telemetry into the system reliably and at scale.

- [ ] Build `RedpandaProducer` — publishes structured metric events (latency, throughput, error rate, row count) from each simulator stage
- [ ] Implement `MetricConsumer` — Redpanda consumer group with at-least-once delivery and offset commit on successful write
- [ ] Design and migrate the `pipeline_metrics` PostgreSQL schema using Alembic — partitioned by `event_time` for query efficiency
- [ ] Build `IngestionWorker` — async batch writer that flushes to PostgreSQL every N records or T seconds, whichever comes first
- [ ] Add structured logging via `structlog` with trace IDs that span producer → consumer → storage
- [ ] Expose Prometheus metrics for consumer lag, write throughput, and error rate

---

## Milestone 3 — Anomaly Detection Engine

Goal: detect real anomalies with low false-positive rate using statistically grounded methods.

- [ ] Implement `CUSUMDetector` — cumulative sum control chart for detecting gradual mean shift in latency and throughput signals
- [ ] Implement `EWMADetector` — exponentially weighted moving average for catching sudden spikes without CUSUM's accumulation lag
- [ ] Build `SlidingWindowAggregator` — maintains per-stage rolling statistics (p50, p95, p99) over configurable time windows
- [ ] Implement `SeasonalBaselineModel` — fits per-hour-of-week baselines so detectors compare against expected, not global, averages
- [ ] Build `AnomalyEventBus` — publishes confirmed anomalies back to Redpanda so causal and healing layers can subscribe independently
- [ ] Write detector benchmarks against known fault scenarios from Milestone 1 — must achieve >90% recall at <5% false positive rate

---

## Milestone 4 — Causal Fault Localization

Goal: identify which pipeline stage caused an anomaly, not just which stage exhibited symptoms.

- [ ] Build `CausalDAG` — wraps the pipeline topology from Milestone 1 as a `networkx.DiGraph` with edge weights representing signal propagation delay
- [ ] Implement `PearlDoCalculus` — uses Pearl's do-calculus to estimate the causal effect of each stage's metrics on downstream anomalies
- [ ] Build `FaultLocalizationEngine` — combines DAG structure with anomaly timestamps to rank candidate root causes by posterior probability
- [ ] Implement `AlertCorrelator` — deduplicates anomaly events within a configurable time window so a single fault doesn't generate N alerts
- [ ] Write causal accuracy tests using ground-truth fault injection labels from Milestone 1
- [ ] Store localization decisions in PostgreSQL `fault_localizations` table for audit and model improvement

---

## Milestone 5 — Self-Healing Actions, API & Observatory Dashboard

Goal: close the loop — detect, localize, and remediate without human intervention; expose everything through an API and dashboard.

- [ ] Implement `HealingPolicyEngine` — maps fault types to remediation actions (circuit break, rate limit, replay, reroute) using a priority-ordered rule table
- [ ] Build `CircuitBreaker` — per-stage breaker that trips after N consecutive failures and resets on exponential backoff
- [ ] Implement `ReplayOrchestrator` — replays failed message ranges from Redpanda with configurable backpressure
- [ ] Build FastAPI `PipelineAPI` — endpoints for pipeline status, anomaly history, active healings, and manual override
- [ ] Implement `HealingAuditLog` — every automated action is written to PostgreSQL with the triggering anomaly ID, policy matched, and outcome
- [ ] Build Streamlit `ObservatoryDashboard` — real-time view of pipeline health, anomaly timeline, causal graphs, and healing activity
- [ ] Write end-to-end integration test: inject fault → detect → localize → heal → confirm pipeline recovers
