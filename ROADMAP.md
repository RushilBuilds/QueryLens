# QueryLens Roadmap

A self-healing data pipeline observatory. Each milestone is one commit — functional, tested, and self-contained.

---

## Phase 1 — Simulator

### Milestone 1 — `WorkloadProfile`: model realistic query arrival patterns

Goal: produce a stream of synthetic pipeline events whose statistical properties match real production workloads — Poisson arrivals, variable payload sizes, and concurrency bursts.

- [x] Define `PipelineEvent` dataclass — fields: `stage_id`, `event_time`, `latency_ms`, `row_count`, `payload_bytes`, `status`
- [x] Implement `WorkloadProfile` — configures arrival rate (λ), payload size distribution (log-normal), and max concurrency per stage
- [x] Add `PoissonEventGenerator` — draws inter-arrival times from `numpy.random.exponential(1/λ)` and yields `PipelineEvent` objects
- [x] Write unit tests: verify arrival rate converges to λ over 10,000 samples and payload distribution matches configured parameters

---

### Milestone 2 — `PipelineTopologyGraph`: define stage relationships as a DAG

Goal: give every downstream component a causal map of the pipeline so fault propagation follows real dependency paths, not random noise.

- [x] Implement `PipelineStage` dataclass — fields: `stage_id`, `stage_type` (source / transform / sink), `upstream_ids`, `propagation_delay_ms`
- [x] Build `PipelineTopologyGraph` — wraps a `networkx.DiGraph`, validates acyclicity on construction, and exposes `downstream_stages(stage_id)`
- [x] Add `TopologyLoader` — loads stage definitions from a YAML file in `config/` so topology is configurable without code changes
- [x] Write tests: verify DAG rejects cycles, correctly resolves multi-hop downstream paths, and loads from fixture YAML

---

### Milestone 3 — `FaultInjector`: inject controlled, reproducible failures

Goal: inject six fault types into the workload stream so every detector and causal component has ground-truth labels to train and validate against.

- [x] Define `FaultSpec` dataclass — fields: `fault_type`, `target_stage_id`, `start_offset_s`, `duration_s`, `magnitude`, `seed`
- [x] Implement `FaultInjector` — applies faults to `PipelineEvent` streams: latency spike, dropped connection, schema drift, partition skew, throughput collapse, error burst
- [x] Add `FaultSchedule` — ordered list of `FaultSpec` objects; `FaultInjector` activates/deactivates faults by simulation clock tick
- [x] Ensure every injected event carries a `fault_label` field so downstream accuracy metrics have ground truth
- [x] Write tests: verify each fault type produces the expected statistical signature (e.g. latency spike raises p99 by ≥ configured magnitude)

---

### Milestone 4 — `SimulationClock` and full simulator integration

Goal: make the simulator deterministic and replayable so test scenarios are reproducible across machines and CI runs.

- [x] Implement `SimulationClock` — tick-based virtual clock; all generators and injectors advance by `tick_interval_ms` per step, never using wall time
- [x] Build `SimulatorEngine` — composes `WorkloadProfile`, `PipelineTopologyGraph`, `FaultInjector`, and `SimulationClock` into a single `run(n_ticks)` loop
- [x] Add `ScenarioConfig` — a single YAML-driven config that seeds the RNG, sets topology, workload, and fault schedule; same config always produces identical output
- [x] Write integration test: run two identical `ScenarioConfig` instances with the same seed, assert byte-for-byte identical event streams

---

## Phase 2 — Ingestion

### Milestone 5 — `PipelineMetric` schema and Alembic migration

Goal: establish the PostgreSQL schema before writing a single byte of ingestion code — schema changes are the hardest things to reverse.

- [x] Define `PipelineMetric` SQLAlchemy model — columns: `id`, `stage_id`, `event_time`, `latency_ms`, `row_count`, `payload_bytes`, `status`, `fault_label`, `trace_id`
- [x] Write Alembic `initial_schema` migration — creates `pipeline_metrics` table partitioned by `event_time` (monthly range partitions)
- [x] Add `AnomalyEvent` and `FaultLocalization` table stubs in the same migration — empty now, filled in later phases
- [x] Write a migration smoke test: apply migration against a real PostgreSQL container (via `testcontainers`), insert one row, assert round-trip

---

### Milestone 6 — `RedpandaProducer`: publish metric events to the stream

Goal: get simulator output onto the Redpanda topic reliably, with serialization that the consumer can version independently.

- [x] Implement `MetricEventSerializer` — serializes `PipelineEvent` to JSON with a `schema_version` field so consumer can handle future schema evolution
- [x] Build `RedpandaProducer` — wraps `confluent_kafka.Producer`, configures `acks=all` and `retries=5`, publishes to `pipeline.metrics` topic
- [x] Add `ProducerHealthCheck` — tracks per-topic delivery callbacks; exposes `failed_deliveries` counter for Prometheus
- [x] Write integration test: produce 1,000 events to a `testcontainers` Redpanda instance, assert all offsets committed and no delivery errors

---

### Milestone 7 — `MetricConsumer` and `IngestionWorker`: consume and persist

Goal: pull events off the topic and write them to PostgreSQL with at-least-once delivery guarantees and no silent data loss.

- [x] Implement `MetricConsumer` — Redpanda consumer group on `pipeline.metrics`; commits offset only after successful batch write, never before
- [x] Build `IngestionWorker` — async batch writer; flushes when batch reaches `batch_size` records OR `flush_interval_s` seconds elapse, whichever comes first
- [x] Add dead-letter handling — events that fail deserialization or validation are written to `pipeline.metrics.dlq` topic with error context, not silently dropped
- [x] Write integration test: produce events with one malformed record; assert all valid records land in PostgreSQL and malformed record lands in DLQ topic

---

### Milestone 8 — structured logging and Prometheus metrics for the ingestion layer

Goal: make the ingestion layer observable before it touches production data — no guessing about consumer lag or write throughput.

- [x] Wire `structlog` into `MetricConsumer` and `IngestionWorker` — every log line carries `trace_id`, `stage_id`, `batch_size`, `offset`
- [x] Expose Prometheus counters via `prometheus_client`: `records_consumed_total`, `records_written_total`, `consumer_lag_seconds`, `dlq_events_total`, `write_latency_seconds` (histogram)
- [x] Add `/metrics` HTTP endpoint to the ingestion process so Prometheus can scrape it without a sidecar
- [x] Write test: run ingestion against `testcontainers`, scrape `/metrics`, assert all counters increment correctly after a known batch

---

## Phase 3 — Detection

### Milestone 9 — `SlidingWindowAggregator`: per-stage rolling statistics

Goal: maintain a continuously updated statistical summary of each pipeline stage — the foundation every detector reads from.

- [ ] Implement `RingBuffer` — fixed-size circular buffer backed by `numpy` array; O(1) insert and O(1) eviction of expired entries
- [ ] Build `SlidingWindowAggregator` — maintains per-`stage_id` windows for `latency_ms`, `row_count`, and `error_rate`; computes p50/p95/p99 on each tick
- [ ] Add `WindowConfig` — configurable window duration, tick interval, and minimum sample count before statistics are considered stable
- [ ] Write unit tests: feed known sequences into the aggregator, assert percentile values match `numpy.percentile` reference calculations

---

### Milestone 10 — `SeasonalBaselineModel`: expected-value baselines per stage

Goal: give detectors a per-hour-of-week expected value to compare against so a slow Monday morning doesn't look like an anomaly.

- [ ] Implement `SeasonalBaselineModel` — fits a per-`(stage_id, hour_of_week)` baseline mean and standard deviation from historical `pipeline_metrics` data
- [ ] Add `BaselineFitter` — queries PostgreSQL for the past N days of data, fits baselines, and writes them to a `stage_baselines` table
- [ ] Implement `BaselineStore` — in-memory cache of fitted baselines with TTL-based refresh; avoids hitting PostgreSQL on every detection tick
- [ ] Write tests: fit baselines on synthetic data with a known weekly pattern, assert model recovers injected means within 5%

---

### Milestone 11 — `CUSUMDetector`: catch gradual mean shift

Goal: detect slow, sustained drift in latency or throughput — the failure mode that z-scores miss because the change is too gradual.

- [ ] Implement `CUSUMDetector` — two-sided cumulative sum control chart; maintains `cusum_upper` and `cusum_lower` accumulators per stage
- [ ] Add `cusum_decision_threshold` and `cusum_slack_parameter` (k) as configurable fields; document the trade-off between sensitivity and false-positive rate in the docstring
- [ ] Integrate with `SeasonalBaselineModel` — CUSUM measures deviation from the seasonal expected value, not the global mean
- [ ] Emit `AnomalyEvent` when either accumulator exceeds threshold; include `detector_type`, `stage_id`, `signal`, `cusum_value`, `threshold`
- [ ] Write unit tests: feed step-change and ramp-change sequences, assert CUSUM detects ramp that z-score misses

---

### Milestone 12 — `EWMADetector`: catch sudden spikes

Goal: catch sharp, short-duration spikes that CUSUM's accumulation lag would miss — latency bursts, sudden error floods.

- [ ] Implement `EWMADetector` — exponentially weighted moving average with configurable smoothing parameter λ (0 < λ ≤ 1)
- [ ] Add control limits derived from EWMA variance formula — tighter than 3σ at startup, stabilizes as n grows
- [ ] Ensure EWMA and CUSUM detectors share the same `AnomalyEvent` schema so the event bus treats them identically
- [ ] Write unit tests: feed impulse and step sequences; assert EWMA reacts to impulse within 1 tick and CUSUM does not

---

### Milestone 13 — `AnomalyEventBus`: publish anomalies back to Redpanda

Goal: decouple detection from causal analysis and healing — each layer subscribes independently and can evolve without touching the detector.

- [ ] Implement `AnomalyEventBus` — wraps `RedpandaProducer` for the `pipeline.anomalies` topic; serializes `AnomalyEvent` with schema version
- [ ] Add `AnomalyPersister` — consumes from `pipeline.anomalies` and writes to the `anomaly_events` PostgreSQL table for audit and replay
- [ ] Write integration test: fire both detectors against a faulted event stream, assert all `AnomalyEvent` objects land in Redpanda and PostgreSQL with correct `fault_label` alignment

---

### Milestone 14 — detection accuracy benchmarks

Goal: measure precision and recall against ground-truth fault labels before the causal layer tries to use detector output — garbage in, garbage out.

- [ ] Build `DetectorBenchmark` — runs a full simulated scenario, collects detector output, and computes precision, recall, F1, and mean detection lag per fault type
- [ ] Assert recall ≥ 0.90 and false-positive rate ≤ 0.05 across all six fault types from Milestone 3
- [ ] Write a benchmark report to `docs/detection_benchmark.md` showing per-detector, per-fault-type breakdown
- [ ] Gate CI: fail the benchmark job if either threshold is breached

---

## Phase 4 — Causal Analysis

### Milestone 15 — `CausalDAG`: lift pipeline topology into a causal graph

Goal: transform the structural pipeline graph from Phase 1 into a causal graph that supports do-calculus queries.

- [ ] Build `CausalDAG` — wraps `PipelineTopologyGraph` as a `networkx.DiGraph`; adds edge attribute `propagation_delay_ms` derived from stage config
- [ ] Implement `AncestorResolver` — given a symptomatic stage, returns all causal ancestors ordered by graph distance and propagation delay
- [ ] Add `CausalDAGValidator` — asserts the graph satisfies d-separation conditions required for valid do-calculus; raises on violation
- [ ] Write tests: assert ancestor resolution is correct on a known 5-stage DAG with two branch paths

---

### Milestone 16 — `FaultLocalizationEngine`: rank root causes by posterior probability

Goal: given a set of co-occurring anomalies and the causal graph, output a ranked list of candidate root-cause stages with probability scores.

- [ ] Implement `AnomalyWindowCollector` — groups `AnomalyEvent` objects within a configurable correlation window into a single `FaultHypothesis`
- [ ] Build `FaultLocalizationEngine` — for each `FaultHypothesis`, walks the `CausalDAG` and scores each ancestor stage using a Bayesian update on anomaly timing relative to `propagation_delay_ms`
- [ ] Add `LocalizationResult` dataclass — fields: `hypothesis_id`, `ranked_candidates` (list of `(stage_id, posterior_probability)`), `triggered_at`, `evidence_events`
- [ ] Write tests using ground-truth fault labels from Milestone 3: assert true root cause appears in top-2 candidates in ≥ 85% of scenarios

---

### Milestone 17 — `AlertCorrelator`: suppress redundant anomaly noise

Goal: a single fault in a 10-stage pipeline shouldn't generate 10 alerts — the correlator collapses symptom events into one actionable signal.

- [ ] Implement `AlertCorrelator` — sliding deduplication window; groups `AnomalyEvent` objects by overlapping time range and causal ancestor set
- [ ] Add `CorrelationPolicy` — configurable window duration and minimum co-occurrence count before events are grouped
- [ ] Ensure correlated groups are emitted as a single `CorrelatedAlert` with all constituent event IDs attached for traceability
- [ ] Write tests: feed 10 simultaneous anomaly events from downstream stages of one root cause; assert exactly one `CorrelatedAlert` emitted

---

### Milestone 18 — causal audit table and localization persistence

Goal: every localization decision is written to PostgreSQL so the system can learn from its mistakes and operators can audit automated actions.

- [ ] Write Alembic migration for `fault_localizations` table — columns: `id`, `hypothesis_id`, `root_cause_stage_id`, `posterior_probability`, `evidence_event_ids`, `true_label` (nullable, filled in post-hoc), `created_at`
- [ ] Add `LocalizationRepository` — async writes `LocalizationResult` to `fault_localizations`; exposes `get_by_hypothesis_id` for the healing layer
- [ ] Write integration test: run localization on a seeded scenario, assert persisted row matches in-memory `LocalizationResult`

---

## Phase 5 — Self-Healing

### Milestone 19 — `CircuitBreaker`: per-stage failure isolation

Goal: stop sending work to a failing stage before its failure cascades into upstream backpressure and downstream starvation.

- [ ] Implement `CircuitBreaker` — three-state FSM (closed → open → half-open); trips after `failure_threshold` consecutive failures, resets on exponential backoff
- [ ] Add `CircuitBreakerRegistry` — one breaker instance per `stage_id`; thread-safe; serializes state to PostgreSQL so restarts don't reset breaker history
- [ ] Write unit tests: drive breaker through all state transitions, assert backoff intervals double correctly and half-open probe logic works

---

### Milestone 20 — `HealingPolicyEngine`: map fault types to remediation actions

Goal: given a `LocalizationResult`, select the right remediation action — the engine is the decision layer, not the executor.

- [ ] Define `HealingAction` enum — `CIRCUIT_BREAK`, `RATE_LIMIT`, `REPLAY_RANGE`, `REROUTE_TRAFFIC`, `SCALE_CONSUMER`, `PAGE_OPERATOR`
- [ ] Implement `HealingPolicyEngine` — priority-ordered rule table mapping `(fault_type, severity, stage_type)` to `HealingAction`; falls back to `PAGE_OPERATOR` when no rule matches
- [ ] Add `PolicyConfig` — YAML-driven rule definitions so remediation logic is configurable without code changes
- [ ] Write tests: assert correct action selected for each fault type, and that `PAGE_OPERATOR` fires when no rule matches

---

### Milestone 21 — `ReplayOrchestrator`: replay failed message ranges

Goal: recover from message processing failures by replaying the exact offset range from Redpanda with backpressure to avoid re-flooding a recovering stage.

- [ ] Implement `ReplayOrchestrator` — seeks `MetricConsumer` to a `(topic, partition, start_offset, end_offset)` range and replays with configurable `replay_rate_limit_rps`
- [ ] Add `ReplayRequest` dataclass — created by `HealingPolicyEngine` when `REPLAY_RANGE` is selected; includes the triggering `hypothesis_id` for audit linkage
- [ ] Write integration test: process a batch, mark offsets as failed, trigger replay, assert all records re-land in PostgreSQL with `replayed=true` flag

---

### Milestone 22 — `HealingAuditLog`: record every automated action

Goal: every action the system takes autonomously must be traceable to the anomaly that triggered it, the policy that selected it, and the outcome.

- [ ] Write Alembic migration for `healing_actions` table — columns: `id`, `hypothesis_id`, `localization_id`, `action_type`, `target_stage_id`, `policy_rule_matched`, `outcome` (pending / success / failed), `started_at`, `resolved_at`
- [ ] Implement `HealingAuditLog` — writes action records on start and updates outcome on resolution; never deletes rows
- [ ] Write integration test: trigger a full heal cycle, assert `healing_actions` row transitions from `pending` → `success` with correct timestamps

---

## Phase 6 — API

### Milestone 23 — FastAPI skeleton, config, and health endpoint

Goal: get a running, tested API process before adding any domain routes — foundation first.

- [ ] Scaffold `api/main.py` — FastAPI app with lifespan context manager for database pool and Redpanda client initialization
- [ ] Add `api/config.py` — `Settings` class using `pydantic-settings`; reads `DATABASE_URL`, `REDPANDA_BROKERS`, `LOG_LEVEL` from environment
- [ ] Implement `GET /health` — returns `{"status": "ok", "db": "<connected|error>", "redpanda": "<connected|error>"}` with real connectivity checks
- [ ] Wire `structlog` and OpenTelemetry into the FastAPI middleware stack
- [ ] Write test: assert `/health` returns 200 with real containers up and 503 with database down

---

### Milestone 24 — pipeline status and metric history endpoints

Goal: expose the current and historical state of every pipeline stage so the dashboard and operators have a query surface.

- [ ] `GET /stages` — returns all stages with current circuit breaker state and latest p99 latency from `SlidingWindowAggregator`
- [ ] `GET /stages/{stage_id}/metrics` — paginated time-series query against `pipeline_metrics` with `start`, `end`, and `resolution` query params
- [ ] `GET /stages/{stage_id}/anomalies` — returns anomaly events for a stage, filterable by detector type and time range
- [ ] Write tests for each endpoint: fixture data in PostgreSQL, assert response schema and pagination behavior

---

### Milestone 25 — fault localization and healing history endpoints

Goal: give operators visibility into every automated decision the system has made and is currently executing.

- [ ] `GET /localizations` — paginated list of `LocalizationResult` records with top candidate and posterior probability
- [ ] `GET /localizations/{hypothesis_id}` — full detail including all evidence event IDs and ranked candidates
- [ ] `GET /healing/actions` — paginated list of healing actions with outcome and timestamps
- [ ] `POST /healing/actions/{hypothesis_id}/override` — manually mark a pending action as cancelled; writes to `healing_actions` with `outcome=cancelled` and operator identity
- [ ] Write tests for all endpoints including the override flow

---

### Milestone 26 — API integration test: full request/response cycle with live services

Goal: verify the API works end-to-end with real Redpanda and PostgreSQL before the dashboard is built on top of it.

- [ ] Write `tests/test_api_integration.py` — spins up `testcontainers` for PostgreSQL and Redpanda, runs `SimulatorEngine` to populate data, hits every API endpoint
- [ ] Assert response latency for paginated queries is under 200ms at 100,000-row table size
- [ ] Assert `/health` correctly reflects container state transitions (kill postgres, assert 503, restart, assert 200)

---

## Phase 7 — Observatory Dashboard

### Milestone 27 — Streamlit skeleton and pipeline health overview

Goal: a live page showing every stage's current status — the first thing an operator opens when something is wrong.

- [ ] Scaffold `dashboard/app.py` — Streamlit app with `st.set_page_config` and auto-refresh via `streamlit-autorefresh`
- [ ] Build `PipelineHealthView` — grid of stage cards; each card shows stage name, current status (healthy / degraded / circuit-open), p99 latency, and error rate; data pulled from `GET /stages`
- [ ] Color-code cards by severity: green / amber / red based on deviation from seasonal baseline
- [ ] Write a smoke test: assert the Streamlit app starts without import errors and all API calls are mocked correctly

---

### Milestone 28 — anomaly timeline and causal graph view

Goal: show the sequence of anomalies and their causal relationships so operators can understand fault propagation at a glance.

- [ ] Build `AnomalyTimelineView` — Plotly event timeline showing anomaly events per stage over the last N minutes; hoverable with detector type and signal value
- [ ] Build `CausalGraphView` — renders the `CausalDAG` using `networkx` + Plotly with nodes coloured by anomaly state and edges weighted by propagation delay
- [ ] Highlight the top-ranked root cause stage from the latest `LocalizationResult` in red
- [ ] Wire both views to auto-refresh every 10 seconds

---

### Milestone 29 — healing activity view and manual override UI

Goal: make automated healing actions visible and give operators a one-click escape hatch to cancel them.

- [ ] Build `HealingActivityView` — table of active and recent healing actions with action type, target stage, status, and elapsed time
- [ ] Add `ManualOverridePanel` — dropdown to select a pending action, confirm button that calls `POST /healing/actions/{hypothesis_id}/override`
- [ ] Show a `HealingAuditTrail` expander per action: triggering anomaly → localization → policy rule matched → outcome
- [ ] Write smoke tests for all three views with mocked API responses

---

## Phase 8 — End-to-End Validation

### Milestone 30 — full self-healing integration test

Goal: prove the entire system works as a closed loop — one test that exercises every component from fault injection to confirmed recovery.

- [ ] Write `tests/test_e2e_self_healing.py` — starts all containers, runs `SimulatorEngine` with a known `FaultSpec`, waits for detection, localization, healing action, and pipeline recovery
- [ ] Assert: fault detected within 30 seconds of injection, root cause stage in top-2 candidates, healing action executed, pipeline metrics return to within 10% of baseline within 60 seconds
- [ ] Parameterize over all six fault types from Milestone 3
- [ ] Add this test as a nightly CI job separate from the unit test suite — it's slow and infrastructure-dependent
