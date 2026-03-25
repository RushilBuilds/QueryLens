# QueryLens

QueryLens is a self-healing data pipeline observatory I'm building to detect anomalies in running pipelines, identify root causes using causal inference on the pipeline dependency graph, and automatically remediate failures through circuit breaking, message replay, and rate limiting.

The goal is a system where an operator can look at a dashboard, see exactly which stage caused a failure, watch the system fix it, and have a full audit trail of every decision made along the way.

---

## Motivation

Most pipeline monitoring tools alert you when something is wrong but leave the diagnosis to you. I wanted to build something that closes the loop. When a sink stage starts dropping records, I want the system to walk the DAG backwards, figure out whether the fault originated two hops upstream at the Kafka source, and trigger the right remediation automatically. QueryLens is that system.

---

## Layers

The system is split into independent layers. Each one can be tested in isolation and produces output that feeds the next.

| Layer | Responsibility |
|---|---|
| **Simulator** | Synthetic workload generation and controlled fault injection with ground-truth labels |
| **Ingestion** | Streaming pipeline metrics through Redpanda into PostgreSQL with at-least-once delivery |
| **Detection** | CUSUM and EWMA detectors running against per-stage seasonal baselines |
| **Causal** | Bayesian root-cause ranking over the pipeline DAG using Pearl do-calculus |
| **Healing** | Policy engine that maps fault types to remediation actions and executes them |
| **API** | FastAPI service for pipeline status, anomaly history, localization results, and healing audit |
| **Dashboard** | Streamlit real-time observatory with anomaly timeline, causal graph, and healing activity |

---

## Current build status

| Milestone | Component | Status |
|---|---|---|
| 1 | `PipelineEvent` + `PoissonEventGenerator` | Done |
| 2 | `PipelineTopologyGraph` + `TopologyLoader` | Done |
| 3 | `FaultInjector` + `FaultSchedule` | Done |
| 4 | `SimulatorEngine` + `ScenarioConfig` | Done |
| 5 | PostgreSQL schema + Alembic migrations | Done |
| 6 | `RedpandaProducer` + serialization | Done |
| 7 | `MetricConsumer` + `IngestionWorker` | Done |
| 8 | Structured logging + Prometheus metrics | Done |
| 9 | `SlidingWindowAggregator` | Done |
| 10 | `SeasonalBaselineModel` | Done |
| 11 | `CUSUMDetector` | Complete |
| 12 | `EWMADetector` | Complete |
| 13 | `AnomalyEventBus` + `AnomalyPersister` | Complete |
| 14 | Detection accuracy benchmarks | Complete |
| 15 | `CausalDAG` + `AncestorResolver` + `CausalDAGValidator` | Complete |
| 16 to 18 | Causal analysis layer | Pending |
| 19 to 22 | Self-healing layer | Pending |
| 23 to 26 | API layer | Pending |
| 27 to 29 | Observatory dashboard | Pending |
| 30 | End-to-end integration test | Pending |

---

## Architecture

```
+------------------------------------------------------------------+
|  Simulator                                                        |
|  WorkloadProfile -> PoissonEventGenerator -> PipelineEvent stream |
|                            |                                      |
|                FaultInjector (6 fault types)                      |
|                PipelineTopologyGraph (DAG)                        |
+----------------------------+-------------------------------------+
                             |  PipelineEvent stream
                             v
+------------------------------------------------------------------+
|  Ingestion                                                        |
|  RedpandaProducer -> pipeline.metrics topic                       |
|  MetricConsumer -> IngestionWorker -> PostgreSQL                  |
+----------------------------+-------------------------------------+
                             |  pipeline_metrics table
                             v
+------------------------------------------------------------------+
|  Detection                                                        |
|  SlidingWindowAggregator + SeasonalBaselineModel                  |
|  CUSUMDetector + EWMADetector -> AnomalyEvent                     |
|  AnomalyEventBus -> pipeline.anomalies topic                      |
+----------------------------+-------------------------------------+
                             |  AnomalyEvent stream
                             v
+------------------------------------------------------------------+
|  Causal Analysis                                                  |
|  AlertCorrelator -> FaultHypothesis                               |
|  FaultLocalizationEngine (Bayesian over CausalDAG)                |
|  LocalizationResult with ranked root-cause candidates             |
+----------------------------+-------------------------------------+
                             |  LocalizationResult
                             v
+------------------------------------------------------------------+
|  Self-Healing                                                     |
|  HealingPolicyEngine -> HealingAction                             |
|  CircuitBreaker | ReplayOrchestrator | RateLimiter                |
|  HealingAuditLog -> PostgreSQL                                    |
+----------------------------+-------------------------------------+
                             |
                   +---------+---------+
                   v                   v
             FastAPI               Streamlit
             PipelineAPI       ObservatoryDashboard
```

---

## What has been built so far

### Milestone 1: PipelineEvent model and Poisson workload generator

**`simulator/models.py`**

`PipelineEvent` represents a single unit of work moving through one pipeline stage. I used a plain dataclass instead of Pydantic because these objects are created in a hot loop at high frequency and Pydantic's validation overhead adds up. Validation happens at the ingestion boundary where data is actually untrusted.

```python
@dataclass
class PipelineEvent:
    stage_id: str
    event_time: datetime
    latency_ms: float
    row_count: int
    payload_bytes: int
    status: str
    fault_label: Optional[str]   # set by FaultInjector so detectors have ground truth
```

**`simulator/workload.py`**

`WorkloadProfile` holds the workload parameters as a pure value object with no generation logic. This keeps it cleanly serializable to YAML so ScenarioConfig can reconstruct it without touching generator internals.

`PoissonEventGenerator` takes a profile and produces a stream of `PipelineEvent` objects:

- Inter-arrival times are drawn from `Exponential(1/lambda)`, which is the correct model for a Poisson arrival process
- Payload sizes use a log-normal distribution because real payload sizes are strictly positive and right-skewed. I parameterize by desired mean and std in bytes and convert to log-space internally via moment-matching so callers never have to think about mu and sigma
- All random draws are done in two vectorized numpy calls before the yield loop. At n=10,000 this is about 80x faster than calling numpy once per event
- The generator accepts an injected `np.random.Generator` so the simulator engine can seed it once and guarantee that the same seed always produces the same event stream

```python
profile = WorkloadProfile(
    arrival_rate_lambda=5.0,
    payload_mean_bytes=4096.0,
    payload_std_bytes=1024.0,
    max_concurrency=8,
)
rng = np.random.default_rng(seed=42)
gen = PoissonEventGenerator(profile=profile, stage_id="source_postgres", rng=rng)

for event in gen.generate(n_events=1000, start_time=datetime(2024, 1, 1)):
    print(event.latency_ms, event.payload_bytes)
```

**Tests** in `tests/test_workload.py`: arrival rate converges to lambda within 5% at n=10,000 and payload mean and std land within 10% of configured values. Both tests use a fixed seed so they are deterministic in CI.

---

### Milestone 2: PipelineTopologyGraph and TopologyLoader

**`simulator/topology.py`**

`PipelineStage` defines where a stage sits in the pipeline and how long it takes for a fault originating there to produce a measurable signal in direct downstream stages. That propagation delay is the edge weight the causal engine uses when scoring root-cause candidates.

```python
@dataclass
class PipelineStage:
    stage_id: str
    stage_type: str               # "source" | "transform" | "sink"
    upstream_ids: List[str]
    propagation_delay_ms: float
```

`PipelineTopologyGraph` wraps a `networkx.DiGraph` and enforces two invariants at construction time before any other code can touch the graph:

1. Every `upstream_ids` entry must reference a stage that actually exists
2. The graph must be acyclic. A cycle breaks causal ordering entirely since you cannot define which stage came first.

Both violations raise a `ValueError` with a clear message rather than silently producing a broken graph.

The two traversal methods the causal engine relies on:

- `downstream_stages(stage_id)` returns all stages reachable from a given node, ordered nearest-first by shortest path length
- `ancestors(stage_id)` returns all stages that can causally influence a given node, ordered nearest-first in reverse

I wrapped DiGraph rather than subclassing it to keep the public surface narrow. Exposing the full networkx API to callers would let them mutate the graph and break the acyclicity invariant without going through construction.

`TopologyLoader` reads stage definitions from a YAML file. Pipeline topology should be a config concern, not a code concern. Changing the shape of a test scenario should not require a code change.

**`config/topology_example.yaml`** defines the default two-source topology used in tests and local dev:

```
source_postgres --+
                  +--> transform_validate --> transform_aggregate --> sink_warehouse
source_kafka -----+
```

```python
from simulator.topology import TopologyLoader
from pathlib import Path

graph = TopologyLoader.from_yaml(Path("config/topology_example.yaml"))

# All stages affected by a fault in source_postgres, nearest first
affected = graph.downstream_stages("source_postgres")

# Candidate root causes of an anomaly in sink_warehouse, nearest first
candidates = graph.ancestors("sink_warehouse")
```

**Tests** in `tests/test_topology.py`: cycle rejection, unknown upstream reference rejection, multi-hop downstream path ordering, empty downstream for a sink node, ancestor path ordering, and a full YAML round-trip against the fixture file.

---

### Milestone 3: FaultInjector and FaultSchedule

**`simulator/fault_injection.py`**

`FaultSpec` is a pure value object that describes one fault: which stage to target, when to start (as a second offset from simulation start), how long it lasts, and how severe it is. `magnitude` is intentionally dimensionless — its interpretation is fault-type-specific rather than having a separate parameter class per fault type. For `latency_spike` it is a multiplier; for `dropped_connection` and `error_burst` it is a probability; for `schema_drift` it is a row-loss fraction; for `partition_skew` and `throughput_collapse` it is a scale factor.

`FaultSchedule` couples the simulation start timestamp with the ordered list of specs. Every active-window calculation needs both; keeping them co-located prevents callers from passing a mismatched start time and silently activating faults at the wrong offsets.

`FaultInjector` pre-seeds one `np.random.Generator` per `FaultSpec` at construction time. Seeding inside `inject()` would reset RNG state on every call, turning a 50% drop rate into 0% or 100% depending on a single draw. Pre-seeded stateful generators advance independently per event, producing the correct probabilistic distribution across the full fault window.

The six fault types and their observable signatures:

| Fault type | What changes | Signature detectors measure |
|---|---|---|
| `latency_spike` | `latency_ms × magnitude ± 20% jitter` | p99 latency elevation |
| `dropped_connection` | `status=error`, `row_count=0`, `payload_bytes=0`, `latency_ms × 3` | error rate + throughput collapse |
| `schema_drift` | `row_count × (1 − magnitude)`, `status=schema_error` | mean row count reduction |
| `partition_skew` | `payload_bytes × magnitude`, `latency_ms × magnitude` | payload volume + latency elevation |
| `throughput_collapse` | `row_count ÷ magnitude` (floor 1) | mean row count collapse |
| `error_burst` | `status=error` at rate `magnitude`, data preserved | error rate spike with stable throughput |

`schema_drift` and `error_burst` produce distinct signatures even though both raise error rates: `schema_drift` simultaneously collapses `row_count` while `error_burst` leaves data volumes intact. The causal engine uses this difference to distinguish a schema incompatibility from a downstream processing failure.

`fault_label` is set on every event inside the fault window, including events that probabilistic faults chose not to mutate. The label is ground truth about the window, not the per-event outcome. A detector that misses a fault-window event is a false negative even if that event kept its normal status.

```python
from datetime import datetime
from simulator.fault_injection import FaultSpec, FaultSchedule, FaultInjector

schedule = FaultSchedule(
    simulation_start=datetime(2024, 1, 1),
    fault_specs=[
        FaultSpec(
            fault_type="latency_spike",
            target_stage_id="source_postgres",
            start_offset_s=30.0,
            duration_s=60.0,
            magnitude=5.0,
            seed=42,
        ),
    ],
)
injector = FaultInjector(schedule)

for event in event_stream:
    labelled_event = injector.inject(event)
```

**Tests** in `tests/test_fault_injection.py`: 11 tests covering object identity for fault-free events, fault_label coverage across the full window, cross-stage isolation, unknown fault_type rejection, per-fault statistical signatures (latency p99, error rate, row count reduction, payload inflation), and inclusive boundary behaviour at `start_offset_s`.

---

### Milestone 4: SimulationClock, SimulatorEngine, ScenarioConfig

**`simulator/engine.py`**

`SimulationClock` tracks virtual time as `start_time + tick_count × tick_interval_ms`. Computing each timestamp multiplicatively from a fixed origin rather than accumulating timedeltas prevents floating-point drift — over 100,000 ticks at 1ms each, cumulative addition drifts by 10–50 microseconds depending on the platform; the multiplicative approach stays within 1 microsecond of the exact value.

`SimulatorEngine` pre-generates all events for the full simulation window upfront rather than generating per tick. Calling `PoissonEventGenerator.generate()` once per tick would restart the inter-arrival chain at each tick boundary, introducing artificial gaps at tick edges and breaking the statistical properties of the Poisson process. Pre-generating and filtering to the window preserves continuous inter-arrival statistics across boundaries. Stage RNGs are derived via `np.random.SeedSequence.spawn()` so each stage has an independent RNG stream — adding or removing a stage does not change any other stage's event sequence.

`ScenarioConfig` is a thin YAML loader that resolves the topology path relative to the scenario file's parent directory (not the process working directory), making scenario files portable across machines regardless of where they are invoked from. `build_engine()` constructs a fresh clock and injector on every call to guarantee clean RNG state — the reproducibility contract is: same YAML + same seed → same event stream, every time.

```python
from pathlib import Path
from simulator.engine import ScenarioConfig

config = ScenarioConfig.load(Path("config/scenario_example.yaml"))

# Both runs produce byte-for-byte identical event streams.
events_a = list(config.build_engine().run(n_ticks=120))
events_b = list(config.build_engine().run(n_ticks=120))
assert events_a == events_b
```

**`config/scenario_example.yaml`** — a 120-tick scenario with a 5× latency spike at `source_postgres` from t=30s to t=90s. Used as the fixture for integration tests and as the reference scenario for detection benchmarking in Milestone 14.

**Tests** in `tests/test_engine.py`: 10 tests covering clock initial state, per-tick advancement, no-drift over 100k ticks, multi-stage event generation, chronological ordering, simulation window boundaries, fault label application, YAML loading, identical-seed reproducibility, and seed sensitivity.

---

### Milestone 5: PipelineMetric schema and Alembic migration

**`ingestion/models.py`**

Three SQLAlchemy ORM models declared against the PostgreSQL schema:

`PipelineMetric` maps to the partitioned parent table `pipeline_metrics`. The composite primary key `(id, event_time)` is a PostgreSQL partitioning constraint — every unique or primary key on a range-partitioned table must include all partition key columns. `id` uses `GENERATED ALWAYS AS IDENTITY` so the database owns sequencing rather than relying on application-side UUID generation, which avoids hot-spot contention at high insert rates. `trace_id` is stored as `String(32)` rather than `UUID` because OpenTelemetry trace IDs are 128-bit hex strings — forcing them through Postgres's uuid type would require dash-formatting, adding a serialization step for a field we only ever filter on.

`AnomalyEvent` and `FaultLocalization` are stub models with only identity and timestamp columns. Their full schemas depend on decisions made in the detection and causal layers; declaring those columns now would require a corrective migration once the detection API is finalized.

**`migrations/versions/001_initial_schema.py`**

Uses raw SQL via `op.execute()` rather than `op.create_table()` for `pipeline_metrics` because SQLAlchemy's DDL compiler does not emit `PARTITION BY RANGE` — it has no first-class concept of declarative partitioning at the DDL level. Raw SQL gives full control over the partition declaration and the child `PARTITION OF ... FOR VALUES FROM/TO` clauses.

12 monthly child partitions for 2024 are pre-created (the scenario fixtures use `2024-01-01` as simulation start), plus a `DEFAULT` partition to catch any out-of-range inserts rather than failing them silently. A `(stage_id, event_time)` compound index on the parent table is automatically propagated to all child partitions by PostgreSQL's partitioned index infrastructure. A sparse partial index on `fault_label WHERE fault_label IS NOT NULL` covers ground-truth recall queries without paying maintenance overhead on the majority of null rows.

**`tests/test_migration.py`**

10 tests across two groups. `TestMigrationApplied` verifies schema structure: parent table exists, 13 partitions created (12 monthly + DEFAULT), stub tables exist, compound index exists. `TestPipelineMetricRoundTrip` verifies ORM semantics: full insert/select round-trip, field-level type preservation for `latency_ms` (float precision), nullable `fault_label`, and partition routing (row lands in `pipeline_metrics_2024_01`, not DEFAULT).

All tests run against a real `postgres:16-alpine` container via testcontainers. In-memory or SQLite fixtures are not used — SQLite does not support `PARTITION BY RANGE`, `GENERATED ALWAYS AS IDENTITY`, or partial indexes with `WHERE` clauses, so any fixture that accepted the DDL would be lying about compatibility.

---

### Milestone 6: RedpandaProducer and MetricEventSerializer

**`ingestion/serializer.py`**

`MetricEventSerializer` encodes `PipelineEvent` to UTF-8 JSON bytes and decodes back. Every message carries a `schema_version: 1` field so the consumer can detect format changes without coordinating a simultaneous deploy. The alternative — versioning via Kafka topic name (`pipeline.metrics.v2`) — forces a new consumer group and full replay every time the schema evolves. Field-level versioning lets the consumer decide compatibility independently.

`datetime` fields are encoded as ISO 8601 strings with explicit UTC offset rather than Unix timestamps. Unix timestamps require the reader to know the precision (seconds vs milliseconds vs microseconds), which has caused silent data loss when that assumption drifted between producer and consumer versions.

**`ingestion/producer.py`**

`ProducerHealthCheck` tracks delivery outcomes in thread-safe integer counters via a `threading.Lock`. librdkafka's delivery callbacks run on an internal thread separate from the producer thread — without the lock, the counters would be subject to lost-update races. The `on_delivery(err, msg)` signature is the confluent_kafka delivery callback contract; `failed_delivery_count` and `successful_delivery_count` are the Prometheus-ready properties.

`RedpandaProducer` wraps `confluent_kafka.Producer` with `acks=all`, `retries=5`, `enable.idempotence=True`, and `linger.ms=5`. Idempotence with `acks=all` and retries guarantees exactly-once delivery at the producer level — without idempotence, a retry after a lost ack would produce a duplicate. `linger.ms=5` batches messages that arrive within 5ms, halving round-trip count at 5,000 events/sec with negligible latency cost at detection-window granularity. `poll(0)` is called after every `produce()` so delivery callbacks fire in near-real-time rather than only at `flush()`.

`stage_id` is used as the Kafka message key so all events from the same pipeline stage land on the same partition, preserving per-stage event ordering for the sliding window aggregator in M9.

**`tests/test_producer.py`** — 14 tests across three groups:

- `TestMetricEventSerializer`: round-trip field preservation, nullable fields, schema_version presence, wrong-version rejection, invalid JSON rejection, missing field rejection — all without a broker
- `TestProducerHealthCheck`: delivery counter routing for success and failure callbacks, total count
- `TestRedpandaProducerIntegration`: 1,000 events produced to a `confluentinc/cp-kafka:7.6.0` KRaft container, zero delivery failures, all 1,000 consumed back, all deserialize cleanly, events distributed across multiple partitions confirming key routing

---

### Milestone 7: MetricConsumer and IngestionWorker

**`ingestion/consumer.py`**

`ConsumerConfig` enforces `enable.auto.commit: false` unconditionally. Auto-commit advances the offset on a timer regardless of whether the DB write succeeded — a transient PostgreSQL failure followed by an auto-commit would silently drop the batch on restart. Manual commit is the only mechanism that guarantees the offset advances only after the write confirms.

`MetricConsumer.poll_batch()` polls up to `max_records` messages in a loop: the first poll uses a blocking timeout so the worker can sleep efficiently on a quiet topic; subsequent polls use `timeout=0` to drain whatever is already buffered without adding latency. Messages that fail deserialization are routed to `<topic>.dlq` via a separate low-acks producer immediately, then added to the batch as invalid entries so their offset still advances (preventing infinite replay of a permanently malformed message).

`commit_batch()` stores offsets for all messages — valid and invalid — then issues a single synchronous broker commit. Storing offsets for invalid messages is correct because the DLQ has already received them; replaying them on restart would re-write to the DLQ without recovering anything.

**`ingestion/worker.py`**

`IngestionWorker` drives a dual-trigger flush: flush when `batch_size` records accumulate OR when `flush_interval_s` seconds elapse with pending records. A batch-size-only trigger leaves records stranded on a quiet topic; an interval-only trigger creates an unbounded replay window on crash during a burst. The dual trigger bounds both latency and crash-replay volume.

`run_once()` is the public step method rather than a blocking `run()` loop — tests drive it tick-by-tick without threads. The write → commit ordering is strict: `session.commit()` before `consumer.commit_batch()`. A crash between them replays the batch from the last committed Kafka offset, producing duplicate DB rows that GENERATED ALWAYS AS IDENTITY absorbs with new IDs and that aggregate detectors tolerate.

`_write_to_db()` uses `session.add_all()` which SQLAlchemy batches into a single `executemany()` at flush time — roughly 50× faster than 500 individual INSERTs at the batch sizes we use.

**`tests/test_ingestion_worker.py`** — 6 tests:

- `TestWorkerConfig`: rejects zero batch_size and negative flush_interval at construction time
- `TestIngestionWorkerHappyPath`: 50 events produced → all 50 land in `pipeline_metrics`; probe consumer on same group_id sees no messages after close, confirming offset was committed
- `TestDLQRouting`: 10 valid + 1 malformed interleaved → 10 rows in PostgreSQL, 1 envelope in DLQ topic with correct `source_topic`, `source_partition`, `source_offset`, and `error` fields

Each test class uses its own Kafka topic to prevent cross-test offset contamination within the module-scoped broker.

---

### Milestone 10: SeasonalBaselineModel

**`migrations/versions/002_stage_baselines.py`**

Adds the `stage_baselines` table with a `UNIQUE (stage_id, hour_of_week, metric)` constraint that enables idempotent UPSERT on every fitter run. `hour_of_week` is `SMALLINT` with a `CHECK (0 <= hour_of_week <= 167)` constraint — the narrower type signals at the schema level that this is a bounded enumeration, not a general integer. A `CHECK` on `metric` restricts values to the three we track rather than letting typos create phantom baseline slots silently.

**`detection/baseline.py`**

`_hour_of_week(dt)` converts a datetime to a 0–167 slot using Python's `weekday()` (Monday=0), matching ISO week convention. `BaselineFitter` uses a Python-side cutoff datetime rather than `NOW() - INTERVAL '28 days'` in SQL to avoid `INTERVAL` parameterisation fragility and make the window testable without depending on DB server time. The SELECT uses a single `GROUP BY (stage_id, dow, hod)` pass for all three metrics — one round trip instead of three. PostgreSQL's `EXTRACT(DOW ...)` returns Sunday=0 so the fitter converts to ISO weekday with `(dow + 6) % 7` to match `_hour_of_week`.

`SeasonalBaselineModel.z_score()` uses `event_time` (not wall time) to look up the seasonal slot so historical replays get the correct baseline. Returns `None` when `baseline_std == 0.0` — a degenerate std means either one sample or perfectly constant data; CUSUM/EWMA must skip the update rather than accumulate a nonsensical value.

`BaselineStore` uses `time.monotonic()` for TTL tracking (never goes backwards, immune to NTP jumps and DST transitions). `get_model()` re-fits lazily on first call and when TTL expires — no background thread needed since the detection loop is synchronous and the fit takes <100ms.

**Tests** in `tests/test_baseline.py`: 22 tests across 4 classes — `_hour_of_week` mapping (Mon midnight=0, Sun 23:00=167), `SeasonalBaselineModel` lookup and z-score arithmetic, `BaselineStore` TTL semantics with a hand-rolled stub fitter, and 4 integration tests against a real Postgres container verifying mean recovery within 5%, row persistence, empty-stage handling, and UPSERT idempotency.

---

### Milestone 11: CUSUMDetector

**`detection/anomaly.py`**

`AnomalyEvent` is a single frozen dataclass shared by both CUSUM and EWMA rather than per-detector schemas. The `detector_value` field carries the raw CUSUM accumulator (`S_upper` or `S_lower`) or the EWMA statistic depending on `detector_type` — callers that only compare against `threshold` work identically for both without branching. `signal` is `Literal["upper", "lower"]`, meaningful for both detectors. `extract_metric` is colocated here so adding a new metric (e.g., `payload_bytes`) requires editing one place, not every detector class.

**`detection/cusum.py`**

`CUSUMConfig` exposes `decision_threshold` (h) and `slack_parameter` (k) as independent first-class fields. The optimal values depend on the false-positive budget and the target shift magnitude, not on each other — deriving one from the other would couple two independent tuning knobs.

`CUSUMDetector` implements the tabular (one-sided) CUSUM applied in both directions on seasonal z-scores:

```
S_upper[t] = max(0, S_upper[t-1] + z[t] - k)
S_lower[t] = max(0, S_lower[t-1] - z[t] - k)
```

Fire when `S > h`. Reset only the accumulator that fired — if a latency spike causes `S_upper` to fire while `S_lower` has been accumulating concurrent negative drift (e.g., row_count collapse), resetting both would discard the lower accumulator's independent evidence stream. Skips the accumulator update entirely when the baseline returns `None` (no entry for this stage/slot) rather than treating it as z=0, so "no baseline" is distinguishable from "on target" in accumulator traces.

**Tests** in `tests/test_cusum.py`: 21 tests across 4 classes — `CUSUMConfig` validation, `extract_metric` for all three metrics, accumulator arithmetic (increment, floor, reset-on-fire), step-change detection in both directions, ramp-change detection that a per-event z-score alarm misses, multi-stage accumulator isolation, explicit `reset()`, and silent no-op on missing baseline.

The ramp scenario is the definitive CUSUM value test: latencies 56–65ms (z=0.6–1.5) all fall below the z=2.0 per-event alarm. CUSUM accumulates `Σ(z_i - k)` = 0.1+0.2+...+0.9 = 4.5 > h=4.0 and fires at event 9 of 10 — proving the detector catches gradual sustained drift rather than reacting to any individual spike.

---

### Milestone 12: EWMADetector

**`detection/ewma.py`**

`EWMAConfig` exposes `smoothing` (λ) and `control_limit_width` (L) as independent fields — λ controls detection lag vs. smoothing and L controls false-positive rate. Merging them into a single sensitivity knob would make it impossible to tune the two independently.

`EWMADetector` applies the Lucas & Saccucci (1990) EWMA control chart to seasonal z-scores:

```
Z_t = λ * z_t + (1 - λ) * Z_{t-1},   Z_0 = 0
σ²_t = (λ / (2 - λ)) * [1 - (1 - λ)^(2t)]
UCL_t = L * σ_t,   LCL_t = -L * σ_t
```

The exact-variance formula has a key algebraic property at t=1: σ_1 = λ, so Z_1 = λ*z_1 and the firing condition `|Z_1| > L*λ` reduces to `|z_1| > L`. The first tick is equivalent to a z-score test — EWMA detects a single impulse with |z| > L immediately, while CUSUM (with h=4, k=0.5) needs `S_upper = z - k = 3.5 - 0.5 = 3.0 < 4.0` and does not fire. This is the complementarity that justifies running both detectors in parallel.

After a fire, the EWMA value resets to 0 but step count `n` is preserved so the control limit stays at its mature (wider) value during re-arm. Resetting `n` too would collapse the limits back to startup tightness and cause spurious fires on the first post-reset event.

**Tests** in `tests/test_ewma.py`: 21 tests across 5 classes — `EWMAConfig` validation, EWMA arithmetic (value after 1 and 2 steps, the σ_1=λ identity, missing-baseline no-op), impulse detection (upward and downward in one tick, confirmed CUSUM non-redundancy, sub-threshold non-fire), no-fire under on-target events, reset semantics (value reset, n preserved, explicit full reset), and multi-stage state isolation.

---

### Milestone 9: SlidingWindowAggregator

**`detection/window.py`**

`RingBuffer` is a fixed-capacity circular buffer backed by two parallel numpy float64 arrays — one for timestamps, one for values. Writes go to `_data[_head % capacity]` and advance the head pointer, overwriting the oldest entry when full. `window_values(cutoff_s)` builds an ordered index array from head/count and returns a masked view of values whose timestamps are >= cutoff. The numpy array avoids the list conversion that a `collections.deque` would require on every `compute()` call — at 1,000 samples per stage the deque approach would add ~40µs of allocation overhead per tick.

`WindowConfig` holds window_duration_s, tick_interval_s, min_sample_count, and ring_buffer_capacity as a frozen dataclass so all three detectors (CUSUM, EWMA, aggregator) share the same config object and cannot silently diverge on these values.

`SlidingWindowAggregator` maintains one `RingBuffer` per `(stage_id, metric_name)` pair. Three metrics are tracked: `latency_ms` and `row_count` from the event directly; `error_rate` is derived at update time as `0.0 if status == "ok" else 1.0` so the detection layer never needs to depend on the raw status string. `compute(stage_id, now)` returns a `WindowStats` with p50/p95/p99/mean using `np.percentile` with linear interpolation — the same method Prometheus uses for histogram quantile estimation, making the values directly comparable. All stat fields are `None` when `sample_count < min_sample_count`, which forces callers to gate on `is_stable` rather than silently consuming zeros as real measurements.

**Tests** in `tests/test_window.py`: 19 tests covering validation, empty buffer edge cases, wrap-around at capacity, cutoff boundary semantics (>= not >), multi-stage isolation, error rate derivation, and percentile match against numpy reference.

---

### Milestone 8: structured logging and Prometheus /metrics

**`ingestion/observability.py`**

All five Prometheus metrics are defined as module-level singletons: `ingestion_records_consumed_total` and `ingestion_records_written_total` (Counters, labelled by `stage_id`), `ingestion_dlq_events_total` (Counter, no label — we don't have `stage_id` for messages that failed to deserialise), `ingestion_write_latency_seconds` (Histogram with sub-10ms buckets tuned for our target p99), and `ingestion_consumer_lag_seconds` (Gauge, labelled by `stage_id`, set to `wall_time - oldest_event_time` per flush). Module-level singletons rather than per-instance because `prometheus_client` raises on duplicate registration — construction inside a class would break the second time a `MetricConsumer` or `IngestionWorker` is instantiated in tests.

`MetricsServer` wraps `wsgiref.simple_server` rather than `prometheus_client.start_http_server` because the return type of `start_http_server` changed between 0.16 and 0.20, making portable `shutdown()` calls impossible. `wsgiref` is stdlib and gives clean lifecycle control via `serve_forever` / `shutdown`. `port=0` lets the OS assign a free port — essential for tests running in parallel.

`configure_structlog()` sets up JSON-rendered structured logging via `PrintLoggerFactory` (stdout). JSON output makes `trace_id` and `stage_id` filterable dimensions in any log aggregator without a custom regex.

**`ingestion/consumer.py`** and **`ingestion/worker.py`**

`MetricConsumer` logs `poll_batch_complete` (batch_size, valid, dlq, first/last offset) and `dlq_routed` (partition, offset, error[:200]) and increments `DLQ_EVENTS` in `_send_to_dlq`. `IngestionWorker` increments `RECORDS_CONSUMED` and `RECORDS_WRITTEN` per stage in `_flush`, observes `WRITE_LATENCY` in `_write_to_db`, and updates `CONSUMER_LAG` per stage before each flush using `_oldest_event_time_per_stage`.

**Tests** in `tests/test_ingestion_observability.py`: 5 tests covering server startup, text-format metric presence, per-stage counter deltas, histogram observation, and DLQ counter increment — all asserting deltas rather than absolute values so they compose safely with other tests in the session.

---

## Repository layout

```
QueryLens/
├── simulator/
│   ├── models.py           PipelineEvent dataclass
│   ├── topology.py         PipelineStage, PipelineTopologyGraph, TopologyLoader
│   ├── workload.py         WorkloadProfile, PoissonEventGenerator
│   ├── fault_injection.py  FaultSpec, FaultSchedule, FaultInjector
│   └── engine.py           SimulationClock, SimulatorEngine, ScenarioConfig
├── ingestion/
│   ├── models.py           PipelineMetric, AnomalyEvent, FaultLocalization ORM models
│   ├── serializer.py       MetricEventSerializer — JSON wire format with schema_version
│   ├── producer.py         RedpandaProducer, ProducerHealthCheck
│   ├── consumer.py         MetricConsumer, ConsumerConfig, DLQ routing
│   ├── worker.py           IngestionWorker (dual-trigger batch flush)
│   └── __init__.py
├── migrations/
│   ├── env.py              Alembic env — injects DATABASE_URL from environment
│   ├── script.py.mako      migration template
│   └── versions/
│       └── 001_initial_schema.py   partitioned pipeline_metrics + stub tables
├── detection/              Milestones 9 to 14
├── causal/                 Milestones 15 to 18
├── healing/                Milestones 19 to 22
├── api/                    Milestones 23 to 26
├── dashboard/              Milestones 27 to 29
├── tests/
│   ├── test_workload.py         Milestone 1 tests
│   ├── test_topology.py         Milestone 2 tests
│   ├── test_fault_injection.py  Milestone 3 tests
│   ├── test_engine.py           Milestone 4 integration tests
│   ├── test_migration.py        Milestone 5 migration smoke test (requires Docker)
│   ├── test_producer.py         Milestone 6 producer integration test (requires Docker)
│   └── test_ingestion_worker.py Milestone 7 consumer/worker integration test (requires Docker)
├── config/
│   ├── topology_example.yaml
│   └── scenario_example.yaml
├── docker-compose.yml
├── requirements.txt
├── ROADMAP.md
├── DECISIONS.md
└── CLAUDE.md
```

---

## Infrastructure

Everything runs locally with `docker compose up`:

| Service | Image | Port | Purpose |
|---|---|---|---|
| Redpanda | `redpandadata/redpanda:v23.3.6` | 9092 | Kafka-compatible broker without ZooKeeper |
| Redpanda Console | `redpandadata/console:v2.4.3` | 8080 | Web UI for inspecting topics and consumer groups |
| PostgreSQL | `postgres:16-alpine` | 5432 | Pipeline metrics, anomaly events, healing audit log |
| QueryLens API | local build | 8000 | FastAPI service (Milestone 23 onwards) |

I chose Redpanda over Kafka because it is fully Kafka-API compatible, runs as a single binary, and eliminates ZooKeeper entirely. For a local development environment this means one fewer service to manage and a cold start time of around 2 seconds instead of 30. See ADR-001 in DECISIONS.md.

---

## Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All tests are fully deterministic. They use seeded RNGs and explicit start timestamps so there is no dependency on wall-clock time or environment state.

---

## Architecture decisions

Every non-obvious technical choice is documented in [DECISIONS.md](DECISIONS.md) with the alternative that was considered and the reason this option was better for this use case.
