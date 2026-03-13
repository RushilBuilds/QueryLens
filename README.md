# QueryLens

A self-healing data pipeline observatory. QueryLens detects anomalies in running data pipelines, localises the root cause using causal inference on the pipeline's dependency graph, and executes automated remediation actions — circuit breaking, message replay, rate limiting — without human intervention.

---

## What it does

Most pipeline monitoring systems tell you that something broke. QueryLens tells you which stage caused it, why it propagated the way it did, and what it did to fix it — all logged and auditable.

The system is built in layers that can be run and tested independently:

| Layer | What it does |
|---|---|
| **Simulator** | Generates realistic synthetic pipeline workloads and injects controlled faults with ground-truth labels |
| **Ingestion** | Streams pipeline metrics through Redpanda, persists to PostgreSQL with at-least-once delivery |
| **Detection** | CUSUM and EWMA detectors identify anomalies against seasonal baselines |
| **Causal** | Pearl do-calculus over the pipeline DAG ranks candidate root-cause stages by posterior probability |
| **Healing** | Policy engine maps fault types to remediation actions; circuit breaker, replay orchestrator execute them |
| **API** | FastAPI service exposes pipeline status, anomaly history, localization results, and healing actions |
| **Dashboard** | Streamlit observatory with live health view, anomaly timeline, causal graph, and healing activity |

---

## Project status

| Milestone | Component | Status |
|---|---|---|
| 1 | `PipelineEvent` model + `PoissonEventGenerator` | Done |
| 2 | `PipelineTopologyGraph` + `TopologyLoader` | Done |
| 3 | `FaultInjector` + `FaultSchedule` | Pending |
| 4 | `SimulatorEngine` + `ScenarioConfig` | Pending |
| 5 | PostgreSQL schema + Alembic migrations | Pending |
| 6 | `RedpandaProducer` + serialization | Pending |
| 7 | `MetricConsumer` + `IngestionWorker` | Pending |
| 8 | Structured logging + Prometheus metrics | Pending |
| 9 | `SlidingWindowAggregator` | Pending |
| 10 | `SeasonalBaselineModel` | Pending |
| 11 | `CUSUMDetector` | Pending |
| 12 | `EWMADetector` | Pending |
| 13 | `AnomalyEventBus` | Pending |
| 14 | Detection accuracy benchmarks | Pending |
| 15–18 | Causal analysis layer | Pending |
| 19–22 | Self-healing layer | Pending |
| 23–26 | API layer | Pending |
| 27–29 | Observatory dashboard | Pending |
| 30 | End-to-end integration test | Pending |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Simulator                                                       │
│  WorkloadProfile → PoissonEventGenerator → PipelineEvent stream │
│                           ↓                                      │
│               FaultInjector (6 fault types)                      │
│               PipelineTopologyGraph (DAG)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ PipelineEvent stream
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Ingestion                                                       │
│  RedpandaProducer → pipeline.metrics topic                       │
│  MetricConsumer → IngestionWorker → PostgreSQL                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ pipeline_metrics table
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Detection                                                       │
│  SlidingWindowAggregator + SeasonalBaselineModel                 │
│  CUSUMDetector + EWMADetector → AnomalyEvent                     │
│  AnomalyEventBus → pipeline.anomalies topic                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ AnomalyEvent stream
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Causal Analysis                                                 │
│  AlertCorrelator → FaultHypothesis                               │
│  FaultLocalizationEngine (Bayesian over CausalDAG)               │
│  → LocalizationResult with ranked root-cause candidates          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ LocalizationResult
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Self-Healing                                                    │
│  HealingPolicyEngine → HealingAction                             │
│  CircuitBreaker | ReplayOrchestrator | RateLimiter               │
│  HealingAuditLog → PostgreSQL                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────┴────────┐
                    ▼                ▼
              FastAPI            Streamlit
              PipelineAPI     ObservatoryDashboard
```

---

## What's been built

### Milestone 1 — `PipelineEvent` and `PoissonEventGenerator`

**`simulator/models.py`**

`PipelineEvent` is a flat dataclass representing a single unit of observable work passing through one pipeline stage. Every field maps to a metric a real data pipeline would emit.

```python
@dataclass
class PipelineEvent:
    stage_id: str
    event_time: datetime
    latency_ms: float
    row_count: int
    payload_bytes: int
    status: str
    fault_label: Optional[str]   # written by FaultInjector for ground-truth evaluation
```

**`simulator/workload.py`**

`WorkloadProfile` is a pure value object — serializable to YAML, no generation state. `PoissonEventGenerator` uses it to produce a stream of `PipelineEvent` objects.

- Inter-arrival times drawn from `Exponential(1/λ)` — correct model for a Poisson process
- Payload sizes drawn from a log-normal distribution parameterised by desired mean and std, converted internally via moment-matching so callers think in bytes, not log-space
- All draws are vectorised via numpy before the yield loop — ~80x faster than per-event calls at n=10,000
- Accepts an injected `np.random.Generator` so `SimulatorEngine` can seed once and guarantee replay reproducibility

```python
profile = WorkloadProfile(
    arrival_rate_lambda=5.0,      # 5 events/second
    payload_mean_bytes=4096.0,
    payload_std_bytes=1024.0,
    max_concurrency=8,
)
rng = np.random.default_rng(seed=42)
gen = PoissonEventGenerator(profile=profile, stage_id="source_postgres", rng=rng)

for event in gen.generate(n_events=1000, start_time=datetime(2024, 1, 1)):
    print(event.latency_ms, event.payload_bytes)
```

**Tests** (`tests/test_workload.py`): arrival rate converges to λ within 5% at n=10,000; payload mean and std land within 10% of configured values.

---

### Milestone 2 — `PipelineTopologyGraph` and `TopologyLoader`

**`simulator/topology.py`**

`PipelineStage` defines a stage's position in the pipeline and its expected fault propagation delay to direct downstream stages.

```python
@dataclass
class PipelineStage:
    stage_id: str
    stage_type: str               # "source" | "transform" | "sink"
    upstream_ids: List[str]
    propagation_delay_ms: float   # time for a fault here to show up downstream
```

`PipelineTopologyGraph` wraps a `networkx.DiGraph` and enforces two invariants at construction:

1. All `upstream_ids` reference stages that exist in the graph
2. The graph is acyclic — cycles break causal ordering and raise a `ValueError` immediately

It exposes two traversal methods used by the causal engine:

- `downstream_stages(stage_id)` — all stages reachable from a given stage, ordered by shortest path length (nearest first)
- `ancestors(stage_id)` — all stages that can causally influence a given stage, ordered by reverse path length (nearest first)

`TopologyLoader` reads stage definitions from a YAML file so pipeline shapes can be changed without touching code.

**`config/topology_example.yaml`**

```
source_postgres ──┐
                  ├──► transform_validate ──► transform_aggregate ──► sink_warehouse
source_kafka ─────┘
```

```python
from simulator.topology import TopologyLoader
from pathlib import Path

graph = TopologyLoader.from_yaml(Path("config/topology_example.yaml"))

# Who does a fault in source_postgres affect?
affected = graph.downstream_stages("source_postgres")
# → [transform_validate, transform_aggregate, sink_warehouse]

# What are the candidate root causes of an anomaly in sink_warehouse?
candidates = graph.ancestors("sink_warehouse")
# → [transform_aggregate, transform_validate, source_postgres, source_kafka]
```

**Tests** (`tests/test_topology.py`): cycle rejection, unknown upstream reference rejection, multi-hop downstream ordering, sink boundary case (empty downstream), ancestor resolution ordering, full YAML round-trip against the fixture topology.

---

## Repository layout

```
QueryLens/
├── simulator/
│   ├── models.py          # PipelineEvent dataclass
│   ├── topology.py        # PipelineStage, PipelineTopologyGraph, TopologyLoader
│   └── workload.py        # WorkloadProfile, PoissonEventGenerator
├── ingestion/             # Milestone 6–8 (Redpanda producer, consumer, worker)
├── detection/             # Milestone 9–14 (CUSUM, EWMA, sliding window)
├── causal/                # Milestone 15–18 (DAG, localization, correlator)
├── healing/               # Milestone 19–22 (circuit breaker, replay, policy engine)
├── api/                   # Milestone 23–26 (FastAPI service)
├── dashboard/             # Milestone 27–29 (Streamlit observatory)
├── tests/
│   ├── test_workload.py   # Milestone 1 tests
│   └── test_topology.py   # Milestone 2 tests
├── config/
│   └── topology_example.yaml
├── docker-compose.yml     # Redpanda, PostgreSQL, API
├── requirements.txt
├── ROADMAP.md
├── DECISIONS.md           # Architecture decision records
└── CLAUDE.md              # Engineering standards for this project
```

---

## Infrastructure

Spun up with a single `docker compose up`:

| Service | Image | Port | Purpose |
|---|---|---|---|
| Redpanda | `redpandadata/redpanda:v23.3.6` | 9092 | Kafka-compatible broker (no ZooKeeper) |
| Redpanda Console | `redpandadata/console:v2.4.3` | 8080 | Web UI for topic and consumer group inspection |
| PostgreSQL | `postgres:16-alpine` | 5432 | Pipeline metrics, anomaly events, healing audit log |
| QueryLens API | local build | 8000 | FastAPI service (Milestone 23+) |

Redpanda was chosen over a full Kafka + ZooKeeper stack — same producer/consumer API, single binary, ~2s cold start. See `DECISIONS.md` ADR-001.

---

## Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All tests are deterministic — seeded RNGs, explicit timestamps, no wall-clock dependency.

---

## Architecture decisions

All non-obvious choices are logged in [DECISIONS.md](DECISIONS.md) with the alternative considered and the reason this option was better for this use case.
