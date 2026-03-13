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
| 15 to 18 | Causal analysis layer | Pending |
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

## Repository layout

```
QueryLens/
├── simulator/
│   ├── models.py           PipelineEvent dataclass
│   ├── topology.py         PipelineStage, PipelineTopologyGraph, TopologyLoader
│   └── workload.py         WorkloadProfile, PoissonEventGenerator
├── ingestion/              Milestones 6 to 8
├── detection/              Milestones 9 to 14
├── causal/                 Milestones 15 to 18
├── healing/                Milestones 19 to 22
├── api/                    Milestones 23 to 26
├── dashboard/              Milestones 27 to 29
├── tests/
│   ├── test_workload.py    Milestone 1 tests
│   └── test_topology.py    Milestone 2 tests
├── config/
│   └── topology_example.yaml
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
