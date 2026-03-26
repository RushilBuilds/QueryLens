from pathlib import Path

import pytest

from simulator.topology import PipelineStage, PipelineTopologyGraph, TopologyLoader

# Path to the shared fixture topology used across simulator tests.
FIXTURE_TOPOLOGY_PATH = Path(__file__).parent.parent / "config" / "topology_example.yaml"


def _build_linear_graph() -> PipelineTopologyGraph:
    """
    Shared linear 3-stage graph factored out to avoid duplicating stage definitions.
    A plain function rather than a pytest fixture because construction is cheap and
    sharing state across tests buys nothing.
    """
    stages = [
        PipelineStage(stage_id="src", stage_type="source", upstream_ids=[], propagation_delay_ms=0.0),
        PipelineStage(stage_id="xfm", stage_type="transform", upstream_ids=["src"], propagation_delay_ms=25.0),
        PipelineStage(stage_id="snk", stage_type="sink", upstream_ids=["xfm"], propagation_delay_ms=15.0),
    ]
    return PipelineTopologyGraph(stages)


def test_dag_rejects_cycle() -> None:
    """
    Two-node cycle (a → b → a) is the minimal case. If networkx.is_directed_acyclic_graph
    correctly rejects the minimal cycle it will reject all longer ones.
    """
    stages = [
        PipelineStage(stage_id="stage_a", stage_type="source", upstream_ids=["stage_b"]),
        PipelineStage(stage_id="stage_b", stage_type="sink", upstream_ids=["stage_a"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        PipelineTopologyGraph(stages)


def test_dag_rejects_unknown_upstream_reference() -> None:
    """
    Separate from the cycle test: a different failure mode. A typo in upstream_ids
    that silently creates a disconnected graph would be worse than a crash — the causal
    engine would treat linked stages as independent.
    """
    stages = [
        PipelineStage(stage_id="stage_a", stage_type="source", upstream_ids=["nonexistent_stage"]),
    ]
    with pytest.raises(ValueError, match="unknown upstream"):
        PipelineTopologyGraph(stages)


def test_downstream_stages_resolves_multihop_path() -> None:
    """
    Both content and order asserted: the causal engine depends on proximity ordering
    (nearest descendants first). Asserting only content would let a sorting bug
    through that silently degrades localization accuracy.
    """
    graph = _build_linear_graph()

    downstream = graph.downstream_stages("src")
    downstream_ids = [s.stage_id for s in downstream]

    # "xfm" is one hop from "src", "snk" is two hops — must come in that order.
    assert downstream_ids == ["xfm", "snk"], (
        f"Expected ['xfm', 'snk'] ordered by path length, got {downstream_ids}"
    )


def test_downstream_stages_returns_empty_for_sink() -> None:
    """
    Sink boundary case: the causal engine calls downstream_stages on every anomalous
    stage including sinks. A KeyError or non-empty result from a sink would produce
    phantom causal candidates.
    """
    graph = _build_linear_graph()
    assert graph.downstream_stages("snk") == []


def test_ancestors_resolves_multihop_path() -> None:
    """
    Both directions tested: the causal engine uses downstream for fault propagation
    and upstream for root-cause candidates. A graph that gets one direction right
    and the other wrong would pass half the tests and fail silently in production.
    """
    graph = _build_linear_graph()

    ancestor_ids = [s.stage_id for s in graph.ancestors("snk")]
    assert ancestor_ids == ["xfm", "src"], (
        f"Expected ['xfm', 'src'] ordered by reverse path length, got {ancestor_ids}"
    )


def test_topology_loader_reads_fixture_yaml() -> None:
    """
    Shared fixture YAML loaded rather than an inline string: the file is also used
    by SimulatorEngine integration tests, so this catches schema drift between the
    loader and the YAML format before it breaks a scenario run.
    """
    graph = TopologyLoader.from_yaml(FIXTURE_TOPOLOGY_PATH)

    stage_ids = {s.stage_id for s in graph.all_stages}
    assert stage_ids == {
        "source_postgres",
        "source_kafka",
        "transform_validate",
        "transform_aggregate",
        "sink_warehouse",
    }

    # Both sources feed into transform_validate — verify both appear as ancestors.
    validate_ancestors = {s.stage_id for s in graph.ancestors("transform_validate")}
    assert validate_ancestors == {"source_postgres", "source_kafka"}

    # sink_warehouse is three hops from either source — verify full downstream chain.
    postgres_downstream = [s.stage_id for s in graph.downstream_stages("source_postgres")]
    assert "sink_warehouse" in postgres_downstream
    assert postgres_downstream.index("transform_validate") < postgres_downstream.index("sink_warehouse")
