from pathlib import Path

import pytest

from simulator.topology import PipelineStage, PipelineTopologyGraph, TopologyLoader

# Path to the shared fixture topology used across simulator tests.
FIXTURE_TOPOLOGY_PATH = Path(__file__).parent.parent / "config" / "topology_example.yaml"


def _build_linear_graph() -> PipelineTopologyGraph:
    """
    I'm factoring out the linear 3-stage graph into a helper because multiple tests
    need it and duplicating the stage definitions would make refactoring the fixture
    painful. It's not a pytest fixture because it returns a value, not a generator,
    and the construction is cheap enough that sharing state across tests buys nothing.
    """
    stages = [
        PipelineStage(stage_id="src", stage_type="source", upstream_ids=[], propagation_delay_ms=0.0),
        PipelineStage(stage_id="xfm", stage_type="transform", upstream_ids=["src"], propagation_delay_ms=25.0),
        PipelineStage(stage_id="snk", stage_type="sink", upstream_ids=["xfm"], propagation_delay_ms=15.0),
    ]
    return PipelineTopologyGraph(stages)


def test_dag_rejects_cycle() -> None:
    """
    I'm testing the two-node cycle (a → b → a) rather than a longer one because it's
    the minimal case that proves the invariant. If networkx.is_directed_acyclic_graph
    correctly rejects the minimal cycle it will reject longer ones — there's no need
    to enumerate them.
    """
    stages = [
        PipelineStage(stage_id="stage_a", stage_type="source", upstream_ids=["stage_b"]),
        PipelineStage(stage_id="stage_b", stage_type="sink", upstream_ids=["stage_a"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        PipelineTopologyGraph(stages)


def test_dag_rejects_unknown_upstream_reference() -> None:
    """
    I'm testing this separately from the cycle test because it's a different failure
    mode — a typo in upstream_ids that silently creates a disconnected graph would be
    worse than a crash, since the causal engine would treat stages as independent when
    they're not.
    """
    stages = [
        PipelineStage(stage_id="stage_a", stage_type="source", upstream_ids=["nonexistent_stage"]),
    ]
    with pytest.raises(ValueError, match="unknown upstream"):
        PipelineTopologyGraph(stages)


def test_downstream_stages_resolves_multihop_path() -> None:
    """
    I'm asserting both the content and order of downstream_stages because the causal
    engine depends on proximity ordering — nearest descendants first. Asserting only
    content would let a sorting bug through that would silently degrade localization
    accuracy without failing any test.
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
    I'm explicitly testing the sink boundary case because the causal engine calls
    downstream_stages on every anomalous stage, including sinks. A KeyError or a
    non-empty list from a sink would produce phantom causal candidates.
    """
    graph = _build_linear_graph()
    assert graph.downstream_stages("snk") == []


def test_ancestors_resolves_multihop_path() -> None:
    """
    I'm testing ancestors in addition to downstream_stages because both directions
    are used by the causal engine: downstream to check fault propagation, upstream
    to find root-cause candidates. A graph that gets one direction right and the
    other wrong would pass half the tests and fail silently in production.
    """
    graph = _build_linear_graph()

    ancestor_ids = [s.stage_id for s in graph.ancestors("snk")]
    assert ancestor_ids == ["xfm", "src"], (
        f"Expected ['xfm', 'src'] ordered by reverse path length, got {ancestor_ids}"
    )


def test_topology_loader_reads_fixture_yaml() -> None:
    """
    I'm loading the shared fixture YAML rather than an inline string because the
    fixture file is also used by SimulatorEngine integration tests — this test
    verifies that the file on disk is valid and parseable, catching schema drift
    between the loader and the YAML format before it breaks a scenario run.
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
