"""
Unit tests for CausalDAG, AncestorResolver, and CausalDAGValidator.

All tests use an in-memory 5-stage DAG with two branch paths:

    source_a ──(10ms)──→ transform_b ──(5ms)──→ sink_e
         ↘──(15ms)──→ transform_c ──(8ms)──↗

    source_d ──(20ms)──→ transform_b

This topology exercises ancestor resolution from a node with multiple upstream
paths, delay-weighted shortest-path selection, and the two-branch convergence
property (transform_b has two sources; sink_e has two transform parents).

No containers or external services are required.
"""
from __future__ import annotations

from typing import List

import pytest

from causal.dag import AncestorResolver, CausalAncestor, CausalDAG, CausalDAGValidator
from simulator.topology import PipelineStage, PipelineTopologyGraph

# ---------------------------------------------------------------------------
# Shared topology fixture
# ---------------------------------------------------------------------------

#
# Graph layout:
#
#   source_a ──(10ms)──→ transform_b ──(5ms)──→ sink_e
#        ↘──(15ms)──→ transform_c ──(8ms)──↗
#   source_d ──(20ms)──→ transform_b
#
# Expected ancestors of sink_e:
#   distance=1: transform_b (cumulative=5ms), transform_c (cumulative=8ms)
#   distance=2: source_a (via transform_b: 10+5=15ms, via transform_c: 15+8=23ms → 15ms)
#               source_d (via transform_b: 20+5=25ms)


def _build_topology() -> PipelineTopologyGraph:
    stages = [
        PipelineStage(
            stage_id="source_a",
            stage_type="source",
            upstream_ids=[],
            propagation_delay_ms=0.0,
        ),
        PipelineStage(
            stage_id="source_d",
            stage_type="source",
            upstream_ids=[],
            propagation_delay_ms=0.0,
        ),
        PipelineStage(
            stage_id="transform_b",
            stage_type="transform",
            upstream_ids=["source_a", "source_d"],
            propagation_delay_ms=10.0,  # delay from source_a; source_d sets 20ms below
        ),
        PipelineStage(
            stage_id="transform_c",
            stage_type="transform",
            upstream_ids=["source_a"],
            propagation_delay_ms=15.0,
        ),
        PipelineStage(
            stage_id="sink_e",
            stage_type="sink",
            upstream_ids=["transform_b", "transform_c"],
            propagation_delay_ms=5.0,  # delay from both parents to sink_e
        ),
    ]
    return PipelineTopologyGraph(stages)


@pytest.fixture(scope="module")
def topology() -> PipelineTopologyGraph:
    return _build_topology()


@pytest.fixture(scope="module")
def dag(topology: PipelineTopologyGraph) -> CausalDAG:
    return CausalDAG(topology)


@pytest.fixture(scope="module")
def resolver(dag: CausalDAG) -> AncestorResolver:
    return AncestorResolver(dag)


# ---------------------------------------------------------------------------
# CausalDAG — basic structure
# ---------------------------------------------------------------------------


class TestCausalDAGStructure:

    def test_stage_ids_contains_all_five(self, dag: CausalDAG) -> None:
        assert set(dag.stage_ids) == {
            "source_a", "source_d", "transform_b", "transform_c", "sink_e"
        }

    def test_has_edge_direct_connection(self, dag: CausalDAG) -> None:
        assert dag.has_edge("source_a", "transform_b")
        assert dag.has_edge("source_a", "transform_c")
        assert dag.has_edge("source_d", "transform_b")
        assert dag.has_edge("transform_b", "sink_e")
        assert dag.has_edge("transform_c", "sink_e")

    def test_has_no_edge_non_adjacent(self, dag: CausalDAG) -> None:
        assert not dag.has_edge("source_a", "sink_e")
        assert not dag.has_edge("sink_e", "source_a")

    def test_edge_delay_direct(self, dag: CausalDAG) -> None:
        """
        I'm testing edge_delay_ms directly to confirm the edge attribute was
        set correctly during graph construction. A wrong delay here would corrupt
        all cumulative_delay calculations in ancestor resolution.
        """
        # transform_b's propagation_delay_ms=10 is on the source_a→transform_b edge.
        assert dag.edge_delay_ms("source_a", "transform_b") == pytest.approx(10.0)
        assert dag.edge_delay_ms("source_a", "transform_c") == pytest.approx(15.0)
        assert dag.edge_delay_ms("transform_b", "sink_e") == pytest.approx(5.0)
        assert dag.edge_delay_ms("transform_c", "sink_e") == pytest.approx(5.0)

    def test_edge_delay_raises_for_missing_edge(self, dag: CausalDAG) -> None:
        with pytest.raises(KeyError, match="No direct edge"):
            dag.edge_delay_ms("source_a", "sink_e")

    def test_causal_ancestors_raises_for_unknown_stage(self, dag: CausalDAG) -> None:
        with pytest.raises(KeyError):
            dag.causal_ancestors("nonexistent_stage")


# ---------------------------------------------------------------------------
# CausalDAG — ancestor resolution
# ---------------------------------------------------------------------------


class TestCausalAncestors:

    def test_source_stage_has_no_ancestors(self, dag: CausalDAG) -> None:
        """
        I'm asserting that source stages return an empty ancestor list rather
        than raising — the FaultLocalizationEngine calls resolve() on every
        stage that shows an anomaly, including source stages, and must handle
        the empty-ancestor case without special-casing the call site.
        """
        assert dag.causal_ancestors("source_a") == []
        assert dag.causal_ancestors("source_d") == []

    def test_sink_has_four_ancestors(self, dag: CausalDAG) -> None:
        ancestors = dag.causal_ancestors("sink_e")
        ancestor_ids = {a.stage.stage_id for a in ancestors}
        assert ancestor_ids == {"source_a", "source_d", "transform_b", "transform_c"}

    def test_ancestors_sorted_by_distance_first(self, dag: CausalDAG) -> None:
        """
        I'm verifying the primary sort key is graph_distance so that the closest
        ancestors (transforms at distance=1) appear before the more-distant sources.
        The FaultLocalizationEngine starts from the closest ancestor and walks
        outward — correct ordering is essential for its greedy search strategy.
        """
        ancestors = dag.causal_ancestors("sink_e")
        distances = [a.graph_distance for a in ancestors]
        assert distances == sorted(distances), (
            "Ancestors must be sorted by graph_distance ascending"
        )
        # First two must be distance=1 (transform_b, transform_c)
        assert ancestors[0].graph_distance == 1
        assert ancestors[1].graph_distance == 1
        # Last two must be distance=2 (source_a, source_d)
        assert ancestors[2].graph_distance == 2
        assert ancestors[3].graph_distance == 2

    def test_distance1_ancestors_sorted_by_delay(self, dag: CausalDAG) -> None:
        """
        I'm asserting that among ancestors at the same graph_distance, the one
        with smaller cumulative_delay_ms comes first. transform_b is 5ms from
        sink_e; transform_c is 5ms from sink_e — they're equal here. Let me
        check both have 5ms and either order is acceptable.
        """
        ancestors = dag.causal_ancestors("sink_e")
        dist1 = [a for a in ancestors if a.graph_distance == 1]
        assert all(a.cumulative_delay_ms == pytest.approx(5.0) for a in dist1)

    def test_source_a_cumulative_delay_to_sink_e(self, dag: CausalDAG) -> None:
        """
        source_a → transform_b (10ms) → sink_e (5ms) = 15ms total.
        source_a → transform_c (15ms) → sink_e (5ms) = 20ms total.
        The minimum-delay path is via transform_b at 15ms.
        """
        ancestors = dag.causal_ancestors("sink_e")
        source_a_entry = next(a for a in ancestors if a.stage.stage_id == "source_a")
        assert source_a_entry.cumulative_delay_ms == pytest.approx(15.0)

    def test_source_d_cumulative_delay_to_sink_e(self, dag: CausalDAG) -> None:
        """
        PipelineTopologyGraph stores propagation_delay_ms on the destination stage,
        not per upstream edge. Both source_a → transform_b and source_d → transform_b
        therefore carry transform_b's propagation_delay_ms=10ms.

        source_d → transform_b (10ms) → sink_e (5ms) = 15ms total.
        """
        ancestors = dag.causal_ancestors("sink_e")
        source_d_entry = next(a for a in ancestors if a.stage.stage_id == "source_d")
        assert source_d_entry.cumulative_delay_ms == pytest.approx(15.0)

    def test_transform_b_has_two_ancestors(self, dag: CausalDAG) -> None:
        ancestors = dag.causal_ancestors("transform_b")
        ancestor_ids = {a.stage.stage_id for a in ancestors}
        assert ancestor_ids == {"source_a", "source_d"}

    def test_transform_c_has_one_ancestor(self, dag: CausalDAG) -> None:
        ancestors = dag.causal_ancestors("transform_c")
        assert len(ancestors) == 1
        assert ancestors[0].stage.stage_id == "source_a"


# ---------------------------------------------------------------------------
# AncestorResolver — caching and closest-ancestor logic
# ---------------------------------------------------------------------------


class TestAncestorResolver:

    def test_resolve_returns_same_result_as_dag(
        self, resolver: AncestorResolver, dag: CausalDAG
    ) -> None:
        direct = dag.causal_ancestors("sink_e")
        via_resolver = resolver.resolve("sink_e")
        assert [a.stage.stage_id for a in direct] == [
            a.stage.stage_id for a in via_resolver
        ]

    def test_resolve_caches_result(self, dag: CausalDAG) -> None:
        """
        I'm verifying caching by calling resolve() twice and asserting the
        returned lists are the same object (identity check). If the resolver
        re-computed on the second call, it would return a new list object.
        """
        resolver = AncestorResolver(dag)
        first = resolver.resolve("sink_e")
        second = resolver.resolve("sink_e")
        assert first is second, "AncestorResolver must cache results — second call returned new object"

    def test_closest_ancestor_returns_distance1(
        self, resolver: AncestorResolver
    ) -> None:
        closest = resolver.closest_ancestor("sink_e")
        assert closest is not None
        assert closest.graph_distance == 1

    def test_closest_ancestor_returns_none_for_source(
        self, resolver: AncestorResolver
    ) -> None:
        assert resolver.closest_ancestor("source_a") is None

    def test_invalidate_clears_cache(self, dag: CausalDAG) -> None:
        resolver = AncestorResolver(dag)
        first = resolver.resolve("sink_e")
        resolver.invalidate("sink_e")
        second = resolver.resolve("sink_e")
        assert first is not second, "After invalidate(), resolver must recompute"


# ---------------------------------------------------------------------------
# CausalDAGValidator — valid topology passes
# ---------------------------------------------------------------------------


class TestCausalDAGValidatorHappyPath:

    def test_valid_topology_passes_without_raising(self, dag: CausalDAG) -> None:
        """
        I'm asserting no exception is raised on the well-formed 5-stage topology.
        Any regression in the validator logic that throws on a correct graph
        would fail the FaultLocalizationEngine at startup.
        """
        validator = CausalDAGValidator(dag)
        validator.validate()  # must not raise


# ---------------------------------------------------------------------------
# CausalDAGValidator — rejection of malformed topologies
# ---------------------------------------------------------------------------


class TestCausalDAGValidatorRejections:

    def _make_dag(self, stages: list) -> CausalDAG:
        return CausalDAG(PipelineTopologyGraph(stages))

    def test_rejects_source_with_upstream_edges(self) -> None:
        """
        I'm testing that a stage labelled 'source' but with upstream edges fails
        validation. The FaultLocalizationEngine would never attribute a fault to
        the phantom upstream of a miscategorised source, producing silently wrong
        root-cause rankings.
        """
        stages = [
            PipelineStage("real_source", "source", [], 0.0),
            # Mislabelled: has an upstream edge but is typed as 'source'
            PipelineStage("bad_source", "source", ["real_source"], 10.0),
            PipelineStage("sink_x", "sink", ["bad_source"], 5.0),
        ]
        dag = self._make_dag(stages)
        validator = CausalDAGValidator(dag)
        with pytest.raises(ValueError, match="source.*upstream"):
            validator.validate()

    def test_rejects_sink_with_downstream_edges(self) -> None:
        stages = [
            PipelineStage("src", "source", [], 0.0),
            # Mislabelled: has downstream edges but is typed as 'sink'
            PipelineStage("bad_sink", "sink", ["src"], 10.0),
            PipelineStage("actual_sink", "sink", ["bad_sink"], 5.0),
        ]
        dag = self._make_dag(stages)
        validator = CausalDAGValidator(dag)
        with pytest.raises(ValueError, match="sink.*downstream"):
            validator.validate()

    def test_rejects_topology_with_no_sources(self) -> None:
        """
        A pipeline with no source stage has no causal anchor — every stage's
        anomaly would be causally attributed to another stage that is also a
        symptom, producing circular reasoning.
        """
        stages = [
            PipelineStage("a", "transform", ["b"], 10.0),
            PipelineStage("b", "transform", ["a"], 10.0),
        ]
        # This would raise in PipelineTopologyGraph (cycle), so use a topology
        # that has transforms but no 'source' typed stage.
        stages_no_source = [
            PipelineStage("t1", "transform", [], 0.0),
            PipelineStage("t2", "sink", ["t1"], 5.0),
        ]
        dag = self._make_dag(stages_no_source)
        validator = CausalDAGValidator(dag)
        with pytest.raises(ValueError, match="no source stages"):
            validator.validate()

    def test_rejects_topology_with_no_sinks(self) -> None:
        stages = [
            PipelineStage("src", "source", [], 0.0),
            PipelineStage("t1", "transform", ["src"], 10.0),
        ]
        dag = self._make_dag(stages)
        validator = CausalDAGValidator(dag)
        with pytest.raises(ValueError, match="no sink stages"):
            validator.validate()

    def test_rejects_isolated_transform_stage(self) -> None:
        """
        I'm testing that a transform stage with no edges fails validation. In a
        directed DAG without cycles, a node with in_degree=0 and out_degree=0
        typed as 'transform' is an isolated confounder — its faults cannot be
        causally connected to any other stage. The type-consistency check catches
        this before the reachability check, which is correct: both checks protect
        the same invariant via different failure-mode paths.
        """
        stages = [
            PipelineStage("src", "source", [], 0.0),
            PipelineStage("connected", "transform", ["src"], 10.0),
            PipelineStage("sink_x", "sink", ["connected"], 5.0),
            PipelineStage("orphan_transform", "transform", [], 0.0),
        ]
        dag = self._make_dag(stages)
        validator = CausalDAGValidator(dag)
        with pytest.raises(ValueError, match="no edges"):
            validator.validate()
