from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from simulator.topology import PipelineStage, PipelineTopologyGraph


@dataclass(frozen=True)
class CausalAncestor:
    """
    Pairs the PipelineStage with graph distance and cumulative propagation delay
    because the FaultLocalizationEngine needs both: distance indicates fault origin
    proximity; cumulative_delay_ms gives the expected time for a fault to surface
    as a symptom. A low-distance ancestor with high delay is weaker than one with
    both low distance and low delay when observed timing matches.
    """

    stage: PipelineStage
    graph_distance: int            # hop count from this stage to the symptomatic stage
    cumulative_delay_ms: float     # sum of propagation_delay_ms along the shortest-delay path


class CausalDAG:
    """
    Wraps PipelineTopologyGraph rather than extending it because CausalDAG adds
    do-calculus-specific behaviour (ancestor scoring by delay, d-separation validation)
    that has no place in the structural topology layer. The two layers have different
    reasons to change: topology changes on pipeline reconfiguration; causal semantics
    change when the fault propagation model evolves.

    The underlying nx.DiGraph is extracted once at construction time — the graph is
    immutable after construction, so a direct reference is safe and avoids per-query
    topology API overhead.
    """

    def __init__(self, topology: PipelineTopologyGraph) -> None:
        self._topology = topology
        self._graph: nx.DiGraph = topology._graph

    # ------------------------------------------------------------------
    # Ancestor resolution
    # ------------------------------------------------------------------

    def causal_ancestors(self, stage_id: str) -> List[CausalAncestor]:
        """
        Sorts ancestors by (graph_distance, cumulative_delay_ms): a node two hops away
        with 5ms total delay is causally closer than one hop away with 200ms delay.
        Primary key preserves the intuition that closer ancestors are more likely root
        causes; secondary key breaks ties using propagation physics.

        Returns an empty list for source stages.
        """
        if stage_id not in self._graph:
            raise KeyError(f"Unknown stage_id '{stage_id}'")

        ancestor_ids = nx.ancestors(self._graph, stage_id)
        if not ancestor_ids:
            return []

        # Dijkstra with propagation_delay_ms weight selects the minimum-delay path —
        # the path along which a fault would first manifest as a symptom.
        try:
            delays = nx.single_target_shortest_path_length(
                self._graph, stage_id
            )
        except nx.NetworkXError:
            delays = {}

        result: List[CausalAncestor] = []
        for ancestor_id in ancestor_ids:
            stage = self._topology.get_stage(ancestor_id)
            hop_distance = delays.get(ancestor_id, -1)
            cumulative_delay = self._cumulative_delay(ancestor_id, stage_id)
            result.append(CausalAncestor(
                stage=stage,
                graph_distance=hop_distance,
                cumulative_delay_ms=cumulative_delay,
            ))

        return sorted(result, key=lambda a: (a.graph_distance, a.cumulative_delay_ms))

    def _cumulative_delay(self, from_id: str, to_id: str) -> float:
        """
        Cumulative delay along the minimum-total-delay path rather than the minimum-hop
        path: a two-hop path via a fast branch appears before a one-hop path via a slow
        branch — fault propagation follows physics, not hop count.

        Returns 0.0 if from_id and to_id are the same node.
        """
        if from_id == to_id:
            return 0.0
        try:
            path = nx.shortest_path(
                self._graph,
                source=from_id,
                target=to_id,
                weight="propagation_delay_ms",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

        return sum(
            self._graph[path[i]][path[i + 1]].get("propagation_delay_ms", 0.0)
            for i in range(len(path) - 1)
        )

    def edge_delay_ms(self, from_id: str, to_id: str) -> float:
        """Returns the propagation_delay_ms on the direct edge from_id → to_id."""
        if not self._graph.has_edge(from_id, to_id):
            raise KeyError(f"No direct edge from '{from_id}' to '{to_id}'")
        return self._graph[from_id][to_id].get("propagation_delay_ms", 0.0)

    def has_edge(self, from_id: str, to_id: str) -> bool:
        return self._graph.has_edge(from_id, to_id)

    @property
    def stage_ids(self) -> List[str]:
        return list(self._graph.nodes())

    @property
    def topology(self) -> PipelineTopologyGraph:
        return self._topology


class AncestorResolver:
    """
    Separate from CausalDAG because the resolver accumulates query-time state
    (per-stage cache) that does not belong on a structural object. The DAG should not
    hold query results; the resolver is a stateful service injected as a collaborator.

    Per-stage caching avoids repeated Dijkstra runs: a burst of 100 anomalies for
    the same stage would otherwise repeat the same computation 100 times.
    """

    def __init__(self, dag: CausalDAG) -> None:
        self._dag = dag
        self._cache: Dict[str, List[CausalAncestor]] = {}

    def resolve(self, symptomatic_stage_id: str) -> List[CausalAncestor]:
        """
        Returns all causal ancestors of symptomatic_stage_id, sorted by
        (graph_distance, cumulative_delay_ms) ascending — closest, fastest
        ancestors first.
        """
        if symptomatic_stage_id not in self._cache:
            self._cache[symptomatic_stage_id] = self._dag.causal_ancestors(
                symptomatic_stage_id
            )
        return self._cache[symptomatic_stage_id]

    def closest_ancestor(
        self, symptomatic_stage_id: str
    ) -> Optional[CausalAncestor]:
        """
        Returns the closest ancestor by (distance, delay), or None for source
        stages with no upstream. This is the starting candidate for the
        FaultLocalizationEngine's Bayesian root-cause ranking — not necessarily
        the final answer, but the highest prior.
        """
        ancestors = self.resolve(symptomatic_stage_id)
        return ancestors[0] if ancestors else None

    def invalidate(self, stage_id: str) -> None:
        """Evict a cached resolution — call after topology changes in tests."""
        self._cache.pop(stage_id, None)


class CausalDAGValidator:
    """
    Structural checks rather than probabilistic d-separation tests because the pipeline
    is fully observed — every stage emits PipelineEvents, so there are no latent
    confounders. The checks catch misconfigurations that would make causal queries
    return silently wrong answers:

    - Type consistency: a 'source' stage with in-edges would have its anomaly incorrectly
      attributed to upstream stages that do not exist in the model.
    - Non-negative delays: negative delays reverse causal ordering in the Bayesian update.
    - Reachability: isolated subgraphs have faults that cannot propagate to observed sinks,
      making them unobservable confounders that invalidate the do-calculus adjustment.
    - Acyclicity: guaranteed by PipelineTopologyGraph but re-checked as defence in depth.
    """

    def __init__(self, dag: CausalDAG) -> None:
        self._dag = dag

    def validate(self) -> None:
        """
        Runs all structural checks in sequence, raising ValueError on the first violation.
        A misconfigured topology is not safe to use at all — partial validation success
        does not mean causal queries will be correct.
        """
        self._check_acyclicity()
        self._check_stage_type_consistency()
        self._check_non_negative_delays()
        self._check_reachability()

    def _check_acyclicity(self) -> None:
        if not nx.is_directed_acyclic_graph(self._dag._graph):
            cycle = nx.find_cycle(self._dag._graph)
            raise ValueError(
                f"CausalDAG contains a cycle: {cycle}. "
                "Causal queries require a DAG — cycles make do-calculus undefined."
            )

    def _check_stage_type_consistency(self) -> None:
        """
        Verifies stage_type labels match graph structure. The FaultLocalizationEngine
        uses stage_type to assign priors; a miscategorised stage would silently skew
        the Bayesian ranking toward the wrong candidates.
        """
        graph = self._dag._graph
        topology = self._dag.topology
        for stage_id in graph.nodes():
            stage = topology.get_stage(stage_id)
            in_deg = graph.in_degree(stage_id)
            out_deg = graph.out_degree(stage_id)

            if stage.stage_type == "source" and in_deg > 0:
                raise ValueError(
                    f"Stage '{stage_id}' is typed as 'source' but has {in_deg} "
                    "upstream edge(s). Sources must have no upstream edges."
                )
            if stage.stage_type == "sink" and out_deg > 0:
                raise ValueError(
                    f"Stage '{stage_id}' is typed as 'sink' but has {out_deg} "
                    "downstream edge(s). Sinks must have no downstream edges."
                )
            if stage.stage_type == "transform" and in_deg == 0 and out_deg == 0:
                raise ValueError(
                    f"Stage '{stage_id}' is typed as 'transform' but has no edges. "
                    "Transform stages must have at least one upstream or downstream edge."
                )

    def _check_non_negative_delays(self) -> None:
        graph = self._dag._graph
        for from_id, to_id, attrs in graph.edges(data=True):
            delay = attrs.get("propagation_delay_ms", 0.0)
            if delay < 0.0:
                raise ValueError(
                    f"Edge '{from_id}' → '{to_id}' has negative propagation_delay_ms={delay}. "
                    "Negative delays are physically impossible and reverse causal ordering."
                )

    def _check_reachability(self) -> None:
        """
        Every stage must either be a source or have a path from at least one source.
        A stage that fails this check is in an isolated subgraph whose anomalies cannot
        be causally connected to the rest of the pipeline model.
        """
        graph = self._dag._graph
        topology = self._dag.topology

        source_ids = {
            s.stage_id for s in topology.all_stages if s.stage_type == "source"
        }
        sink_ids = {
            s.stage_id for s in topology.all_stages if s.stage_type == "sink"
        }

        if not source_ids:
            raise ValueError(
                "CausalDAG has no source stages. At least one 'source' stage "
                "is required for causal queries to be anchored."
            )

        if not sink_ids:
            raise ValueError(
                "CausalDAG has no sink stages. At least one 'sink' stage "
                "is required to validate end-to-end observability."
            )

        # Every non-source stage must be reachable from some source.
        reachable_from_sources: set = set()
        for src_id in source_ids:
            reachable_from_sources |= nx.descendants(graph, src_id)
            reachable_from_sources.add(src_id)

        for stage_id in graph.nodes():
            if stage_id not in reachable_from_sources:
                raise ValueError(
                    f"Stage '{stage_id}' is not reachable from any source stage. "
                    "Isolated stages create unobservable confounders — add an upstream "
                    "connection or reclassify as 'source'."
                )
