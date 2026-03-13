from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import networkx as nx
import yaml


@dataclass
class PipelineStage:
    """
    I'm keeping PipelineStage as a pure value object — no references back to the graph
    that contains it. This lets TopologyLoader construct stages from YAML before the
    graph exists, and lets the graph be rebuilt from stages without mutating the stages
    themselves. Circular references between stage and graph would make serialisation
    and testing significantly messier.

    propagation_delay_ms is the expected time for a fault originating at this stage to
    produce a measurable signal in direct downstream stages. The causal engine uses
    these delays as edge weights when scoring root-cause candidates — a stage that
    showed an anomaly 50ms before its downstream neighbour is a stronger candidate
    than one that showed it 500ms before, given a 60ms propagation delay.
    """

    stage_id: str
    stage_type: str                    # "source" | "transform" | "sink"
    upstream_ids: List[str] = field(default_factory=list)
    propagation_delay_ms: float = 0.0


class PipelineTopologyGraph:
    """
    I'm wrapping networkx.DiGraph rather than subclassing it because I only want to
    expose the domain-relevant surface (downstream resolution, ancestor queries) and
    not leak networkx's full API to callers. Subclassing DiGraph would mean callers
    could mutate the graph directly and bypass the acyclicity invariant I enforce at
    construction time.

    Acyclicity is validated once at construction rather than on every mutation because
    topology is treated as immutable after loading — there's no add_stage() method.
    If topology needs to change at runtime (not a current requirement), the right
    pattern is to construct a new graph, not to mutate an existing one.
    """

    def __init__(self, stages: List[PipelineStage]) -> None:
        self._stages: dict[str, PipelineStage] = {s.stage_id: s for s in stages}
        self._graph: nx.DiGraph = nx.DiGraph()

        for stage in stages:
            self._graph.add_node(stage.stage_id)

        for stage in stages:
            for upstream_id in stage.upstream_ids:
                if upstream_id not in self._stages:
                    raise ValueError(
                        f"Stage '{stage.stage_id}' references unknown upstream "
                        f"'{upstream_id}' — check your topology config"
                    )
                # Edge direction is upstream → downstream so that nx.descendants()
                # from a node returns all stages that depend on it, matching the
                # causal direction: a fault propagates downstream, not upstream.
                self._graph.add_edge(
                    upstream_id,
                    stage.stage_id,
                    propagation_delay_ms=stage.propagation_delay_ms,
                )

        if not nx.is_directed_acyclic_graph(self._graph):
            cycle = nx.find_cycle(self._graph)
            raise ValueError(
                f"Pipeline topology contains a cycle: {cycle}. "
                f"Cycles break causal ordering — every stage must have a clear upstream."
            )

    def downstream_stages(self, stage_id: str) -> List[PipelineStage]:
        """
        I'm returning descendants ordered by shortest topological path length from
        stage_id rather than in arbitrary networkx iteration order. The causal engine
        needs this ordering to prioritise nearby descendants when scoring anomaly
        propagation — a stage two hops away is a weaker causal witness than one
        directly connected.
        """
        if stage_id not in self._stages:
            raise KeyError(f"Unknown stage_id '{stage_id}'")

        descendant_ids = nx.descendants(self._graph, stage_id)
        lengths = nx.single_source_shortest_path_length(self._graph, stage_id)

        return sorted(
            [self._stages[sid] for sid in descendant_ids],
            key=lambda s: lengths[s.stage_id],
        )

    def ancestors(self, stage_id: str) -> List[PipelineStage]:
        """
        I'm exposing ancestors separately from downstream_stages because the causal
        engine queries in the opposite direction: given a symptomatic stage, find
        candidate root-cause stages upstream. Keeping both directions explicit avoids
        callers having to reason about edge direction in the underlying digraph.
        """
        if stage_id not in self._stages:
            raise KeyError(f"Unknown stage_id '{stage_id}'")

        ancestor_ids = nx.ancestors(self._graph, stage_id)
        lengths = nx.single_source_shortest_path_length(
            self._graph.reverse(copy=False), stage_id
        )

        return sorted(
            [self._stages[sid] for sid in ancestor_ids],
            key=lambda s: lengths[s.stage_id],
        )

    def get_stage(self, stage_id: str) -> PipelineStage:
        if stage_id not in self._stages:
            raise KeyError(f"Unknown stage_id '{stage_id}'")
        return self._stages[stage_id]

    @property
    def all_stages(self) -> List[PipelineStage]:
        return list(self._stages.values())


class TopologyLoader:
    """
    I'm loading topology from YAML rather than Python so that scenario configs can
    define different pipeline shapes without touching code. The alternative — hardcoding
    topologies as Python objects — would mean every new test scenario requires a code
    change and a commit, which breaks the reproducibility goal of ScenarioConfig.

    The loader is a stateless class (all class methods) because it holds no mutable
    state between loads — it's purely a parsing boundary.
    """

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineTopologyGraph:
        """
        I'm validating that each required field exists before constructing PipelineStage
        objects so that missing fields raise a clear KeyError at load time rather than
        surfacing as an AttributeError deep inside graph construction.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        stages = []
        for entry in raw["stages"]:
            stages.append(
                PipelineStage(
                    stage_id=entry["stage_id"],
                    stage_type=entry["stage_type"],
                    upstream_ids=entry.get("upstream_ids", []),
                    propagation_delay_ms=float(entry.get("propagation_delay_ms", 0.0)),
                )
            )

        return PipelineTopologyGraph(stages)
