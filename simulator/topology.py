from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import networkx as nx
import yaml


@dataclass
class PipelineStage:
    """
    Pure value object — no back-reference to the graph. TopologyLoader can construct
    stages from YAML before the graph exists, and the graph can be rebuilt from stages
    without mutating them.

    propagation_delay_ms is the expected time for a fault at this stage to surface in
    direct downstream stages. The causal engine uses these as edge weights when scoring
    root-cause candidates.
    """

    stage_id: str
    stage_type: str                    # "source" | "transform" | "sink"
    upstream_ids: List[str] = field(default_factory=list)
    propagation_delay_ms: float = 0.0


class PipelineTopologyGraph:
    """
    Wraps networkx.DiGraph rather than subclassing — exposes only the domain-relevant
    surface (downstream resolution, ancestor queries) and prevents callers from
    bypassing the acyclicity invariant enforced at construction time.

    Acyclicity is validated once at construction because topology is treated as
    immutable after loading. Runtime topology changes should construct a new graph.
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
                # Edge direction is upstream → downstream so nx.descendants() from a node
                # returns all stages that depend on it, matching causal direction.
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
        Returns descendants ordered by shortest topological path length from stage_id.
        The causal engine needs this ordering to prioritise nearby descendants when
        scoring anomaly propagation.
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
        Returns ancestors sorted by reverse path length — nearest upstream first.
        The causal engine queries in this direction to find root-cause candidates.
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
    Loads topology from YAML so scenario configs can define different pipeline shapes
    without touching code. Stateless class (all class methods) — purely a parsing
    boundary.
    """

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineTopologyGraph:
        """
        Validates that each required field exists before constructing PipelineStage
        objects so that missing fields raise a clear KeyError at load time rather than
        an AttributeError deep inside graph construction.
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
