from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple


class InfoType(str, Enum):
    DATA = "data"
    DECISION = "decision"
    STATUS = "status"
    QUERY = "query"


class LayerCategory(str, Enum):
    INWARD_OPERATIONAL = "inward_operational"
    OUTWARD_OPERATIONAL = "outward_operational"
    MANAGEMENT = "management"
    CROSS_CUTTING = "cross_cutting"


class ComponentCategory(str, Enum):
    INWARD = "inward"
    OUTWARD = "outward"
    SHARED = "shared"


@dataclass(frozen=True)
class Layer:
    identifier: str
    position: int
    category: LayerCategory
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Component:
    identifier: str
    layer_id: str
    category: ComponentCategory
    info_in: Set[InfoType]
    info_out: Set[InfoType]
    resilience_required: bool = False
    usage_cost: float = 1.0
    processing_time: float = 1.0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeAnnotation:
    flow_weight: Optional[int] = None
    urgency: Optional[bool] = None


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    similarity: float
    layer_alignment: float
    flow_weight: int
    urgency: bool
    weight: float


@dataclass
class ComponentMetrics:
    out_degree: Dict[str, int]
    in_degree: Dict[str, int]
    weighted_out_degree: Dict[str, float]
    weighted_in_degree: Dict[str, float]
    centrality: Dict[str, float]


def jaccard_similarity(a: Set[InfoType], b: Set[InfoType]) -> float:
    if not a or not b:
        return 0.0
    intersection = a.intersection(b)
    if not intersection:
        return 0.0
    union = a.union(b)
    return len(intersection) / len(union)


class DigitalTwinPlatform:
    """
    Container for layers, components, dependencies, and candidate interconnections.
    Provides utilities to derive edge weights and per-component metrics used by optimizers.
    """

    def __init__(
        self,
        theta_1: float = 1.0,
        theta_2: float = 1.0,
        theta_3: float = 0.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
    ) -> None:
        self.layers: Dict[str, Layer] = {}
        self.components: Dict[str, Component] = {}
        self.layer_dependencies: Set[Tuple[str, str]] = set()
        self.edge_annotations: Dict[Tuple[str, str], EdgeAnnotation] = {}
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def add_layer(self, layer: Layer) -> None:
        self.layers[layer.identifier] = layer

    def add_component(self, component: Component) -> None:
        if component.layer_id not in self.layers:
            raise ValueError(f"Layer {component.layer_id} not registered")
        self.components[component.identifier] = component

    def add_layer_dependency(self, depender: str, dependee: str) -> None:
        if depender not in self.layers or dependee not in self.layers:
            raise ValueError("Layer dependency references unknown layer")
        self.layer_dependencies.add((depender, dependee))

    def annotate_edge(
        self, source: str, target: str, flow_weight: Optional[int] = None, urgency: Optional[bool] = None
    ) -> None:
        if source not in self.components or target not in self.components:
            raise ValueError("Edge annotation references unknown component")
        self.edge_annotations[(source, target)] = EdgeAnnotation(flow_weight=flow_weight, urgency=urgency)

    def _layer_alignment(self, src_layer: Layer, dst_layer: Layer) -> float:
        positional_alignment = 1.0 / (1.0 + abs(src_layer.position - dst_layer.position))
        category_bonus = 0.5 if src_layer.category == dst_layer.category else 0.0
        dependency_bonus = 0.0
        if (src_layer.identifier, dst_layer.identifier) in self.layer_dependencies:
            dependency_bonus = 0.75
        elif (dst_layer.identifier, src_layer.identifier) in self.layer_dependencies:
            dependency_bonus = 0.5
        return max(0.0, positional_alignment + category_bonus + dependency_bonus)

    def _flow_weight(self, similarity: float, override: Optional[int]) -> int:
        if override is not None:
            return max(1, min(3, override))
        if similarity > 0.75:
            return 3
        if similarity > 0.35:
            return 2
        return 1

    def _urgency(self, source: Component, target: Component, override: Optional[bool]) -> bool:
        if override is not None:
            return override
        return source.resilience_required or target.resilience_required

    def _base_edge_weight(self, similarity: float, alignment: float) -> float:
        return self.theta_1 * similarity + self.theta_3 * alignment

    def _compute_component_metrics(self, edges: Iterable[Edge]) -> ComponentMetrics:
        out_degree = {cid: 0 for cid in self.components}
        in_degree = {cid: 0 for cid in self.components}
        weighted_out_degree = {cid: 0.0 for cid in self.components}
        weighted_in_degree = {cid: 0.0 for cid in self.components}
        centrality = {cid: 0.0 for cid in self.components}
        for edge in edges:
            out_degree[edge.source] += 1
            in_degree[edge.target] += 1
            weighted_out_degree[edge.source] += edge.weight
            weighted_in_degree[edge.target] += edge.weight
        for cid in self.components:
            centrality[cid] = (
                self.alpha * out_degree[cid]
                + self.beta * in_degree[cid]
                + self.gamma * (weighted_out_degree[cid] + weighted_in_degree[cid])
            )
        return ComponentMetrics(out_degree, in_degree, weighted_out_degree, weighted_in_degree, centrality)

    def build_candidate_edges(self) -> Tuple[List[Edge], ComponentMetrics]:
        raw_edges: List[Edge] = []
        for source in self.components.values():
            for target in self.components.values():
                if source.identifier == target.identifier:
                    continue
                similarity = jaccard_similarity(source.info_out, target.info_in)
                if similarity == 0.0:
                    continue
                src_layer = self.layers[source.layer_id]
                dst_layer = self.layers[target.layer_id]
                alignment = self._layer_alignment(src_layer, dst_layer)
                annotation = self.edge_annotations.get((source.identifier, target.identifier))
                flow_weight = self._flow_weight(similarity, annotation.flow_weight if annotation else None)
                urgency = self._urgency(source, target, annotation.urgency if annotation else None)
                base_weight = self._base_edge_weight(similarity, alignment)
                raw_edges.append(
                    Edge(
                        source=source.identifier,
                        target=target.identifier,
                        similarity=similarity,
                        layer_alignment=alignment,
                        flow_weight=flow_weight,
                        urgency=urgency,
                        weight=base_weight,
                    )
                )
        metrics = self._compute_component_metrics(raw_edges)
        enriched_edges: List[Edge] = []
        for edge in raw_edges:
            updated_weight = edge.weight + self.theta_2 * (
                metrics.centrality[edge.source] + metrics.centrality[edge.target]
            )
            enriched_edges.append(replace(edge, weight=updated_weight))
        return enriched_edges, metrics
