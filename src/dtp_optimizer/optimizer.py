from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from .models import Component, DigitalTwinPlatform, Edge


@dataclass
class OptimizationConfig:
    lambda_edge_count: float = 1.0
    lambda_resilience_penalty: float = 5.0
    lambda_usage_cost: float = 0.5
    lambda_processing_time: float = 0.5
    lambda_flow_weight: float = 1.0
    lambda_urgency_weight: float = 1.5
    min_degree: int = 2
    enforce_connectivity: bool = True


@dataclass
class OptimizationResult:
    selected_edges: Set[Edge]
    component_degrees: Dict[str, Dict[str, int]]
    objective_value: float
    feasibility_report: Dict[str, bool]
    iterations: int = 0


class UnionFind:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node: str) -> str:
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_b < rank_a:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

    def connected_components(self) -> int:
        roots = {self.find(node) for node in self.parent}
        return len(roots)


class EdgeSelectionOptimizer:
    """
    Greedy-plus-pruning optimizer that enforces resilience, urgency, and connectivity
    constraints before locally improving the objective value.
    """

    def __init__(
        self,
        platform: DigitalTwinPlatform,
        config: OptimizationConfig | None = None,
    ) -> None:
        self.platform = platform
        self.config = config or OptimizationConfig()
        self.candidate_edges, self.metrics = self.platform.build_candidate_edges()
        if not self.candidate_edges:
            raise ValueError("No feasible edges discovered for the platform definition")
        self.components = platform.components
        self.mandatory_edges = {
            edge for edge in self.candidate_edges if edge.flow_weight == 3 and edge.urgency
        }
        self.edges_by_component: Dict[str, List[Edge]] = {cid: [] for cid in self.components}
        for edge in self.candidate_edges:
            self.edges_by_component[edge.source].append(edge)
            self.edges_by_component[edge.target].append(edge)

    def optimize(self, max_iterations: int = 50) -> OptimizationResult:
        selected = set(self.mandatory_edges)
        ds = UnionFind(self.components.keys())
        for edge in selected:
            ds.union(edge.source, edge.target)

        sorted_edges = sorted(
            self.candidate_edges,
            key=lambda e: (self._edge_score(e), -e.similarity),
        )

        if self.config.enforce_connectivity:
            self._build_connectivity(selected, ds, sorted_edges)
            if ds.connected_components() > 1:
                raise RuntimeError("Unable to find a weakly connected subgraph with the available edges")

        self._satisfy_resilience(selected, sorted_edges)
        self._local_pruning(selected, max_iterations)

        degrees = self._degree_map(selected)
        objective = self._objective(selected, degrees)
        feasibility = {
            "connectivity": self._is_connected(selected) if self.config.enforce_connectivity else True,
            "resilience": self._resilience_met(degrees),
            "urgency": self._urgency_met(selected),
        }
        return OptimizationResult(
            selected_edges=selected,
            component_degrees=degrees,
            objective_value=objective,
            feasibility_report=feasibility,
            iterations=max_iterations,
        )

    def _edge_score(self, edge: Edge) -> float:
        penalty = (
            self.config.lambda_edge_count
            + self.config.lambda_flow_weight * edge.flow_weight
            + self.config.lambda_urgency_weight * (1 if edge.urgency else 0)
        )
        return penalty * edge.weight

    def _build_connectivity(self, selected: Set[Edge], ds: UnionFind, sorted_edges: List[Edge]) -> None:
        for edge in sorted_edges:
            if edge in selected:
                continue
            if ds.find(edge.source) != ds.find(edge.target):
                selected.add(edge)
                ds.union(edge.source, edge.target)
            if ds.connected_components() == 1:
                break

    def _satisfy_resilience(self, selected: Set[Edge], sorted_edges: List[Edge]) -> None:
        degree_map = self._degree_map(selected)
        resilient_nodes = [
            cid for cid, component in self.components.items() if component.resilience_required
        ]
        for node in resilient_nodes:
            while degree_map["total"][node] < self.config.min_degree:
                candidate = self._best_edge_incident(node, sorted_edges, selected)
                if candidate is None:
                    break
                selected.add(candidate)
                degree_map = self._degree_map(selected)

    def _best_edge_incident(self, node: str, edges: List[Edge], selected: Set[Edge]) -> Edge | None:
        for edge in edges:
            if edge in selected:
                continue
            if edge.source == node or edge.target == node:
                return edge
        return None

    def _local_pruning(self, selected: Set[Edge], max_iterations: int) -> None:
        iterations = 0
        improved = True
        while improved and iterations < max_iterations:
            iterations += 1
            improved = False
            for edge in sorted(selected, key=self._edge_score, reverse=True):
                if edge in self.mandatory_edges:
                    continue
                tentative = selected - {edge}
                degrees = self._degree_map(tentative)
                if self.config.enforce_connectivity and not self._is_connected(tentative):
                    continue
                if not self._resilience_met(degrees):
                    continue
                if not self._urgency_met(tentative):
                    continue
                current = self._objective(selected, self._degree_map(selected))
                candidate_obj = self._objective(tentative, degrees)
                if candidate_obj <= current:
                    selected.remove(edge)
                    improved = True
                    break

    def _degree_map(self, edges: Iterable[Edge]) -> Dict[str, Dict[str, int]]:
        out_deg = {cid: 0 for cid in self.components}
        in_deg = {cid: 0 for cid in self.components}
        for edge in edges:
            out_deg[edge.source] += 1
            in_deg[edge.target] += 1
        total = {cid: out_deg[cid] + in_deg[cid] for cid in self.components}
        return {"out": out_deg, "in": in_deg, "total": total}

    def _objective(self, edges: Set[Edge], degrees: Dict[str, Dict[str, int]]) -> float:
        edge_term = self.config.lambda_edge_count * len(edges)
        flow_term = self.config.lambda_flow_weight * sum(edge.flow_weight for edge in edges)
        urgency_term = self.config.lambda_urgency_weight * sum(1 for edge in edges if edge.urgency)
        usage_term = 0.0
        time_term = 0.0
        for cid, degree in degrees["out"].items():
            component = self.components[cid]
            usage_term += component.usage_cost * degree
            time_term += component.processing_time * degree
        resilience_penalty = 0.0
        for cid, component in self.components.items():
            if not component.resilience_required:
                continue
            deficit = max(0, self.config.min_degree - degrees["total"][cid])
            if deficit > 0:
                resilience_penalty += deficit
        resilience_term = self.config.lambda_resilience_penalty * resilience_penalty
        return edge_term + flow_term + urgency_term + self.config.lambda_usage_cost * usage_term + self.config.lambda_processing_time * time_term + resilience_term

    def _is_connected(self, edges: Iterable[Edge]) -> bool:
        if not self.components:
            return True
        edge_list = list(edges)
        if not edge_list:
            return len(self.components) <= 1
        adjacency: Dict[str, Set[str]] = {cid: set() for cid in self.components}
        for edge in edge_list:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        start = next(iter(self.components))
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node] - visited)
        return len(visited) == len(self.components)

    def _resilience_met(self, degrees: Dict[str, Dict[str, int]]) -> bool:
        for cid, component in self.components.items():
            if component.resilience_required and degrees["total"][cid] < self.config.min_degree:
                return False
        return True

    def _urgency_met(self, edges: Iterable[Edge]) -> bool:
        selected_pairs = {(edge.source, edge.target) for edge in edges}
        for edge in self.mandatory_edges:
            if (edge.source, edge.target) not in selected_pairs:
                return False
        return True
