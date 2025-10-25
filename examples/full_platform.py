import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from dtp_optimizer import (
    Component,
    ComponentCategory,
    DigitalTwinPlatform,
    Edge,
    EdgeSelectionOptimizer,
    InfoType,
    Layer,
    LayerCategory,
    OptimizationConfig,
)
from dtp_optimizer.visualization import plot_result

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "platform_config.json"


@lru_cache(maxsize=1)
def load_platform_definition(path: Path = CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def build_full_platform(layer_overrides: Dict[str, str] | None = None) -> DigitalTwinPlatform:
    definition = load_platform_definition()
    platform = DigitalTwinPlatform(theta_1=1.15, theta_2=0.85, theta_3=0.75)

    layer_map = {}
    for layer_def in definition["layers"]:
        layer = Layer(
            layer_def["id"],
            position=layer_def["position"],
            category=LayerCategory[layer_def["category"]],
        )
        platform.add_layer(layer)
        layer_map[layer.identifier] = layer

    for depender, dependee in definition.get("layer_dependencies", []):
        platform.add_layer_dependency(depender, dependee)

    overrides = layer_overrides or {}
    component_defs = {comp["id"]: comp for comp in definition["components"]}
    for comp_id, layer_id in overrides.items():
        comp_def = component_defs.get(comp_id)
        if comp_def is None:
            raise ValueError(f"Override references unknown component '{comp_id}'")
        if not comp_def.get("movable", True) or comp_def["layer"] == "L_twin":
            raise ValueError(f"Component '{comp_id}' is fixed to layer '{comp_def['layer']}' and cannot be moved")
        if layer_id not in layer_map:
            raise ValueError(f"Override references unknown layer '{layer_id}' for component '{comp_id}'")

    def make_component(comp_def: Dict[str, object]) -> Component:
        comp_id = comp_def["id"]
        target_layer = overrides.get(comp_id, comp_def["layer"])
        info_in = {InfoType[item] for item in comp_def["info_in"]}
        info_out = {InfoType[item] for item in comp_def["info_out"]}
        return Component(
            identifier=comp_id,
            layer_id=target_layer,
            category=ComponentCategory[comp_def["category"]],
            info_in=info_in,
            info_out=info_out,
            resilience_required=bool(comp_def["resilience_required"]),
            usage_cost=float(comp_def["usage_cost"]),
            processing_time=float(comp_def["processing_time"]),
        )

    for comp_def in definition["components"]:
        platform.add_component(make_component(comp_def))

    for edge_def in definition["edges"]:
        platform.annotate_edge(
            edge_def["source"],
            edge_def["target"],
            flow_weight=edge_def.get("flow_weight"),
            urgency=edge_def.get("urgency"),
        )

    return platform


def compute_edge_metrics(platform: DigitalTwinPlatform, edges: Iterable[Edge]) -> Dict[str, float]:
    edge_list = list(edges)
    degrees = {cid: 0 for cid in platform.components}
    urgent_edges = 0
    cross_layer = 0
    long_range = 0
    flow_weight_sum = 0
    for edge in edge_list:
        if edge.source in degrees:
            degrees[edge.source] += 1
        if edge.target in degrees:
            degrees[edge.target] += 1
        if edge.urgency:
            urgent_edges += 1
        flow_weight_sum += edge.flow_weight
        src_layer = platform.components[edge.source].layer_id
        dst_layer = platform.components[edge.target].layer_id
        if src_layer != dst_layer:
            cross_layer += 1
        if abs(platform.layers[src_layer].position - platform.layers[dst_layer].position) > 1:
            long_range += 1
    node_count = max(1, len(degrees))
    total_degree = sum(degrees.values())
    avg_degree = total_degree / node_count
    max_degree = max(degrees.values()) if degrees else 0
    mean_degree = avg_degree
    variance = sum((deg - mean_degree) ** 2 for deg in degrees.values()) / node_count
    return {
        "edge_count": len(edge_list),
        "urgent_edge_count": urgent_edges,
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "total_degree": total_degree,
        "degree_variance": variance,
        "cross_layer_edges": cross_layer,
        "long_range_edges": long_range,
        "flow_weight_sum": flow_weight_sum,
    }


def log_effectiveness(before_stats: Dict[str, float], after_stats: Dict[str, float], objective_value: float) -> None:
    reduction = before_stats["edge_count"] - after_stats["edge_count"]
    reduction_pct = (reduction / max(1, before_stats["edge_count"])) * 100
    urgent_retention = (
        after_stats["urgent_edge_count"] / max(1, before_stats["urgent_edge_count"]) * 100
        if before_stats["urgent_edge_count"]
        else 0.0
    )
    print("\n=== Effectiveness Metrics ===")
    print(f"Edges reduced: {reduction} ({reduction_pct:.1f}%)")
    print(
        f"Avg degree before/after: {before_stats['avg_degree']:.2f} -> {after_stats['avg_degree']:.2f} "
        f"(delta {after_stats['avg_degree'] - before_stats['avg_degree']:.2f})"
    )
    print(
        f"Max degree before/after: {before_stats['max_degree']} -> {after_stats['max_degree']} "
        f"(delta {after_stats['max_degree'] - before_stats['max_degree']})"
    )
    print(
        f"Urgent edge retention: {after_stats['urgent_edge_count']} / {before_stats['urgent_edge_count']} "
        f"({urgent_retention:.1f}%)"
    )
    print(
        f"Cross-layer edges before/after: {before_stats['cross_layer_edges']} -> {after_stats['cross_layer_edges']} "
        f"(delta {after_stats['cross_layer_edges'] - before_stats['cross_layer_edges']})"
    )
    print(
        f"Long-range edges before/after: {before_stats['long_range_edges']} -> {after_stats['long_range_edges']} "
        f"(delta {after_stats['long_range_edges'] - before_stats['long_range_edges']})"
    )
    print(
        f"Degree variance before/after: {before_stats['degree_variance']:.2f} -> {after_stats['degree_variance']:.2f}"
    )
    print(
        f"Total flow weight before/after: {before_stats['flow_weight_sum']} -> {after_stats['flow_weight_sum']} "
        f"(delta {after_stats['flow_weight_sum'] - before_stats['flow_weight_sum']})"
    )
    print(f"Objective value (post-optimisation): {objective_value:.2f}")


def render_effectiveness_figure(
    platform: DigitalTwinPlatform,
    candidate_edges: List[Edge],
    result,
    before_stats: Dict[str, float],
    after_stats: Dict[str, float],
    output_path: Path | None = None,
) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping effectiveness comparison figure.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    plot_result(platform, edges=candidate_edges, ax=axes[0], title="Candidate Graph")
    plot_result(platform, result=result, ax=axes[1], title="Optimised Graph")
    metric_names = [
        "Edges",
        "Avg Degree",
        "Max Degree",
        "Urgent Edges",
        "Cross-Layer",
        "Long-Range",
        "Degree Var",
        "Flow Weight",
    ]
    before_values = [
        before_stats["edge_count"],
        before_stats["avg_degree"],
        before_stats["max_degree"],
        before_stats["urgent_edge_count"],
        before_stats["cross_layer_edges"],
        before_stats["long_range_edges"],
        before_stats["degree_variance"],
        before_stats["flow_weight_sum"],
    ]
    after_values = [
        after_stats["edge_count"],
        after_stats["avg_degree"],
        after_stats["max_degree"],
        after_stats["urgent_edge_count"],
        after_stats["cross_layer_edges"],
        after_stats["long_range_edges"],
        after_stats["degree_variance"],
        after_stats["flow_weight_sum"],
    ]
    positions = list(range(len(metric_names)))
    axes[2].bar([p - 0.2 for p in positions], before_values, width=0.4, label="Before", color="#1f77b4")
    axes[2].bar([p + 0.2 for p in positions], after_values, width=0.4, label="After", color="#ff7f0e")
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(metric_names, rotation=15, ha="right")
    axes[2].set_title("Effectiveness Metrics")
    axes[2].legend()
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    target_path = output_path or Path("artifacts/full_effectiveness_summary.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {target_path}")


def main() -> None:
    platform = build_full_platform()
    candidate_edges, _ = platform.build_candidate_edges()
    config = OptimizationConfig(
        lambda_edge_count=0.7,
        lambda_resilience_penalty=6.0,
        lambda_usage_cost=0.6,
        lambda_processing_time=0.6,
        lambda_flow_weight=1.2,
        lambda_urgency_weight=1.8,
        min_degree=3,
    )
    optimizer = EdgeSelectionOptimizer(platform, config)
    result = optimizer.optimize(max_iterations=120)

    print("=== Optimized Edge Set (Full Platform) ===")
    for edge in sorted(result.selected_edges, key=lambda e: (e.source, e.target)):
        print(
            f"{edge.source} -> {edge.target} | flow={edge.flow_weight} urgency={edge.urgency} weight={edge.weight:.2f}"
        )

    print("\n=== Component Degrees ===")
    for cid in sorted(platform.components):
        print(
            f"{cid}: out={result.component_degrees['out'][cid]}, in={result.component_degrees['in'][cid]}, total={result.component_degrees['total'][cid]}"
        )

    print("\nObjective Value:", result.objective_value)
    print("Feasibility:", result.feasibility_report)

    plot_result(
        platform,
        edges=candidate_edges,
        output_path=Path("artifacts/full_before_optimization.png"),
        title="Full Platform - Candidate Graph",
    )
    plot_result(
        platform,
        result,
        output_path=Path("artifacts/full_after_optimization.png"),
        title="Full Platform - Optimised Graph",
    )
    before_stats = compute_edge_metrics(platform, candidate_edges)
    after_stats = compute_edge_metrics(platform, result.selected_edges)
    log_effectiveness(before_stats, after_stats, result.objective_value)
    render_effectiveness_figure(platform, candidate_edges, result, before_stats, after_stats)
    critical_nodes = {
        cid
        for cid, component in platform.components.items()
        if component.resilience_required
    }
    critical_edges = {
        (edge.source, edge.target)
        for edge in result.selected_edges
        if edge.urgency and edge.flow_weight == 3
    }
    plot_result(
        platform,
        result,
        output_path=Path("artifacts/full_critical_paths.png"),
        title="Critical Nodes & Pathways",
        highlight_nodes=critical_nodes,
        highlight_edge_pairs=critical_edges,
    )


if __name__ == "__main__":
    main()
