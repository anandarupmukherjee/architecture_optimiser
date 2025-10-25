from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtp_optimizer import EdgeSelectionOptimizer, OptimizationConfig
from examples.full_platform import (
    build_full_platform,
    compute_edge_metrics,
    load_platform_definition,
    log_effectiveness,
    render_effectiveness_figure,
)

DEFINITION = load_platform_definition()
LAYER_IDS = [layer["id"] for layer in DEFINITION["layers"]]
IMMOVABLE_COMPONENTS = {
    comp["id"]
    for comp in DEFINITION["components"]
    if not comp.get("movable", True) or comp["layer"] == "L_twin"
}

SCENARIOS: Dict[str, Dict[str, str]] = {
    "baseline": {},
    "integration_co_located": {
        "api_gateway": "L_management",
        "data_transformation": "L_management",
        "interoperability_services": "L_management",
    },
    "resilience_cluster": {
        "platform_manager": "L_resilience",
        "orchestrator": "L_resilience",
        "analytics_insights": "L_resilience",
    },
    "compliance_embedded": {
        "service_registry": "L_integration",
        "twin_registry": "L_integration",
        "audit_trail": "L_integration",
    },
    "access_promoted": {
        "sensor_access": "L_management",
        "data_store": "L_management",
        "analytics_insights": "L_management",
    },
    "balanced_alignment": {
        "api_gateway": "L_management",
        "data_transformation": "L_management",
        "interoperability_services": "L_management",
        "platform_manager": "L_integration",
        "analytics_insights": "L_integration",
        "service_registry": "L_integration",
    },
}

CONFIG = OptimizationConfig(
    lambda_edge_count=0.7,
    lambda_resilience_penalty=6.0,
    lambda_usage_cost=0.6,
    lambda_processing_time=0.6,
    lambda_flow_weight=1.2,
    lambda_urgency_weight=1.8,
    min_degree=3,
)


def evaluate_scenario(name: str, overrides: Dict[str, str]) -> Dict:
    platform = build_full_platform(layer_overrides=overrides)
    candidate_edges, _ = platform.build_candidate_edges()
    optimizer = EdgeSelectionOptimizer(platform, CONFIG)
    result = optimizer.optimize(max_iterations=120)
    before_stats = compute_edge_metrics(platform, candidate_edges)
    after_stats = compute_edge_metrics(platform, result.selected_edges)
    score: Tuple[float, float, float, float] = (
        result.objective_value,
        after_stats["edge_count"],
        after_stats["cross_layer_edges"],
        after_stats["avg_degree"],
    )
    return {
        "name": name,
        "platform": platform,
        "candidate_edges": candidate_edges,
        "result": result,
        "before": before_stats,
        "after": after_stats,
        "score": score,
        "overrides": dict(overrides),
    }


def format_row(label: str, after_stats: Dict[str, float], objective: float) -> str:
    return (
        f"{label:<22} | edges={after_stats['edge_count']:>4} | "
        f"cross={after_stats['cross_layer_edges']:>4} | "
        f"long={after_stats['long_range_edges']:>4} | "
        f"avg_deg={after_stats['avg_degree']:>5.2f} | "
        f"obj={objective:>8.2f}"
    )


def suggest_component_placements(component_ids: List[str]) -> None:
    baseline_eval = evaluate_scenario("baseline_check", {})
    baseline_layers = {
        cid: comp.layer_id for cid, comp in baseline_eval["platform"].components.items()
    }
    baseline_score = baseline_eval["score"]
    for cid in component_ids:
        current_layer = baseline_layers.get(cid)
        if current_layer is None:
            continue
        if cid in IMMOVABLE_COMPONENTS or current_layer == "L_twin":
            print(f"{cid:<20} fixed in {current_layer} (movement disallowed).")
            continue
        best_eval = None
        best_score = baseline_score
        best_layer = current_layer
        for layer in LAYER_IDS:
            if layer == current_layer:
                continue
            evaluation = evaluate_scenario(f"{cid}->{layer}", {cid: layer})
            if evaluation["score"] < best_score:
                best_score = evaluation["score"]
                best_eval = evaluation
                best_layer = layer
        if best_eval is None:
            print(f"{cid:<20} stays in {current_layer} (no better placement found).")
        else:
            print(
                f"{cid:<20} move {current_layer} -> {best_layer} | "
                f"edges {best_eval['after']['edge_count']}, cross {best_eval['after']['cross_layer_edges']}, "
                f"obj {best_eval['result'].objective_value:.2f}"
            )


def main() -> None:
    evaluations: List[Dict] = []
    for name, overrides in SCENARIOS.items():
        print(f"Evaluating scenario: {name}")
        try:
            evaluation = evaluate_scenario(name, overrides)
        except ValueError as exc:
            print(f"  Skipping scenario due to constraint: {exc}")
            continue
        evaluations.append(evaluation)
        print(
            format_row(
                name,
                evaluation["after"],
                evaluation["result"].objective_value,
            )
        )

    best = min(evaluations, key=lambda e: e["score"])
    print("\n=== Recommended Placement Scenario ===")
    print(format_row(best["name"], best["after"], best["result"].objective_value))
    overrides = best.get("overrides", {}) or SCENARIOS.get(best["name"], {})
    if overrides:
        print("Layer overrides:")
        for comp, layer in overrides.items():
            print(f"  - {comp} -> {layer}")

    print("\nDetailed effectiveness metrics for best scenario:")
    log_effectiveness(best["before"], best["after"], best["result"].objective_value)
    output_path = Path(f"artifacts/layer_layout_{best['name']}.png")
    render_effectiveness_figure(
        best["platform"],
        best["candidate_edges"],
        best["result"],
        best["before"],
        best["after"],
        output_path=output_path,
    )
    print(f"Saved best-scenario visual comparison to {output_path}")

    baseline_platform = best["platform"] if best["name"] == "baseline" else build_full_platform()
    critical_components = [
        cid for cid, component in baseline_platform.components.items() if component.resilience_required
    ]
    print("\n=== Per-Component Placement Suggestions (Critical Nodes) ===")
    suggest_component_placements(critical_components)


if __name__ == "__main__":
    main()
