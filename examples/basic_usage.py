from pathlib import Path

from dtp_optimizer import (
    Component,
    ComponentCategory,
    DigitalTwinPlatform,
    EdgeSelectionOptimizer,
    InfoType,
    Layer,
    LayerCategory,
    OptimizationConfig,
)
from dtp_optimizer.visualization import plot_result


def build_demo_platform() -> DigitalTwinPlatform:
    platform = DigitalTwinPlatform(theta_1=1.2, theta_2=0.8, theta_3=0.6)
    layers = [
        Layer("L_data", position=0, category=LayerCategory.INWARD_OPERATIONAL),
        Layer("L_service", position=1, category=LayerCategory.MANAGEMENT),
        Layer("L_delivery", position=2, category=LayerCategory.OUTWARD_OPERATIONAL),
    ]
    for layer in layers:
        platform.add_layer(layer)
    platform.add_layer_dependency("L_service", "L_data")
    platform.add_layer_dependency("L_delivery", "L_service")

    components = [
        Component(
            identifier="ingest",
            layer_id="L_data",
            category=ComponentCategory.INWARD,
            info_in={InfoType.DATA},
            info_out={InfoType.DATA, InfoType.STATUS},
            resilience_required=True,
            usage_cost=1.0,
            processing_time=1.5,
        ),
        Component(
            identifier="simulator",
            layer_id="L_service",
            category=ComponentCategory.SHARED,
            info_in={InfoType.DATA, InfoType.STATUS},
            info_out={InfoType.DATA, InfoType.DECISION},
            resilience_required=True,
            usage_cost=1.2,
            processing_time=2.0,
        ),
        Component(
            identifier="monitor",
            layer_id="L_service",
            category=ComponentCategory.SHARED,
            info_in={InfoType.DATA, InfoType.QUERY},
            info_out={InfoType.STATUS, InfoType.DATA},
            resilience_required=False,
            usage_cost=0.8,
            processing_time=1.0,
        ),
        Component(
            identifier="advisor",
            layer_id="L_delivery",
            category=ComponentCategory.OUTWARD,
            info_in={InfoType.DECISION, InfoType.STATUS},
            info_out={InfoType.DECISION, InfoType.QUERY},
            resilience_required=False,
            usage_cost=1.5,
            processing_time=2.5,
        ),
    ]
    for component in components:
        platform.add_component(component)

    platform.annotate_edge("ingest", "simulator", flow_weight=3, urgency=True)
    platform.annotate_edge("simulator", "advisor", flow_weight=2, urgency=True)
    return platform


def main() -> None:
    platform = build_demo_platform()
    candidate_edges, _ = platform.build_candidate_edges()
    optimizer = EdgeSelectionOptimizer(
        platform,
        OptimizationConfig(lambda_edge_count=0.8, lambda_resilience_penalty=4.0, min_degree=2),
    )
    result = optimizer.optimize()
    print("=== Optimized Edge Set ===")
    for edge in sorted(result.selected_edges, key=lambda e: (e.source, e.target)):
        print(
            f"{edge.source} -> {edge.target} | flow={edge.flow_weight} urgency={edge.urgency} weight={edge.weight:.2f}"
        )
    print("\n=== Component Degrees ===")
    for cid, degrees in result.component_degrees["total"].items():
        print(f"{cid}: out={result.component_degrees['out'][cid]}, in={result.component_degrees['in'][cid]}, total={degrees}")
    print("\nObjective Value:", result.objective_value)
    print("Feasibility:", result.feasibility_report)
    plot_result(
        platform,
        edges=candidate_edges,
        output_path=Path("artifacts/basic_before_optimization.png"),
        title="Pre-Optimisation Candidate Graph",
    )
    plot_result(
        platform,
        result,
        output_path=Path("artifacts/basic_after_optimization.png"),
        title="Optimised Graph",
    )


if __name__ == "__main__":
    main()
