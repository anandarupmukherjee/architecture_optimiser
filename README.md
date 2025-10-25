# Digital Twin Platform Architecture Optimiser

This repository implements the graph-based model and optimisation workflow described in the accompanying paper. The goal is to automatically derive a low-complexity yet constraint-compliant interconnection plan for modular Digital Twin Platform (DTP) components while accounting for information reuse, resilience, and communication urgency.

## Features
- Layer/component abstractions with semantic categories, information types, and operational attributes.
- Automatic derivation of candidate edges between compatible components, including similarity scores, layer-alignment factors, and configurable flow/urgency metadata.
- Centrality-aware edge weighting based on degrees and weighted degrees.
- Greedy-plus-pruning optimiser that enforces connectivity, resilience-degree, and urgency constraints while minimising a composite cost.
- Lightweight example demonstrating how to define a platform and run the optimiser without external dependencies.
- Optional Matplotlib visualisation helper (`dtp_optimizer.visualization.plot_result`) to render the optimised topology.
- Layer-placement explorer that cycles through candidate layouts, scores them on complexity/effectiveness metrics, and prints per-component placement suggestions for critical nodes.

## Repository Layout
- `src/dtp_optimizer/models.py` — Core data structures for layers, components, candidate edges, and metric calculations.
- `src/dtp_optimizer/optimizer.py` — Heuristic solver that produces an optimised subgraph subject to the defined constraints.
- `examples/basic_usage.py` — End-to-end example that instantiates a small platform and runs the optimiser.
- `examples/full_platform.py` — Larger scenario covering all component categories/constraints described in the mathematical model.
- `examples/layer_placement_explorer.py` — Iterates over alternative layer placements for key components and reports the best-performing layout using the effectiveness metrics.
- `config/platform_config.json` — Declarative description of layers, components, immovability flags, and annotated inter-component flows.

## Quick Start
```bash
cd /Users/anandarupmukherjee/Dropbox\ (Personal)/Mac/Desktop/PILOTS/Paper_DT_Arch_optimiser
PYTHONPATH=src python3 examples/basic_usage.py
PYTHONPATH=src python3 examples/full_platform.py
PYTHONPATH=src python3 examples/layer_placement_explorer.py
```
The script prints the selected edge set, component degrees, overall objective value, and feasibility flags for the hard constraints.

## Defining a Platform
1. **Create layers** using `Layer(identifier, position, category)` and register them with `DigitalTwinPlatform.add_layer`.
2. **Add components** via `Component`, specifying:
   - `info_in` / `info_out` sets drawn from `{data, decision, status, query}`;
   - `category` (`inward`, `outward`, or `shared`);
   - Boolean `resilience_required`, plus usage-cost and processing-time weights.
3. **Describe inter-layer dependencies** with `add_layer_dependency(depender, dependee)` to bias alignment scores.
4. **Optional edge annotations** allow overriding automatically inferred flow weights or urgency flags for specific pairs.

Calling `platform.build_candidate_edges()` returns the enriched edge list and per-component metrics, which are consumed by the optimiser.

## Optimisation Workflow
Instantiate `EdgeSelectionOptimizer` with the platform plus an `OptimizationConfig` that sets the λ-weights from the mathematical formulation:

```python
config = OptimizationConfig(
    lambda_edge_count=0.8,
    lambda_resilience_penalty=4.0,
    lambda_usage_cost=0.5,
    lambda_processing_time=0.5,
    lambda_flow_weight=1.0,
    lambda_urgency_weight=1.5,
    min_degree=2,
)
optimizer = EdgeSelectionOptimizer(platform, config)
result = optimizer.optimize()
```

The optimiser executes three stages:
1. **Connectivity assembly** using a union-find structure to establish a weakly connected backbone (respecting mandatory urgent edges).
2. **Resilience satisfaction** adds the cheapest incident edges until each resilient node reaches the required degree.
3. **Local pruning** iteratively removes redundant edges when doing so does not violate constraints and does not increase the objective value.

`OptimizationResult` provides the selected edge set, degree tables, total objective value, feasibility report, and iteration count for transparency.

## Configuration & Layer Placement
- `config/platform_config.json` enumerates every layer, component, and annotated edge along with whether a component is movable. Components in the twin layer are flagged as immovable; any attempt to override their placement raises an error.
- `examples/layer_placement_explorer.py` consumes the same config, systematically tries alternative placements (respecting the immovable constraint), reports effectiveness metrics for each scenario, and prints per-component suggestions for critical nodes.

## Next Steps
- Extend `EdgeSelectionOptimizer` with additional heuristics (e.g., simulated annealing or QAOA-compatible QUBO encoding).
- Integrate persistence/visualisation utilities to export the optimised architecture.
- Wrap the package with CLI/REST interfaces for integration with larger tooling ecosystems.
