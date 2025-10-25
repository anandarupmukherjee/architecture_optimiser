from __future__ import annotations

from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .models import ComponentCategory, DigitalTwinPlatform, Edge

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


LAYER_COLOR_MAP = {
    "L_twin": "#b6e4a7",
    "L_access": "#c9c3f3",
    "L_management": "#f4d7a6",
    "L_integration": "#b7d3f2",
    "L_resilience": "#ffe29a",
    "L_compliance": "#c8e6c9",
}

FALLBACK_LAYER_COLORS = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
]

CATEGORY_MARKERS = {
    ComponentCategory.INWARD: "o",
    ComponentCategory.SHARED: "s",
    ComponentCategory.OUTWARD: "^",
}


def plot_result(
    platform: DigitalTwinPlatform,
    result=None,
    *,
    edges: Optional[Iterable[Edge]] = None,
    ax=None,
    output_path: Path | None = None,
    show: bool = False,
    title: str = "DTP Graph",
    highlight_nodes: Optional[Iterable[str]] = None,
    highlight_edge_pairs: Optional[Iterable[tuple[str, str]]] = None,
) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping graph visualisation.")
        return
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        created_fig = True
    else:
        fig = ax.figure
    edges_to_plot: Sequence[Edge]
    if edges is not None:
        edges_to_plot = list(edges)
    elif result is not None:
        edges_to_plot = list(result.selected_edges)
    else:
        raise ValueError("Either 'result' or 'edges' must be provided to plot_result")
    highlight_nodes_set = set(highlight_nodes or [])
    highlight_edges_set = set(highlight_edge_pairs or [])

    layers = list(platform.layers.values())
    layer_order = {layer.identifier: idx for idx, layer in enumerate(sorted(layers, key=lambda l: l.position))}
    components_by_layer: defaultdict[str, list[str]] = defaultdict(list)
    for component in platform.components.values():
        components_by_layer[component.layer_id].append(component.identifier)

    layer_colors = {}
    fallback_cycle = cycle(FALLBACK_LAYER_COLORS)
    for layer_id in components_by_layer:
        color = LAYER_COLOR_MAP.get(layer_id)
        if color is None:
            color = next(fallback_cycle)
        layer_colors[layer_id] = color

    positions: dict[str, tuple[float, float]] = {}
    for layer_id, comps in components_by_layer.items():
        comps.sort()
        count = len(comps)
        for idx, comp_id in enumerate(comps):
            x = layer_order[layer_id] * 3.0
            y = (idx - (count - 1) / 2) * 1.5
            positions[comp_id] = (x, y)

    for comp_id, component in platform.components.items():
        x, y = positions[comp_id]
        marker = CATEGORY_MARKERS.get(component.category, "o")
        node_color = layer_colors.get(component.layer_id, "#7f7f7f")
        highlight = comp_id in highlight_nodes_set
        ax.scatter(
            x,
            y,
            s=880 if highlight else 820,
            color=node_color,
            edgecolor="#d62728" if highlight else "black",
            linewidths=2.5 if highlight else 1.2,
            marker=marker,
            zorder=3,
        )
        ax.text(x, y, comp_id, ha="center", va="center", color="black", fontsize=10, weight="bold", zorder=4)

    for edge in edges_to_plot:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        pair = (edge.source, edge.target)
        if pair in highlight_edges_set:
            color = "#d62728"
            width = 3.8
        else:
            color = "#ff7f0e" if edge.urgency else "#bbbbbb"
            width = 2 + edge.flow_weight * 0.3
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=width),
            zorder=2,
        )
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, f"f={edge.flow_weight}", color=color, fontsize=8, ha="center")

    ax.set_axis_off()
    ax.set_title(title, fontsize=14, weight="bold")
    layer_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=layer_id,
            markerfacecolor=color,
            markersize=12,
            markeredgecolor="black",
        )
        for layer_id, color in layer_colors.items()
    ]
    category_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=CATEGORY_MARKERS.get(category, "o"),
            color="black",
            label=category.name.title(),
            markersize=10,
            linestyle="None",
        )
        for category in ComponentCategory
    ]
    urgent_handle = plt.Line2D([0], [0], color="#ff7f0e", lw=2, label="Urgent Edge")
    layer_legend = ax.legend(handles=layer_handles, loc="upper left", title="Layers")
    ax.add_artist(layer_legend)
    category_handles.append(urgent_handle)
    ax.legend(handles=category_handles, loc="lower left", title="Category/Flow")

    if created_fig:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
            print(f"Saved graph visualisation to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
