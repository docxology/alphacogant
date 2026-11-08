#!/usr/bin/env python3
"""Responsibility map for the self-forecasting cycle and external handoffs.

Solid arrows show local computations. Dashed arrows show conceptual observation
and execution handoffs, not an implemented corporate deployment or learning loop.
Illustrative selected-action labels are queried from the fixed engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as PlotPath  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.efe.free_energy import greedy_action  # noqa: E402
from alphacogant.model.channels import ACTIONS  # noqa: E402
from alphacogant.model.generative_model import default_model  # noqa: E402
from alphacogant.model.operating_points import COASTING, IMPROVING  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    CHANNEL_COLORS,
    DPI,
    EPISTEMIC_COLOR,
    NEUTRAL_COLOR,
    apply_style,
)

OUTPUT_PATH = PROJECT_ROOT / "output" / "figures" / "self_forecasting_loop.png"
INK = "#172033"


def _box(ax, x, y, w, h, title, body, color, *, external=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="white", edgecolor=color, linewidth=1.8,
        linestyle="--" if external else "-",
    ))
    ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top",
            fontsize=14, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 - 0.024, body, ha="center", va="center",
            fontsize=13, color=INK, linespacing=1.35)


def _route(ax, points, *, external=False):
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=15, linewidth=1.8,
        color=NEUTRAL_COLOR if external else CHANNEL_COLORS["S"],
        linestyle="--" if external else "-",
    ))


def main() -> int:
    apply_style()
    model = default_model()
    improving_action = ACTIONS[greedy_action(model, IMPROVING)]
    coasting_action = ACTIONS[greedy_action(model, COASTING)]

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.99)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.965, "Self-forecasting: who closes the loop?", fontsize=20,
            fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.905,
            "Solid = local computation     Dashed = conceptual external handoff",
            fontsize=13, color=INK, va="top")

    ax.add_patch(FancyBboxPatch(
        (0.02, 0.487), 0.96, 0.347,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#eff6ff", edgecolor=CHANNEL_COLORS["S"], linewidth=1.2,
    ))
    ax.text(0.045, 0.822, "LOCAL ENGINE • fixed A/B/C/D", fontsize=13,
            fontweight="bold", color=CHANNEL_COLORS["S"], va="top")
    _box(ax, 0.055, 0.53, 0.25, 0.177, "FILTER",
         "Supplied observations\n+ current factor prior\none local sweep", CHANNEL_COLORS["S"])
    _box(ax, 0.375, 0.53, 0.25, 0.177, "SCORE / SELECT",
         f"{len(ACTIONS)} discrete actions\nminimize G\nmaximize value = −G", EPISTEMIC_COLOR)
    _box(ax, 0.695, 0.53, 0.25, 0.177, "PREDICT",
         "$B_k$ state transitions\nnext factor prior\nnot model refitting", CHANNEL_COLORS["U"])
    _route(ax, [(0.315, 0.615), (0.365, 0.615)])
    _route(ax, [(0.635, 0.615), (0.685, 0.615)])
    _route(ax, [(0.82, 0.715), (0.82, 0.748), (0.18, 0.748), (0.18, 0.715)])
    ax.text(0.5, 0.759, "prediction becomes the next prior", fontsize=13,
            ha="center", va="bottom", color=CHANNEL_COLORS["S"])

    _box(ax, 0.055, 0.225, 0.29, 0.17, "OBSERVE • EXTERNAL",
         "Obtain reward / loss\nevidence; supply buckets", NEUTRAL_COLOR, external=True)
    _box(ax, 0.655, 0.225, 0.29, 0.17, "EXECUTE • EXTERNAL",
         "Decide and commit\nreal-world resources", NEUTRAL_COLOR, external=True)
    _route(ax, [(0.50, 0.52), (0.50, 0.425), (0.80, 0.425), (0.80, 0.405)], external=True)
    ax.text(0.65, 0.435, "selected-action handoff", ha="center", va="bottom",
            fontsize=13, color=INK)
    _route(ax, [(0.645, 0.296), (0.355, 0.296)], external=True)
    ax.text(0.50, 0.319, "new evidence", ha="center", va="bottom", fontsize=13, color=INK)
    _route(ax, [(0.18, 0.405), (0.18, 0.52)], external=True)

    ax.text(0.5, 0.165,
            f"Engine examples, not company results: IMPROVING → {improving_action};\n"
            f"COASTING → {coasting_action} (one-step point-belief selections).",
            ha="center", va="center", fontsize=13, color=INK, linespacing=1.35)
    ax.text(0.5, 0.064,
            "No refitting, convex allocator, temporal validation or deployment commitment is run.\n"
            "Source: fixed NumPy engine; corporate cycle is an analogy, not financial evidence.",
            ha="center", va="center", fontsize=13, color=INK, linespacing=1.4)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(str(OUTPUT_PATH.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
