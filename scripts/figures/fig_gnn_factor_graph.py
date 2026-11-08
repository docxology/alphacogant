#!/usr/bin/env python3
"""Untangled fixed-model dependency graph in Generative-Notation style.

Dashed transition arrows, solid likelihood arrows and dotted score readouts
encode distinct relationships. Geometry is schematic, not inferred topology or
neural message passing. No scientific computation is duplicated in this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.channels import ACTIONS, CHANNELS  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    CHANNEL_COLORS,
    DPI,
    EPISTEMIC_COLOR,
    NEUTRAL_COLOR,
    apply_style,
)

OUTPUT_PATH = PROJECT_ROOT / "output" / "figures" / "gnn_factor_graph.png"
INK = "#172033"

# Display order makes each likelihood's parents contiguous; canonical engine
# ordering is unchanged. Theta is shared; Z deliberately has no outgoing edge.
FACTOR_X = {"I": 0.09, "U": 0.26, "Theta": 0.46, "S": 0.68, "Z": 0.90}
LIKELIHOODS = {
    "A_R": (0.22, ("I", "U", "Theta"), "reward o_R"),
    "A_L": (0.65, ("S", "Theta"), "loss o_L"),
}


def _node(ax, x, y, w, h, label, color, *, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        facecolor="white", edgecolor=color, linewidth=2,
        linestyle="--" if dashed else "-", zorder=3,
    ))
    ax.text(x, y, label, ha="center", va="center", fontsize=14,
            fontweight="bold", color=INK, zorder=4, linespacing=1.3)


def _edge(ax, start, end, *, style="-", color=INK):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "lw": 1.8,
                            "linestyle": style, "color": color}, zorder=1)


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.99)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.967, "Economic World Model: exact dependency wiring", fontsize=20,
            fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.912,
            "Dashed: $B_k$ transitions    Solid: likelihoods    Dotted: score readouts",
            fontsize=13, color=INK, va="top")

    _node(ax, 0.46, 0.841, 0.35, 0.067,
          f"control u • {len(ACTIONS)} actions", EPISTEMIC_COLOR, dashed=True)
    ax.plot([min(FACTOR_X.values()), max(FACTOR_X.values())], [0.774, 0.774],
            color=EPISTEMIC_COLOR, linestyle="--", lw=1.8)
    ax.plot([0.46, 0.46], [0.80, 0.774], color=EPISTEMIC_COLOR, linestyle="--", lw=1.8)
    ax.text(0.04, 0.834, "$B_k$ controls\nevery factor", fontsize=13,
            va="center", color=INK, linespacing=1.3)

    for channel in CHANNELS:
        x = FACTOR_X[channel]
        _edge(ax, (x, 0.774), (x, 0.707), style="--", color=EPISTEMIC_COLOR)
        _node(ax, x, 0.662, 0.11, 0.072,
              "Θ" if channel == "Theta" else channel, CHANNEL_COLORS[channel])
    ax.text(0.90, 0.568, "Z has no\nlikelihood\nor value edge", ha="center",
            va="center", fontsize=13, color=INK, linespacing=1.35)

    for name, (x, parents, observation) in LIKELIHOODS.items():
        for parent in parents:
            _edge(ax, (FACTOR_X[parent], 0.618), (x, 0.536))
        parent_label = ",".join("Θ" if p == "Theta" else p for p in parents)
        _node(ax, x, 0.482, 0.26, 0.10, f"{name}\nP(o | {parent_label})", CHANNEL_COLORS["S"])
        _edge(ax, (x, 0.424), (x, 0.347))
        _node(ax, x, 0.305, 0.23, 0.067, observation, NEUTRAL_COLOR)

    # These are score dependencies, not observation-to-action feedback arrows.
    # Readouts occupy the central corridor, away from the observation edges.
    _edge(ax, (0.358, 0.47), (0.405, 0.211), style=":", color=EPISTEMIC_COLOR)
    _edge(ax, (0.512, 0.47), (0.495, 0.211), style=":", color=EPISTEMIC_COLOR)
    ax.add_patch(FancyBboxPatch(
        (0.18, 0.075), 0.61, 0.128,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#f5f3ff", edgecolor=EPISTEMIC_COLOR, linewidth=1.6, zorder=3,
    ))
    ax.text(0.485, 0.139,
            "Minimize G; value = −G\n"
            "Pragmatic: expected reward/loss log preference\n"
            "Epistemic: information about Θ via A_R + A_L",
            ha="center", va="center", fontsize=13, color=INK, linespacing=1.4, zorder=4)
    ax.text(0.5, 0.028,
            "Fixed NumPy model topology; no learned neural network or coefficient learning.",
            ha="center", va="center", fontsize=13, color=INK)
    return fig


def main() -> Path:
    apply_style()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(str(OUTPUT_PATH.resolve()))
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
