#!/usr/bin/env python3
"""Landscape graphical abstract of local inference and a separate synthetic diagnostic.

Only layout is defined here. Counts, recipe constants and diagnostic results come
from the existing manuscript-token generator; no model computation is duplicated.
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

from alphacogant.model.operating_points import BOOTSTRAP_SEED  # noqa: E402
from alphacogant.tokens.manuscript_variables import generate_variables  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    CHANNEL_COLORS,
    CREATE_COLOR,
    DPI,
    EPISTEMIC_COLOR,
    NEUTRAL_COLOR,
    apply_style,
)

OUTPUT_PATH = PROJECT_ROOT / "output" / "figures" / "cover_art.png"
INK = "#172033"


def _panel(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#f8fafc", edgecolor=color, linewidth=1.8,
    ))
    ax.text(x + w / 2, y + h - 0.027, title, ha="center", va="top",
            fontsize=14, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 - 0.018, body, ha="center", va="center",
            fontsize=13, color=INK, linespacing=1.35)


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": INK})


def main() -> int:
    apply_style()
    variables = generate_variables()
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.99)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    ax.text(0.02, 0.965, "AlphaCOGANT: inference, action scores, and diagnostics",
            fontsize=19, fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.905,
            f"Standalone NumPy model • {variables['NUM_CHANNELS']} binary factors • "
            f"{variables['NUM_ACTIONS']} discrete actions • fixed A/B/C/D",
            fontsize=13, color=INK, va="top")
    ax.text(0.02, 0.851, "LOCAL OBSERVATION-CONDITIONED WORKFLOW", fontsize=14,
            fontweight="bold", color=CHANNEL_COLORS["S"], va="top")

    stages = (
        ("1  Input", "Caller supplies\nreward / loss\nbuckets + prior", NEUTRAL_COLOR),
        ("2  Filter", "One factorized\nlikelihood sweep\n(not exact joint)", CHANNEL_COLORS["S"]),
        ("3  Score / select", "One-step EFE\nmin G / max −G\ngreedy action", EPISTEMIC_COLOR),
        ("4  Predict", "$B_k$ transitions\nnext factor beliefs\nnot real execution",
         CHANNEL_COLORS["U"]),
    )
    for index, (title, body, color) in enumerate(stages):
        x = 0.025 + index * 0.25
        _panel(ax, x, 0.627, 0.20, 0.172, title, body, color)
        if index < len(stages) - 1:
            _arrow(ax, (x + 0.205, 0.713), (x + 0.24, 0.713))
    ax.text(0.5, 0.595,
            "Value = −G = pragmatic + epistemic; information gain targets Θ only.",
            ha="center", va="center", fontsize=13, color=INK)

    ax.plot([0.02, 0.98], [0.558, 0.558], color=NEUTRAL_COLOR, lw=1)
    ax.text(0.02, 0.532, "SEPARATE SYNTHETIC DIAGNOSTIC — NO ACTION-GATING ARROW",
            fontsize=14, fontweight="bold", color=CREATE_COLOR, va="top")
    ax.text(0.02, 0.483,
            "IMPROVING belief: Dirichlet perturbations; rates in nats/cycle\n"
            f"n = {variables['BOOTSTRAP_N']} • seed = {BOOTSTRAP_SEED} • "
            f"c = {variables['BOOTSTRAP_CONCENTRATION']} • "
            f"H = {variables['PLANNING_HORIZON']}; prediction only, no observations",
            fontsize=13, color=INK, va="top", linespacing=1.4)

    _panel(ax, 0.025, 0.257, 0.245, 0.147, "Paired rollouts",
           "Greedy vs hold\nfrom each same draw", CREATE_COLOR)
    _panel(ax, 0.32, 0.257, 0.35, 0.147, "Means over $b_1$ … $b_H$",
           "Create: gain over hold\nDecay: residual Θ gap", CHANNEL_COLORS["Theta"])
    _panel(ax, 0.72, 0.257, 0.255, 0.147, "t-RSI (unitless)",
           "m = create − decay\nmean(m) / SE(m)", NEUTRAL_COLOR)
    _arrow(ax, (0.279, 0.33), (0.308, 0.33))
    _arrow(ax, (0.679, 0.33), (0.708, 0.33))
    ax.text(0.5, 0.222,
            f"t-RSI {variables['HEADLINE_T_RSI']} • mean margin "
            f"{variables['BREAK_EVEN_MARGIN_MEAN']} • paired SE {variables['PAIRED_MARGIN_SE']}",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(0.5, 0.187,
            f"Positive-margin fraction {variables['BREAK_EVEN_PROB']} ≠ mean / SE; "
            "t magnitude depends on draw count.",
            ha="center", va="center", fontsize=13, color=INK)

    ax.add_patch(FancyBboxPatch(
        (0.02, 0.018), 0.96, 0.128,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#f1f5f9", edgecolor=NEUTRAL_COLOR, linewidth=1,
    ))
    ax.text(0.5, 0.082,
            "SCOPE: no coefficient learning, convex allocator or enforced admission gate.\n"
            "No temporal-authenticity validation or financial calibration.\n"
            "Source: fixed engine + canonical synthetic recipe; not measured company data.",
            ha="center", va="center", fontsize=13, color=INK, linespacing=1.4)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(str(OUTPUT_PATH.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
