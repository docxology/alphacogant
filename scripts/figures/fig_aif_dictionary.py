#!/usr/bin/env python3
"""Print-scale dictionary separating corporate names from finite-model objects.

The channel table describes implemented likelihood dependencies, not the broader
conceptual role tags. Model coefficients and scientific computations are not
reimplemented here; live structural counts come from canonical channel metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.channels import ACTIONS, CHANNELS  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    CHANNEL_COLORS,
    CHANNEL_LABELS,
    DPI,
    EPISTEMIC_COLOR,
    NEUTRAL_COLOR,
    apply_style,
)

OUTPUT_PNG = PROJECT_ROOT / "output" / "figures" / "aif_dictionary.png"
INK = "#172033"

# A display dictionary of the fixed likelihood topology, not a role taxonomy.
DEPENDENCIES = {
    "I": "Reward likelihood A_R",
    "S": "Loss likelihood A_L",
    "U": "Reward likelihood A_R",
    "Theta": "A_R + A_L; information-gain target",
    "Z": "Transition only; no likelihood / value edge",
}


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.99)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.965, "Corporate vocabulary → finite-model meaning", fontsize=20,
            fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.907,
            "Names are analogies; the dependency column describes what the engine uses.",
            fontsize=13, color=INK, va="top")

    ax.add_patch(Rectangle((0.02, 0.802), 0.96, 0.065, facecolor="#172033", edgecolor="none"))
    for x, label in (
        (0.045, "Factor"), (0.185, "Corporate analogy"), (0.46, "Implemented dependency"),
    ):
        ax.text(x, 0.834, label, fontsize=13, fontweight="bold", color="white", va="center")
    for index, channel in enumerate(CHANNELS):
        y = 0.802 - (index + 1) * 0.065
        ax.add_patch(Rectangle((0.02, y), 0.96, 0.065,
                               facecolor="#f1f5f9" if index % 2 == 0 else "white",
                               edgecolor="none"))
        ax.add_patch(Rectangle((0.02, y), 0.008, 0.065,
                               facecolor=CHANNEL_COLORS[channel], edgecolor="none"))
        ax.text(0.045, y + 0.0325, "Θ" if channel == "Theta" else channel,
                fontsize=15, fontweight="bold", color=INK, va="center")
        analogy = "Parameters" if channel == "Theta" else CHANNEL_LABELS[channel]
        ax.text(0.185, y + 0.0325, analogy, fontsize=13, color=INK, va="center")
        ax.text(0.46, y + 0.0325, DEPENDENCIES[channel], fontsize=13, color=INK, va="center")
    ax.plot([0.02, 0.98], [0.477, 0.477], color=NEUTRAL_COLOR, lw=1)

    ax.text(0.02, 0.444, "EWM → FIXED A/B/C/D", fontsize=14, fontweight="bold",
            color=EPISTEMIC_COLOR, va="top")
    ax.text(0.02, 0.401,
            "A: observation likelihoods     B: action-conditioned state transitions\n"
            "C: reward/loss log preferences     D: initial factor beliefs",
            fontsize=13, color=INK, va="top", linespacing=1.5)

    ax.text(0.02, 0.303, "DECISION TERMS → COMPUTATION AND BOUNDARY", fontsize=14,
            fontweight="bold", color=EPISTEMIC_COLOR, va="top")
    terms = (
        ("Allocation", f"{len(ACTIONS)} discrete actions; not continuous capital weights"),
        ("Marginal return", "Value −G(a); not an allocation derivative"),
        ("Value", "Expected log preference + information about Θ"),
        ("t-RSI", "Paired synthetic mean / SE; not a deployment gate"),
        ("Temporal filtration", "Conceptual no-peeking rule; not enforced here"),
    )
    for index, (term, meaning) in enumerate(terms):
        y = 0.255 - index * 0.041
        ax.text(0.02, y, term, fontsize=13, fontweight="bold", color=INK, va="center")
        ax.text(0.30, y, meaning, fontsize=13, color=INK, va="center")
    ax.text(0.5, 0.032,
            f"Fixed NumPy source • {len(CHANNELS)} factors • "
            "no coefficient learning or financial calibration.",
            ha="center", va="center", fontsize=13, color=INK)
    return fig


def main() -> int:
    apply_style()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(str(OUTPUT_PNG.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
