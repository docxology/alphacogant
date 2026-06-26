#!/usr/bin/env python3
"""Two-column visual dictionary: AlphaFund constructs -> Active-Inference objects.

Thin-orchestrator figure script. All economic / Active-Inference math is owned by
``src/alphacogant``; this script only queries the engine for a few live operating-
point numbers (used in the provenance footer) and draws the schematic. The figure
itself is a static, deterministic dictionary mapping: each left-hand AlphaFund
construct is connected by a line to its right-hand Active-Inference counterpart,
following the coverage rubric in ``docs/whitepaper_concepts.md`` and the sign /
value conventions in ``manuscript/05_epistemic_and_pragmatic_value.md``.

Mapping (left -> right):
    EWM                          -> generative model  Ŵ_t
    channels Ξ_t                 -> hidden state  s
    a_t                          -> control / policy  π
    J_t                          -> negative EFE pragmatic  (−E[EFE pragmatic])
    marginal-return g_t          -> −∂G/∂a  (the negative-EFE-per-action vector)
    Sensors / R&D                -> epistemic value  (KL information gain ≥ 0)
    Investments / Actuators      -> pragmatic value  (expected log-equity)
    t-RSI                        -> EFE-improvement certificate
    filtration F_t               -> measurability / no-peeking

Run:
    cd projects/working/alphacogant
    PYTHONPATH=src MPLBACKEND=Agg python scripts/figures/fig_aif_dictionary.py

Prints the absolute PNG path on success.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic raster backend

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# --- engine import (business logic lives only in src/alphacogant) -------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402

from alphacogant.channels import ACTIONS, CHANNELS  # noqa: E402
from alphacogant.free_energy import (  # noqa: E402
    expected_free_energy,
    marginal_return_vector,
)
from alphacogant.generative_model import default_model  # noqa: E402

OUTPUT_PNG = PROJECT_ROOT / "output" / "figures" / "aif_dictionary.png"

from alphacogant.operating_points import IMPROVING, COASTING  # noqa: E402

# Dictionary rows: (AlphaFund construct, Active-Inference object, semantic tag).
ROWS: tuple[tuple[str, str, str], ...] = (
    ("EWM  $\\hat W_t$", "generative model  $\\hat W_t$", "model"),
    ("channels  $\\Xi_t \\to (I,S,U,\\Theta,Z)$", "hidden state  $s$", "state"),
    ("action vector  $a_t$", "control / policy  $\\pi$", "control"),
    ("objective  $J_t$", "$-\\mathbb{E}[$EFE pragmatic$]$", "pragmatic"),
    ("marginal return  $g_t$", "$-\\partial G/\\partial a$", "marginal"),
    ("Sensors / R&D", "epistemic value  ($D_{KL}\\!\\geq 0$)", "epistemic"),
    ("Investments / Actuators", "pragmatic value (log-equity)", "pragmatic"),
    ("t-RSI", "EFE-improvement certificate", "certificate"),
    ("filtration  $F_t$", "measurability / no-peeking", "measure"),
)

# Color per semantic tag (deterministic, colourblind-aware muted palette).
TAG_COLORS: dict[str, str] = {
    "model": "#3B5BA5",
    "state": "#4C8C99",
    "control": "#7A5BA5",
    "pragmatic": "#2E7D5B",
    "marginal": "#B07A2E",
    "epistemic": "#A53B6B",
    "certificate": "#A53B3B",
    "measure": "#566573",
}


def _operating_point_facts() -> dict[str, float]:
    """Pull a few live engine numbers for the provenance footer (no plotting math).

    Returns the marginal-return argmax action and the funded channel's EFE
    pragmatic / epistemic split at the IMPROVING operating point, plus a
    KL-nonnegativity confirmation across all actions/regimes.
    """
    model = default_model()

    g = marginal_return_vector(model, IMPROVING)
    funded_action = max(g, key=lambda a: g[a])

    funded_efe = expected_free_energy(model, IMPROVING, funded_action)

    # epistemic value is a KL -> always >= 0 (assert the sign convention holds).
    min_epistemic = min(
        expected_free_energy(model, belief, action).epistemic
        for belief in (IMPROVING, COASTING)
        for action in range(len(ACTIONS))
    )

    return {
        "funded_action_idx": float(funded_action),
        "funded_pragmatic": funded_efe.pragmatic,
        "funded_epistemic": funded_efe.epistemic,
        "min_epistemic": min_epistemic,
    }


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
        mutation_aspect=1.0,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#1A1A1A",
    )


def build_figure(facts: dict[str, float]) -> plt.Figure:
    """Draw the two-column dictionary schematic. Pure layout; no business logic."""
    fig, ax = plt.subplots(figsize=(11.0, 8.5), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column geometry.
    n = len(ROWS)
    box_w = 0.34
    box_h = 0.072
    left_x = 0.04
    right_x = 0.62
    top = 0.86
    bottom = 0.16
    ys = np.linspace(top, bottom, n)

    # Headers.
    ax.text(
        0.5,
        0.955,
        "AlphaFund $\\leftrightarrow$ Active Inference: a construct dictionary",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color="#1A1A1A",
    )
    ax.text(
        left_x + box_w / 2.0,
        0.915,
        "AlphaFund construct",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        right_x + box_w / 2.0,
        0.915,
        "Active-Inference object",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )

    # Rows: boxes + connecting line.
    for (lhs, rhs, tag), y in zip(ROWS, ys, strict=True):
        color = TAG_COLORS[tag]
        cy = y - box_h / 2.0
        _draw_box(ax, left_x, y - box_h, box_w, box_h, lhs, color)
        _draw_box(ax, right_x, y - box_h, box_w, box_h, rhs, color)
        ax.add_line(
            Line2D(
                [left_x + box_w, right_x],
                [cy, cy],
                color=color,
                linewidth=1.6,
                alpha=0.7,
                solid_capstyle="round",
            )
        )
        # small arrow head toward the AIF object
        ax.annotate(
            "",
            xy=(right_x, cy),
            xytext=(right_x - 0.03, cy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, alpha=0.85),
        )

    # Sign-convention legend.
    ax.text(
        0.5,
        0.11,
        "Sign convention: EFE $G$ is minimized; value $= -G$. "
        "Pragmatic and epistemic are values (higher = better); "
        "epistemic (a KL) is $\\geq 0$.",
        ha="center",
        va="center",
        fontsize=9.5,
        style="italic",
        color="#444444",
    )

    # Provenance footer with live engine values (proves engine is the data source).
    funded_idx = int(facts["funded_action_idx"])
    funded_name = ACTIONS[funded_idx]
    footer = (
        f"Live engine (IMPROVING operating point): "
        f"marginal-return argmax = {funded_name}  |  "
        f"funded pragmatic value = {facts['funded_pragmatic']:.4f}  |  "
        f"funded epistemic value = {facts['funded_epistemic']:.4f}  |  "
        f"min epistemic over all actions/regimes = {facts['min_epistemic']:.4f} ($\\geq 0$)"
    )
    ax.text(
        0.5,
        0.045,
        footer,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#666666",
    )
    ax.text(
        0.5,
        0.018,
        f"channels = {', '.join(CHANNELS)}   |   actions = {len(ACTIONS)}",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#888888",
    )

    fig.tight_layout()
    return fig


def main() -> int:
    """Generate the AIF construct-dictionary figure deterministically."""
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    facts = _operating_point_facts()
    fig = build_figure(facts)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(str(OUTPUT_PNG.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
