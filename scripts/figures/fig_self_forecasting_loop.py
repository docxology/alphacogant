#!/usr/bin/env python3
"""Schematic of the Self-Forecasting Loop (whitepaper Figure 1).

Thin-orchestrator figure script. All engine math is sourced from
``src/alphacogant``; this script only orchestrates the engine query and draws a
pure-matplotlib schematic of the four-node cycle that defines the
self-forecasting loop in the AlphaFund whitepaper, mapped onto the
Active-Inference / AlphaCOGANT formalism:

    PREDICT  → query the Economic World Model (EWM) for the marginal-return
               vector g_t (= -EFE.total per action). Active-Inference
               counterpart: expected free energy evaluation under the
               generative model.
    OPTIMIZE → convex inner program over the Expected Free Energy. AIF
               counterpart: precision-weighted policy posterior G* = softmax(g_t).
    EXECUTE  → commit the argmax allocation a*_t. AIF counterpart: action
               selection / control.
    LEARN    → the realized row (Xi_{t+1}, E_{t+1}, R_t) tightens the next
               forecast. AIF counterpart: belief update / evidence accrual
               that widens the filtration F_t.

The schematic is fully deterministic (no RNG draws affect geometry; the engine
is queried only to label the PREDICT node with the real funded channel /
action from a fixed operating-point belief). No engine data file is required.

Run:
    cd projects/working/alphacogant
    PYTHONPATH=src MPLBACKEND=Agg python scripts/figures/fig_self_forecasting_loop.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# --- engine import (thin orchestrator: all math comes from src/alphacogant) ---
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alphacogant.efe.free_energy import marginal_return_vector  # noqa: E402
from alphacogant.model.channels import ACTIONS, CHANNELS  # noqa: E402
from alphacogant.model.generative_model import default_model  # noqa: E402

# Two operating-point beliefs used in the manuscript. The self-improving regime
# drives the PREDICT-node label so the schematic reflects a real engine output.
from alphacogant.model.operating_points import COASTING, IMPROVING  # noqa: E402

OUTPUT_PATH = _PROJECT_ROOT / "output" / "figures" / "self_forecasting_loop.png"


def _belief_map(spec: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """Build a validated dict[channel, array([weak, strong])] from a spec."""
    return {ch: np.array(spec[ch], dtype=float) for ch in CHANNELS}


def _funded_channel(belief: dict[str, np.ndarray]) -> tuple[str, str]:
    """Return (action_name, channel_label) the engine funds at this belief.

    All scoring is delegated to ``free_energy.marginal_return_vector`` — the
    script only reads its argmax.
    """
    mrv = marginal_return_vector(default_model(), belief)
    best_action_idx = max(mrv, key=mrv.__getitem__)
    action_name = ACTIONS[best_action_idx]
    # action names are "fund_<CHANNEL>" or "hold"
    channel = action_name.split("_", 1)[1] if action_name.startswith("fund_") else "—"
    return action_name, channel


def _draw_node(ax, xy, title, subtitle, color):
    """Draw a rounded box node centered at xy with a title and subtitle."""
    x, y = xy
    w, h = 2.7, 1.5
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=2.0,
        edgecolor=color,
        facecolor="white",
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y + 0.32,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=color,
        zorder=4,
    )
    ax.text(
        x,
        y - 0.22,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#222222",
        zorder=4,
        wrap=True,
    )


def _draw_arrow(ax, start, end, label, color, rad):
    """Draw a curved arrow between two node-edge anchor points with a label."""
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=2.0,
        color=color,
        zorder=2,
    )
    ax.add_patch(arrow)
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    # nudge label outward along the curve normal
    nx, ny = -(end[1] - start[1]), (end[0] - start[0])
    norm = np.hypot(nx, ny) or 1.0
    off = 0.62 * np.sign(rad if rad != 0 else 1.0)
    lx, ly = mx + off * nx / norm, my + off * ny / norm
    ax.text(
        lx,
        ly,
        label,
        ha="center",
        va="center",
        fontsize=7.6,
        style="italic",
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f5f5", edgecolor="none"),
        zorder=5,
    )


def main() -> int:
    """Render the Self-Forecasting Loop schematic deterministically."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Query the engine (real model outputs) for the PREDICT-node label.
    improving = _belief_map(IMPROVING)
    coasting = _belief_map(COASTING)
    imp_action, imp_channel = _funded_channel(improving)
    coast_action, _coast_channel = _funded_channel(coasting)

    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_aspect("equal")

    # Node centers arranged in a clockwise cycle.
    P_PREDICT = (3.0, 8.2)
    P_OPTIMIZE = (9.0, 8.2)
    P_EXECUTE = (9.0, 2.8)
    P_LEARN = (3.0, 2.8)

    colors = {
        "predict": "#1f6feb",
        "optimize": "#8957e5",
        "execute": "#2da44e",
        "learn": "#bf8700",
    }

    _draw_node(
        ax,
        P_PREDICT,
        "PREDICT",
        f"query EWM for marginal-return\nvector g_t  (funds {imp_channel})",
        colors["predict"],
    )
    _draw_node(
        ax,
        P_OPTIMIZE,
        "OPTIMIZE",
        "convex inner program\nover EFE  (G* = softmax g_t)",
        colors["optimize"],
    )
    _draw_node(
        ax,
        P_EXECUTE,
        "EXECUTE",
        f"commit a*_t\n({imp_action})",
        colors["execute"],
    )
    _draw_node(
        ax,
        P_LEARN,
        "LEARN",
        "realized row tightens\nthe next forecast",
        colors["learn"],
    )

    # Clockwise arrows between node edges, each annotated with its
    # Active-Inference counterpart.
    _draw_arrow(
        ax,
        (P_PREDICT[0] + 1.35, P_PREDICT[1]),
        (P_OPTIMIZE[0] - 1.35, P_OPTIMIZE[1]),
        "AIF: expected free energy\nevaluation, value = -G",
        colors["predict"],
        rad=-0.18,
    )
    _draw_arrow(
        ax,
        (P_OPTIMIZE[0], P_OPTIMIZE[1] - 0.78),
        (P_EXECUTE[0], P_EXECUTE[1] + 0.78),
        "AIF: precision-weighted\npolicy posterior",
        colors["optimize"],
        rad=-0.18,
    )
    _draw_arrow(
        ax,
        (P_EXECUTE[0] - 1.35, P_EXECUTE[1]),
        (P_LEARN[0] + 1.35, P_LEARN[1]),
        "AIF: action selection\n(control a*_t)",
        colors["execute"],
        rad=-0.18,
    )
    _draw_arrow(
        ax,
        (P_LEARN[0], P_LEARN[1] + 0.78),
        (P_PREDICT[0], P_PREDICT[1] - 0.78),
        "AIF: belief update,\nfiltration F_t widens",
        colors["learn"],
        rad=-0.18,
    )

    ax.text(
        6.0,
        10.4,
        "The Self-Forecasting Loop",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        6.0,
        9.75,
        "Whitepaper Figure 1 — the corporate loop as Active-Inference perception-action",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
    )

    # Center caption tying the cycle to the engine + the two operating points.
    ax.text(
        6.0,
        5.5,
        f"engine: {len(CHANNELS)} channels {CHANNELS}\n"
        f"{len(ACTIONS)} actions  •  improving regime funds {imp_channel}\n"
        f"coasting regime selects {coast_action}",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#fbfbfb",
            edgecolor="#cccccc",
        ),
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    abs_path = OUTPUT_PATH.resolve()
    print(str(abs_path))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    raise SystemExit(main())
