#!/usr/bin/env python3
"""Render the EWM-as-GNN factor graph for the AlphaCOGANT manuscript.

This is a deterministic thin-orchestrator figure script. All Active Inference
math (model matrices, marginal-return vector, expected free energy, policy
posterior) lives in ``src/alphacogant``; this script only *orchestrates* an
engine query at two manuscript operating points and *draws* the resulting
factor graph. No business logic is implemented here.

The figure depicts the Economic World Model (EWM) as a Generative-Notation /
factor graph:

* Five hidden-state factor nodes — ``sI, sS, sU, sTheta, sZ`` — labelled from
  :data:`alphacogant.channels.CHANNELS`.
* Two likelihoods, ``A_R`` (wired from ``sI, sU, sTheta``) and ``A_L`` (wired
  from ``sS, sTheta``), feeding observation nodes ``o_R`` and ``o_L``.
* A control node ``u`` with ``B_k`` transition edges into each factor; the edge
  labels are read from :data:`alphacogant.channels.ACTIONS`.
* An Expected-Free-Energy node ``G`` fed by *epistemic* value from the
  ``S, Z, Theta`` factors and *pragmatic* value from the ``I, U`` factors. Sign
  convention: ``G`` is minimized, value ``= -G``; epistemic (a KL) is ``>= 0``.

Node layout is deterministic (fixed manual coordinates; networkx is used only
for the edge-drawing convenience and a deterministic ``MultiDiGraph`` build —
positions never depend on a layout RNG). The two manuscript operating points
(IMPROVING / COASTING) are queried purely to annotate the ``G`` node with the
funded channel and the pragmatic/epistemic split at each regime; this exercises
the engine but does not change the deterministic geometry.

Run:
    cd projects/working/alphacogant
    PYTHONPATH=src MPLBACKEND=Agg python scripts/figures/fig_gnn_factor_graph.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic raster backend
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- Engine import (thin orchestrator: math comes only from src/alphacogant) ---
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alphacogant.channels import ACTIONS, CHANNELS, action_index, channel_index  # noqa: E402
from alphacogant.free_energy import (  # noqa: E402
    expected_free_energy,
    marginal_return_vector,
)
from alphacogant.generative_model import default_model, validate_belief_map  # noqa: E402
from alphacogant.operating_points import IMPROVING_RAW, COASTING_RAW  # noqa: E402

OUTPUT_PATH = _PROJECT_ROOT / "output" / "figures" / "gnn_factor_graph.png"

# Likelihood wiring (which factors parameterize each likelihood column tensor).
A_R_PARENTS = ("I", "U", "Theta")
A_L_PARENTS = ("S", "Theta")
# Epistemic vs pragmatic decomposition feeding the EFE node G.
EPISTEMIC_PARENTS = ("S", "Z", "Theta")
PRAGMATIC_PARENTS = ("I", "U")

# Deterministic node coordinates (x, y) — fixed, no layout RNG.
POSITIONS: dict[str, tuple[float, float]] = {
    "u": (0.0, 2.5),
    "sI": (3.0, 4.5),
    "sS": (3.0, 3.25),
    "sU": (3.0, 2.0),
    "sTheta": (3.0, 0.75),
    "sZ": (3.0, -0.5),
    "A_R": (6.0, 3.0),
    "A_L": (6.0, 1.0),
    "o_R": (8.5, 3.0),
    "o_L": (8.5, 1.0),
    "G": (4.6, -2.0),
}

# Palette.
C_FACTOR = "#1e3a8a"
C_LIKELIHOOD = "#0f766e"
C_OBS = "#7c2d12"
C_CONTROL = "#6d28d9"
C_EFE = "#b91c1c"
C_EDGE = "#475569"
C_EPI = "#0e7490"
C_PRAG = "#a16207"


def _build_belief(raw: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """Turn a (weak, strong) spec into a validated, normalized belief map."""
    belief = {ch: np.array(raw[ch], dtype=float) for ch in CHANNELS}
    return validate_belief_map(belief, context="operating_point")


def _funded_split(model, belief: dict[str, np.ndarray]) -> tuple[str, float, float]:
    """Return (funded_channel_or_action, pragmatic, epistemic) from the engine.

    Uses the engine's own marginal-return vector to pick the funded action, then
    the engine's expected-free-energy decomposition to read the pragmatic and
    epistemic *values* (value = -G; both higher-is-better) for that action.
    """
    g_t = marginal_return_vector(model, belief)
    funded_action = max(g_t, key=lambda a: g_t[a])
    efe = expected_free_energy(model, belief, funded_action)
    return ACTIONS[funded_action], float(efe.pragmatic), float(efe.epistemic)


def _draw_node(ax, name: str, label: str, color: str, *, shape: str = "circle") -> None:
    x, y = POSITIONS[name]
    if shape == "square":
        patch = mpatches.FancyBboxPatch(
            (x - 0.55, y - 0.32),
            1.1,
            0.64,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6,
            edgecolor="black",
            facecolor=color,
            zorder=3,
        )
    else:
        patch = mpatches.Circle(
            (x, y), 0.45, linewidth=1.6, edgecolor="black", facecolor=color, zorder=3
        )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
        zorder=4,
    )


def _draw_edge(ax, src: str, dst: str, color: str, *, label: str = "", style: str = "-") -> None:
    x0, y0 = POSITIONS[src]
    x1, y1 = POSITIONS[dst]
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.3,
            "linestyle": style,
            "shrinkA": 18,
            "shrinkB": 18,
            "alpha": 0.85,
        },
        zorder=1,
    )
    if label:
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.5, color=color, zorder=2)


def build_figure(model, improving: dict[str, np.ndarray], coasting: dict[str, np.ndarray]):
    """Assemble the matplotlib figure from deterministic geometry + engine annotations."""
    imp_action, imp_prag, imp_epi = _funded_split(model, improving)
    coa_action, coa_prag, coa_epi = _funded_split(model, coasting)

    fig, ax = plt.subplots(figsize=(13.0, 9.0))
    ax.set_xlim(-1.6, 10.2)
    ax.set_ylim(-3.6, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "The Economic World Model as a GNN factor graph",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    factor_labels = {f"s{ch}": f"s{ch}" for ch in CHANNELS}

    # --- Control -> factor transition edges (B_k), labelled from ACTIONS ---
    # Show the action-set label once on the topmost factor edge to avoid clutter,
    # but draw a B_k transition edge into every factor (engine-defined ACTIONS).
    action_label = "B_k: " + ", ".join(ACTIONS)
    for i, ch in enumerate(CHANNELS):
        node = f"s{ch}"
        lbl = action_label if i == 0 else f"B_{ch}"
        _draw_edge(ax, "u", node, C_CONTROL, label=lbl, style="--")

    # --- Factor -> likelihood edges ---
    for ch in A_R_PARENTS:
        _draw_edge(ax, f"s{ch}", "A_R", C_LIKELIHOOD)
    for ch in A_L_PARENTS:
        _draw_edge(ax, f"s{ch}", "A_L", C_LIKELIHOOD)

    # --- Likelihood -> observation edges ---
    _draw_edge(ax, "A_R", "o_R", C_OBS)
    _draw_edge(ax, "A_L", "o_L", C_OBS)

    # --- Factor -> EFE edges: epistemic (S,Z,Theta) and pragmatic (I,U) ---
    for ch in EPISTEMIC_PARENTS:
        _draw_edge(ax, f"s{ch}", "G", C_EPI, label="epistemic", style=":")
    for ch in PRAGMATIC_PARENTS:
        _draw_edge(ax, f"s{ch}", "G", C_PRAG, label="pragmatic", style=":")

    # --- Nodes (drawn after edges so they sit on top) ---
    _draw_node(ax, "u", "u", C_CONTROL, shape="circle")
    for ch in CHANNELS:
        _draw_node(ax, f"s{ch}", factor_labels[f"s{ch}"], C_FACTOR, shape="circle")
    _draw_node(ax, "A_R", "A_R", C_LIKELIHOOD, shape="square")
    _draw_node(ax, "A_L", "A_L", C_LIKELIHOOD, shape="square")
    _draw_node(ax, "o_R", "o_R", C_OBS, shape="circle")
    _draw_node(ax, "o_L", "o_L", C_OBS, shape="circle")
    _draw_node(ax, "G", "G", C_EFE, shape="square")

    # --- Engine-derived annotation block at the two operating points ---
    annotation = (
        "Expected Free Energy G  (minimized; value = -G)\n"
        f"  epistemic value = KL >= 0  from {{{', '.join('s' + c for c in EPISTEMIC_PARENTS)}}}\n"
        f"  pragmatic value           from {{{', '.join('s' + c for c in PRAGMATIC_PARENTS)}}}\n"
        "\n"
        "Funded policy (engine marginal-return argmax):\n"
        f"  IMPROVING: {imp_action}  |  pragmatic {imp_prag:+.3f}  epistemic {imp_epi:+.3f}\n"
        f"  COASTING:  {coa_action}  |  pragmatic {coa_prag:+.3f}  epistemic {coa_epi:+.3f}"
    )
    ax.text(
        7.1,
        -2.2,
        annotation,
        ha="left",
        va="center",
        fontsize=8.0,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8fafc", "edgecolor": C_EDGE},
        zorder=5,
    )

    # --- Legend ---
    legend_handles = [
        mpatches.Patch(color=C_FACTOR, label="hidden-state factor s_k"),
        mpatches.Patch(color=C_LIKELIHOOD, label="likelihood (A_R, A_L)"),
        mpatches.Patch(color=C_OBS, label="observation (o_R, o_L)"),
        mpatches.Patch(color=C_CONTROL, label="control u (B_k transitions)"),
        mpatches.Patch(color=C_EFE, label="Expected Free Energy G"),
        mpatches.Patch(color=C_EPI, label="epistemic value -> G"),
        mpatches.Patch(color=C_PRAG, label="pragmatic value -> G"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.95)

    # Sanity assertions tying labels to the engine's canonical ordering.
    assert tuple("s" + c for c in CHANNELS) == ("sI", "sS", "sU", "sTheta", "sZ")
    assert channel_index("Theta") == 3
    assert action_index("hold") == len(ACTIONS) - 1

    fig.tight_layout()
    return fig


def main() -> Path:
    """Render the GNN factor-graph figure and return its absolute path."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = default_model()
    improving = _build_belief(IMPROVING_RAW)
    coasting = _build_belief(COASTING_RAW)

    fig = build_figure(model, improving, coasting)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(str(OUTPUT_PATH.resolve()))
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
