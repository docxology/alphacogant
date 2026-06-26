#!/usr/bin/env python3
"""Epistemic vs pragmatic value per capital channel, across operating regimes.

Thin-orchestrator figure script for AlphaCOGANT. All Active Inference math is
imported from ``src/alphacogant`` (the engine); this script only orchestrates the
belief construction, calls ``free_energy.expected_free_energy`` for each funding
action, and renders the result. No business logic lives here.

For the manuscript's *improving* (stale-EWM) and *coasting* (fresh-EWM) operating
points, it decomposes the Expected Free Energy of each of the five ``fund_*``
actions into its pragmatic and epistemic value (value = -G; epistemic is a KL so
always >= 0, pragmatic is the single-cycle expected log-preference and is
negative). Grouped bars (five capital channels x {epistemic, pragmatic}) are drawn
as a two-panel small multiple (improving vs coasting), making visible that
pragmatic value rises (less negative) as the EWM freshens, while the epistemic
peak shifts from Θ (when stale) to Sensors (when fresh).

Deterministic: depends only on the engine's fixed default model and the two
literal beliefs declared below. No RNG is required, but seeded generators would
use ``np.random.default_rng(seed)`` rather than any global seed.

Run:
    cd projects/working/alphacogant
    PYTHONPATH=src MPLBACKEND=Agg python scripts/figures/fig_value_by_regime.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic backend

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from alphacogant.channels import ACTIONS, CHANNELS, action_index  # noqa: E402
from alphacogant.free_energy import expected_free_energy  # noqa: E402
from alphacogant.generative_model import default_model  # noqa: E402
from alphacogant.operating_points import IMPROVING, COASTING  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "output" / "figures" / "value_by_regime.png"

# Five capital channels in fixed order; their fund_* action and a display label.
FUNDING_CHANNELS: tuple[str, ...] = CHANNELS  # ("I", "S", "U", "Theta", "Z")
CHANNEL_LABELS: dict[str, str] = {
    "I": "Investments (I)",
    "S": "Sensors (S)",
    "U": "Actuators (U)",
    "Theta": "Theta (Θ)",
    "Z": "R&D (Z)",
}

IMPROVING_BELIEF: dict[str, np.ndarray] = IMPROVING
COASTING_BELIEF: dict[str, np.ndarray] = COASTING


def _value_matrix(model, belief) -> tuple[np.ndarray, np.ndarray]:
    """Return (epistemic, pragmatic) value vectors over the five fund_* actions.

    All math is delegated to the engine's ``expected_free_energy``. Value = -G,
    so the engine's already-non-negative ``pragmatic`` and ``epistemic`` fields
    are exactly the values plotted.
    """
    epistemic = np.zeros(len(FUNDING_CHANNELS), dtype=float)
    pragmatic = np.zeros(len(FUNDING_CHANNELS), dtype=float)
    for position, channel in enumerate(FUNDING_CHANNELS):
        action = action_index(f"fund_{channel}")
        result = expected_free_energy(model, belief, action)
        epistemic[position] = result.epistemic
        pragmatic[position] = result.pragmatic
    return epistemic, pragmatic


def _plot_panel(ax, title: str, epistemic: np.ndarray, pragmatic: np.ndarray) -> None:
    """Render one regime panel of grouped epistemic/pragmatic bars."""
    n = len(FUNDING_CHANNELS)
    x = np.arange(n)
    width = 0.38
    ax.bar(
        x - width / 2,
        epistemic,
        width,
        label="Epistemic value",
        color="#1f6feb",
        edgecolor="#0b2545",
    )
    ax.bar(
        x + width / 2,
        pragmatic,
        width,
        label="Pragmatic value",
        color="#e8893a",
        edgecolor="#5c3210",
    )
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CHANNEL_LABELS[channel] for channel in FUNDING_CHANNELS],
        rotation=30,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Value = -G (nats)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def main() -> Path:
    """Build the figure and return the written PNG path."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = default_model()

    improving_epi, improving_prag = _value_matrix(model, IMPROVING_BELIEF)
    coasting_epi, coasting_prag = _value_matrix(model, COASTING_BELIEF)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    _plot_panel(axes[0], "Improving belief (stale EWM)", improving_epi, improving_prag)
    _plot_panel(axes[1], "Coasting belief (fresh EWM)", coasting_epi, coasting_prag)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.suptitle(
        "Epistemic vs pragmatic value per capital channel, across regimes",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )
    fig.text(
        0.5,
        -0.02,
        "Pragmatic value rises (less negative) as the firm strengthens and the EWM "
        "freshens (improving → coasting); epistemic value (info gain about Θ) "
        "shifts locus, peaking at Θ when the EWM is stale and at Sensors when it is "
        f"fresh. Actions evaluated: {', '.join(a for a in ACTIONS if a != 'hold')}.",
        ha="center",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(str(OUTPUT_PATH.resolve()))
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
