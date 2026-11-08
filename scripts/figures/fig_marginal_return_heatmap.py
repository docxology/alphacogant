#!/usr/bin/env python3
"""Figure: marginal-return vector (negative EFE) as a heatmap over actions × cycles.

Thin-orchestrator figure script. All engine math comes from
``src/alphacogant.simulation`` and ``alphacogant.free_energy``.

The figure shows the negative-EFE value (= pragmatic + epistemic) for each
of the six actions across all cycles of a greedy prediction-only trajectory
from IMPROVING. Markers show the action actually selected each cycle;
no explore/exploit switch, eventual hold, observations or learning is assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.channels import ACTIONS  # noqa: E402
from alphacogant.model.generative_model import default_model  # noqa: E402
from alphacogant.model.operating_points import IMPROVING  # noqa: E402
from alphacogant.stats.simulation import simulate_trajectory  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    DPI,
    REDUCED_SIM_FOOTER,
    add_provenance_footer,
    apply_style,
)

HORIZON = 12


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    apply_style()
    import matplotlib.pyplot as plt

    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()
    trajectory = simulate_trajectory(model, IMPROVING, horizon=HORIZON, policy="greedy")

    # Build the value matrix: [action, cycle]
    value_matrix = np.array(
        [
            [record.marginal_returns[action] for action in range(len(ACTIONS))]
            for record in trajectory.cycles
        ]
    ).T  # shape (6, HORIZON)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    im = ax.imshow(
        value_matrix,
        aspect="auto",
        cmap="RdYlGn",
        interpolation="nearest",
    )

    # Mark the greedy (selected) action each cycle
    for cycle_idx, record in enumerate(trajectory.cycles):
        ax.plot(
            cycle_idx + 0.32, record.action - 0.28,
            marker="*", color="black", markeredgecolor="white", markeredgewidth=0.5,
            markersize=7, zorder=5,
        )

    ax.set_xticks(range(HORIZON))
    ax.set_xticklabels([f"{i}" for i in range(HORIZON)])
    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels(list(ACTIONS))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Action")
    ax.set_title(
        "Marginal-return vector (negative EFE) over the greedy prediction-only trajectory\n"
        f"star marker = selected action; horizon={HORIZON}; fixed model, no observations"
    )

    # Add value annotations
    for action_idx in range(len(ACTIONS)):
        for cycle_idx in range(HORIZON):
            value = value_matrix[action_idx, cycle_idx]
            rgb = np.array(im.cmap(im.norm(value))[:3])
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
            color = "white" if luminance < 0.179 else "black"
            ax.text(
                cycle_idx,
                action_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )

    fig.colorbar(im, ax=ax, label="Negative EFE (value = pragmatic + epistemic)")
    fig.tight_layout()
    add_provenance_footer(fig, REDUCED_SIM_FOOTER)  # greedy-trajectory simulation

    out_path = out_fig / "marginal_return_heatmap.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
