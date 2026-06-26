#!/usr/bin/env python3
"""Figure: marginal-return vector (negative EFE) as a heatmap over actions × cycles.

Thin-orchestrator figure script. All engine math comes from
``src/alphacogant.simulation`` and ``alphacogant.free_energy``.

The figure shows the negative-EFE value (= pragmatic + epistemic) for each
of the six actions across all cycles of a greedy trajectory from the
self-improving operating point. This makes visible the explore→exploit
transition: early cycles show high value on epistemic actions (Sensors, R&D,
Theta), while later cycles shift toward pragmatic actions (Investments,
Actuators) as the model freshens and epistemic value falls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.channels import ACTIONS  # noqa: E402
from alphacogant.generative_model import default_model  # noqa: E402
from alphacogant.operating_points import IMPROVING  # noqa: E402
from alphacogant.simulation import simulate_trajectory  # noqa: E402

HORIZON = 12


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
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
        ax.plot(cycle_idx, record.action, marker="*", color="black", markersize=12, zorder=5)

    ax.set_xticks(range(HORIZON))
    ax.set_xticklabels([f"{i}" for i in range(HORIZON)])
    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels(list(ACTIONS))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Action")
    ax.set_title("Marginal-return vector (negative EFE) over the greedy trajectory\n★ = selected action")

    # Add value annotations
    for action_idx in range(len(ACTIONS)):
        for cycle_idx in range(HORIZON):
            value = value_matrix[action_idx, cycle_idx]
            color = "white" if abs(value) > abs(value_matrix).max() * 0.6 else "black"
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

    out_path = out_fig / "marginal_return_heatmap.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
