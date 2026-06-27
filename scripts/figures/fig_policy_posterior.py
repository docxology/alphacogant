#!/usr/bin/env python3
"""Figure: policy posterior evolution across the greedy trajectory.

Thin-orchestrator figure script. All engine math comes from
``src/alphacogant.simulation`` and ``alphacogant.free_energy.policy_posterior``.

Shows the precision-weighted softmax posterior over the six actions at each
cycle of the greedy trajectory from the self-improving operating point.
Early cycles concentrate probability on epistemic actions (Sensors, R&D,
Theta); as the model freshens, probability mass shifts to pragmatic actions
(Investments, Actuators). The explore→exploit transition is visible as a
redistribution of probability mass, not a hard switch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.efe.free_energy import policy_posterior  # noqa: E402
from alphacogant.model.channels import ACTIONS  # noqa: E402
from alphacogant.model.generative_model import default_model  # noqa: E402
from alphacogant.model.operating_points import IMPROVING  # noqa: E402
from alphacogant.stats.simulation import simulate_trajectory  # noqa: E402

HORIZON = 12
GAMMA = 1.0  # precision for the softmax


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()
    trajectory = simulate_trajectory(model, IMPROVING, horizon=HORIZON, policy="greedy")

    # Build the posterior matrix: [cycle, action]
    posteriors = np.array(
        [policy_posterior(model, record.belief, gamma=GAMMA) for record in trajectory.cycles]
    )  # shape (HORIZON, 6)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    bottom = np.zeros(HORIZON)
    colors = ["#b91c1c", "#2563eb", "#ea580c", "#7c3aed", "#0891b2", "#64748b"]
    for action_idx in range(len(ACTIONS)):
        ax.bar(
            range(HORIZON),
            posteriors[:, action_idx],
            bottom=bottom,
            color=colors[action_idx],
            label=ACTIONS[action_idx],
            width=0.8,
        )
        bottom += posteriors[:, action_idx]

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Policy posterior probability")
    ax.set_title(
        f"Policy posterior evolution (γ={GAMMA}) — explore→exploit transition\n"
        "under the greedy trajectory from the self-improving operating point"
    )
    ax.set_xticks(range(HORIZON))
    ax.set_xlim(-0.5, HORIZON - 0.5)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()

    out_path = out_fig / "policy_posterior.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
