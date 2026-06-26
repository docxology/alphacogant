#!/usr/bin/env python3
"""Figure: multi-cycle belief trajectory under the greedy policy.

Thin-orchestrator figure script. All engine math comes from
``src/alphacogant.simulation`` (which wraps the generative model, EFE
evaluation, and transition machinery). This script only calls
``simulate_trajectory`` and renders the result.

Two-panel figure:
  (a) Self-improving start (stale Theta) — the greedy policy funds Theta
      until it converges to fresh, then holds. Beliefs move weak→strong
      for Theta; the funded action is epistemic then idle.
  (b) Fresh-Theta start (weak production channels) — the greedy policy
      funds Sensors then holds. Shows that with a fresh model, the
      controller explores briefly then exploits the now-accurate forecasts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.channels import ACTIONS, CHANNELS  # noqa: E402
from alphacogant.generative_model import default_model, validate_belief_map  # noqa: E402
from alphacogant.operating_points import IMPROVING  # noqa: E402
from alphacogant.simulation import simulate_trajectory  # noqa: E402

HORIZON = 12

# A second starting point: Theta fresh but production channels weak.
FRESH_THETA_RAW: dict[str, tuple[float, float]] = {
    "I": (0.80, 0.20),
    "S": (0.80, 0.20),
    "U": (0.80, 0.20),
    "Theta": (0.10, 0.90),
    "Z": (0.70, 0.30),
}

CHANNEL_COLORS = {
    "I": "#b91c1c",
    "S": "#2563eb",
    "U": "#ea580c",
    "Theta": "#7c3aed",
    "Z": "#0891b2",
}

CHANNEL_LABELS = {
    "I": "Investments",
    "S": "Sensors",
    "U": "Actuators",
    "Theta": "Parameters",
    "Z": "R&D",
}

ACTION_SHORT = {
    "fund_I": "I",
    "fund_S": "S",
    "fund_U": "U",
    "fund_Theta": "Θ",
    "fund_Z": "Z",
    "hold": "·",
}


def _draw_panel(ax, trajectory, title: str) -> None:
    cycles = np.arange(len(trajectory.cycles))
    for channel in CHANNELS:
        p_strong = trajectory.belief_history[channel]
        ax.plot(
            cycles,
            p_strong,
            marker="o",
            color=CHANNEL_COLORS[channel],
            linewidth=1.8,
            markersize=4,
            label=CHANNEL_LABELS[channel],
        )
    for record in trajectory.cycles:
        label = ACTION_SHORT.get(record.action_name, "?")
        ax.annotate(
            label,
            (record.cycle, 0.03),
            fontsize=7,
            ha="center",
            va="bottom",
            color="#475569",
            fontweight="bold",
        )
    ax.set_xlabel("Cycle")
    ax.set_ylabel(r"$P(\text{strong})$")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.5, len(trajectory.cycles) - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(cycles)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=7)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()
    traj_improving = simulate_trajectory(model, IMPROVING, horizon=HORIZON, policy="greedy")
    fresh_theta_belief = {
        channel: np.array(vals, dtype=float) for channel, vals in FRESH_THETA_RAW.items()
    }
    fresh_theta_belief = validate_belief_map(fresh_theta_belief, context="fresh_theta")
    traj_fresh = simulate_trajectory(model, fresh_theta_belief, horizon=HORIZON, policy="greedy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    _draw_panel(ax1, traj_improving, "(a) Self-improving start (stale Θ)")
    _draw_panel(ax2, traj_fresh, "(b) Fresh-Θ start (weak production)")

    fig.suptitle(
        "Belief trajectory under the greedy EFE policy — funded action per cycle shown at bottom",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = out_fig / "belief_trajectory.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
