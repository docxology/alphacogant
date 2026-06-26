#!/usr/bin/env python3
"""Thin orchestrator: alpha-decay vs refresh of the EWM-parameter belief Theta.

Illustrates the whitepaper's *alpha-decay* term — the residual freshness gap that
the t-RSI certificate charges when the firm neglects its model parameters Theta.
Starting from a fresh parameter belief ``Theta = (0.1, 0.9)`` (P(fresh) = 0.9), the
belief is rolled forward ``HORIZON`` cycles under two stationary policies, both read
directly off ``model.B["Theta"]``:

  * **hold** — never re-fund Theta; the hold transition column drives P(fresh) down
    toward its stale fixed point (the alpha-decay the certificate prices).
  * **fund_Theta** — refresh Theta every cycle; the fund transition column keeps the
    belief fresh (it returns toward / holds near P(fresh) = 1).

All transition math comes from ``src/alphacogant`` (the engine's ``model.B`` and the
``channel_index`` helper); this script only orchestrates the deterministic forward
roll and renders the two trajectories. No mocks; no global RNG seeding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.channels import action_index, channel_index  # noqa: E402
from alphacogant.generative_model import default_model  # noqa: E402

# Two-level Theta factor: index 0 = stale/weak, index 1 = fresh/strong.
FRESH_INDEX = 1
HORIZON = 12
# Fresh starting belief over Theta = (stale, fresh).
THETA_FRESH = np.array([0.1, 0.9], dtype=float)


def _roll_p_fresh(B_theta: np.ndarray, action: int, start: np.ndarray, horizon: int) -> np.ndarray:
    """Roll the Theta belief forward under a stationary action; return P(fresh) per cycle.

    ``B_theta`` is the engine's column-stochastic transition tensor with layout
    ``[s_next, s_current, action]``; applying ``B_theta[:, :, action] @ belief``
    advances one cycle. Returns an array of length ``horizon + 1`` (cycle 0 is the
    starting belief).
    """
    transition = B_theta[:, :, action]
    belief = start / start.sum()
    p_fresh = np.empty(horizon + 1, dtype=float)
    p_fresh[0] = belief[FRESH_INDEX]
    for cycle in range(1, horizon + 1):
        belief = transition @ belief
        p_fresh[cycle] = belief[FRESH_INDEX]
    return p_fresh


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    # All transition data comes from the engine's generative model.
    model = default_model()
    # channel_index validates Theta against the fixed factor order; B is keyed by name.
    _ = channel_index("Theta")
    B_theta = model.B["Theta"]

    hold = action_index("hold")
    fund = action_index("fund_Theta")

    cycles = np.arange(HORIZON + 1)
    p_fresh_hold = _roll_p_fresh(B_theta, hold, THETA_FRESH, HORIZON)
    p_fresh_fund = _roll_p_fresh(B_theta, fund, THETA_FRESH, HORIZON)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        cycles,
        p_fresh_fund,
        marker="o",
        color="#0f766e",
        linewidth=2.0,
        label="fund_Theta (refresh): stays fresh",
    )
    ax.plot(
        cycles,
        p_fresh_hold,
        marker="s",
        color="#b91c1c",
        linewidth=2.0,
        label="hold (neglect): alpha-decay",
    )
    ax.axhline(
        THETA_FRESH[FRESH_INDEX],
        color="#94a3b8",
        linestyle=":",
        linewidth=1.0,
        label="initial freshness",
    )

    gap = p_fresh_fund[-1] - p_fresh_hold[-1]
    ax.annotate(
        f"freshness gap at H={HORIZON}: {gap:.3f}",
        xy=(HORIZON, (p_fresh_fund[-1] + p_fresh_hold[-1]) / 2.0),
        xytext=(HORIZON - 5.4, (p_fresh_fund[-1] + p_fresh_hold[-1]) / 2.0),
        fontsize=9,
        color="#334155",
        arrowprops={"arrowstyle": "->", "color": "#334155", "linewidth": 1.0},
        va="center",
    )

    ax.set_xlabel("Cycle")
    ax.set_ylabel(r"$P(\Theta = \mathrm{fresh})$")
    ax.set_title("Alpha-decay vs refresh of the EWM-parameter belief $\\Theta$")
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(cycles)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", framealpha=0.95)
    fig.tight_layout()

    out_path = out_fig / "theta_decay.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
