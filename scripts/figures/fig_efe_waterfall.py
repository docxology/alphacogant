#!/usr/bin/env python3
"""Figure: EFE decomposition waterfall for the funded channel.

Thin-orchestrator figure script. All engine math comes from
``src/alphacogant.free_energy`` and ``src/alphacogant.statistics``.

Shows the EFE decomposition (pragmatic cost + epistemic value = total G)
as a waterfall chart for all six actions at the IMPROVING operating point.
The waterfall makes visible how each action's total G is built from its
pragmatic and epistemic components, and why the greedy policy selects the
action it does (lowest G = highest value = highest pragmatic + epistemic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.efe.free_energy import expected_free_energy, marginal_return_vector  # noqa: E402
from alphacogant.model.channels import ACTIONS  # noqa: E402
from alphacogant.model.generative_model import default_model  # noqa: E402
from alphacogant.model.operating_points import IMPROVING  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    EPISTEMIC_COLOR,
    PRAGMATIC_COLOR,
    apply_style,
)


def main() -> int:
    import matplotlib.pyplot as plt

    apply_style()
    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()
    belief = IMPROVING

    # Compute EFE for all actions
    prags = []
    epis = []
    totals = []
    for action in range(len(ACTIONS)):
        efe = expected_free_energy(model, belief, action)
        prags.append(efe.pragmatic)
        epis.append(efe.epistemic)
        totals.append(efe.total)

    returns = marginal_return_vector(model, belief)
    greedy_action = max(returns, key=lambda a: returns[a])

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    x = np.arange(len(ACTIONS))
    w = 0.6

    # Draw pragmatic bars (negative — costs)
    ax.bar(
        x,
        prags,
        w,
        color=PRAGMATIC_COLOR,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        label="pragmatic (expected log-preference)",
    )

    # Draw epistemic bars on top of pragmatic
    # total = -(pragmatic + epistemic), so G = -pragmatic - epistemic
    # We draw: pragmatic (negative) then epistemic (positive, reduces total)
    ax.bar(
        x,
        epis,
        w,
        bottom=prags,
        color=EPISTEMIC_COLOR,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        label="epistemic (info gain, >=0)",
    )

    # Draw total markers
    ax.scatter(
        x,
        totals,
        color="black",
        s=60,
        zorder=5,
        marker="D",
        label="G (total EFE, minimized)",
    )

    # Highlight the greedy action
    ax.bar(x[greedy_action], 0, w, color="none", edgecolor="#16a34a", linewidth=3)
    ax.annotate(
        "GREEDY\n(lowest G)",
        (x[greedy_action], totals[greedy_action]),
        xytext=(x[greedy_action] + 0.5, totals[greedy_action] + 0.15),
        fontsize=8,
        fontweight="bold",
        color="#16a34a",
        arrowprops={"arrowstyle": "->", "color": "#16a34a", "lw": 1.5},
    )

    # Value annotations
    for i in range(len(ACTIONS)):
        ax.text(
            x[i],
            prags[i] - 0.08,
            f"{prags[i]:.3f}",
            ha="center",
            va="top",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
        ax.text(
            x[i],
            prags[i] + epis[i] + 0.02,
            f"+{epis[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=EPISTEMIC_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(ACTIONS), rotation=20, ha="right")
    ax.set_ylabel("EFE components (nats)")
    ax.set_title(
        "Expected Free Energy decomposition per action at the IMPROVING operating point\n"
        "black diamonds show G = -(pragmatic + epistemic); greedy selects lowest G"
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    out_path = out_fig / "efe_waterfall.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
