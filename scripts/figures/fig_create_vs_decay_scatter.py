#!/usr/bin/env python3
"""Figure: create-rate vs decay-rate scatter across bootstrap perturbations.

The engine's paired sampler supplies both rates for each perturbed belief.
With create on x and decay on y, a positive margin lies BELOW y=x;
a negative margin lies ABOVE it. Region colors encode that raw event,
not a deployment decision or a certificate threshold.

Both regimes use the canonical n, concentration and seed, with the seed
restarted for each regime, matching the manuscript's paired sample recipes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.generative_model import default_model  # noqa: E402
from alphacogant.model.operating_points import (  # noqa: E402
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COASTING,
    IMPROVING,
)
from alphacogant.trsi.t_rsi import DEFAULT_HORIZON, paired_bootstrap_samples  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    BOOTSTRAP_FOOTER,
    COASTING_COLOR,
    DPI,
    IMPROVING_COLOR,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    add_provenance_footer,
    apply_style,
)


def main() -> int:
    import matplotlib.pyplot as plt

    apply_style()
    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()
    n = BOOTSTRAP_N

    imp_create, imp_decay = paired_bootstrap_samples(
        model,
        IMPROVING,
        np.random.default_rng(BOOTSTRAP_SEED),
        n=n,
        horizon=DEFAULT_HORIZON,
        concentration=BOOTSTRAP_CONCENTRATION,
    )
    coast_create, coast_decay = paired_bootstrap_samples(
        model,
        COASTING,
        np.random.default_rng(BOOTSTRAP_SEED),
        n=n,
        horizon=DEFAULT_HORIZON,
        concentration=BOOTSTRAP_CONCENTRATION,
    )

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Break-even diagonal
    all_vals = np.concatenate([imp_create, imp_decay, coast_create, coast_decay])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = 0.05 * (hi - lo)
    lims = [lo - pad, hi + pad]
    ax.plot(
        lims,
        lims,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="break-even (create = decay)",
    )
    ax.fill_between(lims, lims, [lims[1], lims[1]], alpha=0.05, color=NEGATIVE_COLOR)
    ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=0.05, color=POSITIVE_COLOR)
    ax.text(
        0.03,
        0.80,
        "negative margin\n(decay > create; y > x)",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        color=NEGATIVE_COLOR,
        alpha=0.7,
    )
    ax.text(
        0.97,
        0.03,
        "positive margin\n(create > decay; y < x)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        color=POSITIVE_COLOR,
        alpha=0.7,
    )

    # Scatter
    ax.scatter(
        imp_create,
        imp_decay,
        s=25,
        marker="o",
        alpha=0.5,
        color=IMPROVING_COLOR,
        edgecolors="black",
        linewidths=0.3,
        label=f"Improving (n={n})",
    )
    ax.scatter(
        coast_create,
        coast_decay,
        s=25,
        marker="^",
        alpha=0.5,
        color=COASTING_COLOR,
        edgecolors="black",
        linewidths=0.3,
        label=f"Coasting (n={n})",
    )

    # Mean markers
    ax.scatter(
        [np.mean(imp_create)],
        [np.mean(imp_decay)],
        marker="X",
        s=150,
        color=IMPROVING_COLOR,
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
    ax.scatter(
        [np.mean(coast_create)],
        [np.mean(coast_decay)],
        marker="P",
        s=150,
        color=COASTING_COLOR,
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )

    imp_cm = float(np.mean(imp_create))
    imp_dm = float(np.mean(imp_decay))
    ax.annotate(
        f"Improving mean\n({imp_cm:.3f}, {imp_dm:.3f})",
        (imp_cm, imp_dm),
        xytext=(imp_cm + 0.03, imp_dm - 0.05),
        fontsize=7,
        color=IMPROVING_COLOR,
        arrowprops={"arrowstyle": "->", "color": IMPROVING_COLOR, "lw": 1},
    )
    coast_cm = float(np.mean(coast_create))
    coast_dm = float(np.mean(coast_decay))
    ax.annotate(
        f"Coasting mean\n({coast_cm:.3f}, {coast_dm:.3f})",
        (coast_cm, coast_dm),
        xytext=(coast_cm + 0.14, coast_dm - 0.03),
        fontsize=7,
        color=COASTING_COLOR,
        arrowprops={"arrowstyle": "->", "color": COASTING_COLOR, "lw": 1},
    )

    ax.set_xlabel("create-rate (pragmatic value of greedy policy over hold)")
    ax.set_ylabel("decay-rate (residual Theta-staleness erosion)")
    ax.set_title(
        "Bootstrap create vs decay scatter at two operating points\n"
        "one point = one paired belief draw; X = Improving mean, P = Coasting mean\n"
        f"n={n} per regime; seed={BOOTSTRAP_SEED}; "
        f"concentration={BOOTSTRAP_CONCENTRATION:g}; horizon={DEFAULT_HORIZON}"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()
    add_provenance_footer(fig, BOOTSTRAP_FOOTER)  # Dirichlet bootstrap perturbations

    out_path = out_fig / "create_vs_decay_scatter.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
