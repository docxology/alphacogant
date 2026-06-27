#!/usr/bin/env python3
"""Figure: create-rate vs decay-rate scatter across bootstrap perturbations.

Thin-orchestrator figure script. All computation comes from the engine's
t_rsi module (bootstrap_t_rsi, create_rate, decay_rate).

Each point is one bootstrap perturbation of the operating belief. The
diagonal y=x is the break-even line (create = decay). Points above it
(self-improvement) are green; points below (bleeding) are red. The
standardized distance of the cloud from the diagonal is the t-RSI.

Two operating points are shown: IMPROVING (stale, should self-improve at
the point estimate) and COASTING (fresh, should not). The cloud locations
and their overlap with the break-even line make the honesty of the
comparator visible: the IMPROVING cloud is not entirely above the line,
and the COASTING cloud is not entirely below — the bootstrap uncertainty
makes the sign-discrimination imperfect at this model resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.generative_model import default_model, validate_belief_map  # noqa: E402
from alphacogant.model.operating_points import (  # noqa: E402
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COASTING,
    IMPROVING,
)
from alphacogant.trsi.t_rsi import create_rate, decay_rate  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    COASTING_COLOR,
    IMPROVING_COLOR,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    apply_style,
)


def _scatter_samples(model, belief, seed, n, horizon=12, concentration=12.0):
    """Return (create_samples, decay_samples) arrays from Dirichlet perturbations."""
    rng = np.random.default_rng(seed)
    normalized = validate_belief_map(belief, context="belief")
    creates = np.empty(n, dtype=float)
    decays = np.empty(n, dtype=float)
    for i in range(n):
        perturbed: dict[str, np.ndarray] = {}
        for channel, vector in normalized.items():
            alpha = np.clip(vector, 1e-9, None) * concentration + 1.0
            perturbed[channel] = rng.dirichlet(alpha)
        creates[i] = create_rate(model, perturbed, horizon)
        decays[i] = decay_rate(model, perturbed, horizon)
    return creates, decays


def main() -> int:
    import matplotlib.pyplot as plt

    apply_style()
    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()
    n = BOOTSTRAP_N

    imp_create, imp_decay = _scatter_samples(
        model,
        IMPROVING,
        BOOTSTRAP_SEED,
        n,
        concentration=BOOTSTRAP_CONCENTRATION,
    )
    coast_create, coast_decay = _scatter_samples(
        model,
        COASTING,
        BOOTSTRAP_SEED + 1,
        n,
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
    ax.fill_between(lims, lims, [lims[1], lims[1]], alpha=0.05, color=POSITIVE_COLOR)
    ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=0.05, color=NEGATIVE_COLOR)
    ax.text(
        0.03,
        0.97,
        "self-improvement\n(create > decay)",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        color=POSITIVE_COLOR,
        alpha=0.7,
    )
    ax.text(
        0.97,
        0.03,
        "bleeding\n(decay > create)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        color=NEGATIVE_COLOR,
        alpha=0.7,
    )

    # Scatter
    ax.scatter(
        imp_create,
        imp_decay,
        s=25,
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
        marker="X",
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
        xytext=(coast_cm + 0.03, coast_dm + 0.05),
        fontsize=7,
        color=COASTING_COLOR,
        arrowprops={"arrowstyle": "->", "color": COASTING_COLOR, "lw": 1},
    )

    ax.set_xlabel("create-rate (pragmatic value of greedy policy over hold)")
    ax.set_ylabel("decay-rate (residual Theta-staleness erosion)")
    ax.set_title(
        "Bootstrap create vs decay scatter at two operating points\n"
        "each point = one Dirichlet-perturbed belief; X = mean; standardized gap = t-RSI"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()

    out_path = out_fig / "create_vs_decay_scatter.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
