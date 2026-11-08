#!/usr/bin/env python3
"""Figure: regime comparison with bootstrap confidence intervals.

Thin-orchestrator figure script. All statistical computation comes from
``src/alphacogant.statistics`` (which wraps the engine's t-RSI bootstrap).

Panel (a) shows marginal 95% percentile draw intervals and paired-margin
t-RSI at n=128 per regime, distinct from the canonical n=2560 headline.
These are intervals of synthetic rate draws, not uncertainty bands for t-RSI.
Panel (b) is the one-step EFE decomposition of the selected action, including hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.operating_points import (  # noqa: E402
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_SEED,
)
from alphacogant.stats.statistics import compare_regimes  # noqa: E402
from alphacogant.trsi.t_rsi import DEFAULT_HORIZON  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    BOOTSTRAP_FOOTER,
    COASTING_COLOR,
    CREATE_COLOR,
    DECAY_COLOR,
    DPI,
    EPISTEMIC_COLOR,
    IMPROVING_COLOR,
    PRAGMATIC_COLOR,
    add_provenance_footer,
    apply_style,
)


def main() -> int:
    import matplotlib.pyplot as plt

    apply_style()
    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    n = 128
    comparison = compare_regimes(
        seed=BOOTSTRAP_SEED, n=n, horizon=DEFAULT_HORIZON,
        concentration=BOOTSTRAP_CONCENTRATION,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5))

    # ── Panel (a): Create vs Decay with CIs ──
    regimes = [comparison.improving, comparison.coasting]
    regime_names = ["Improving", "Coasting"]
    regime_colors = [IMPROVING_COLOR, COASTING_COLOR]
    x = np.arange(len(regimes))
    w = 0.35

    create_means = [r.create_ci.mean for r in regimes]
    create_errs = [
        [r.create_ci.mean - r.create_ci.ci_lower for r in regimes],
        [r.create_ci.ci_upper - r.create_ci.mean for r in regimes],
    ]
    decay_means = [r.decay_ci.mean for r in regimes]
    decay_errs = [
        [r.decay_ci.mean - r.decay_ci.ci_lower for r in regimes],
        [r.decay_ci.ci_upper - r.decay_ci.mean for r in regimes],
    ]

    ax1.bar(
        x - w / 2,
        create_means,
        w,
        yerr=create_errs,
        color=CREATE_COLOR,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        label="create-rate",
        alpha=0.85,
    )
    ax1.bar(
        x + w / 2,
        decay_means,
        w,
        yerr=decay_errs,
        color=DECAY_COLOR,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        label="decay-rate",
        alpha=0.85,
    )

    # Annotate t-RSI values
    for i, reg in enumerate(regimes):
        ax1.text(
            x[i],
            0.97,
            f"t-RSI = {reg.t_rsi:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=regime_colors[i],
            transform=ax1.get_xaxis_transform(),
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_names)
    ax1.set_ylabel("Rate (nats / cycle)")
    ax1.set_title("(a) Rate means and 95% percentile draw intervals")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.legend(fontsize=8)
    ax1.set_xlim(-0.6, 1.6)

    # ── Panel (b): EFE decomposition for funded channel ──
    funded_labels = [r.funded_channel for r in regimes]
    prags = [r.funded_pragmatic for r in regimes]
    epis = [r.funded_epistemic for r in regimes]

    ax2.bar(
        x - w / 2,
        prags,
        w,
        color=PRAGMATIC_COLOR,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        label="pragmatic value",
    )
    ax2.bar(
        x + w / 2,
        epis,
        w,
        color=EPISTEMIC_COLOR,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        label="epistemic value",
    )

    for i, _reg in enumerate(regimes):
        ax2.text(
            x[i] - w / 2,
            prags[i] - 0.15,
            f"{prags[i]:.3f}",
            ha="center",
            va="top",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
        ax2.text(
            x[i] + w / 2,
            epis[i] + 0.01,
            f"{epis[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [
            f"{name}\nselects {label}"
            for name, label in zip(regime_names, funded_labels, strict=True)
        ]
    )
    ax2.set_ylabel("Value (nats)")
    ax2.set_title("(b) One-step EFE of the selected action")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Regime comparison: synthetic draw intervals and one-step EFE\n"
        f"n={n} per regime; seed={BOOTSTRAP_SEED} restarted per regime; "
        f"concentration={BOOTSTRAP_CONCENTRATION:g}; horizon={DEFAULT_HORIZON}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    add_provenance_footer(fig, BOOTSTRAP_FOOTER)  # 95% bootstrap CIs

    out_path = out_fig / "regime_comparison.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
