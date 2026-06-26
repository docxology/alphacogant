#!/usr/bin/env python3
"""Figure: regime comparison with bootstrap confidence intervals.

Thin-orchestrator figure script. All statistical computation comes from
``src/alphacogant.statistics`` (which wraps the engine's t-RSI bootstrap).

Two-panel figure:
  (a) Create-rate vs decay-rate for IMPROVING and COASTING, with 95% bootstrap
      CIs shown as error bars. The t-RSI is the standardized gap between the
      two rates within each regime.
  (b) EFE decomposition (pragmatic + epistemic) for the funded channel at
      each regime, showing the explore/exploit split.

This is the key statistical evidence figure: it shows the CIs, the effect
sizes, and the sign-discrimination property in one view.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.plot_style import (  # noqa: E402
    apply_style,
    COASTING_COLOR,
    CREATE_COLOR,
    DECAY_COLOR,
    EPISTEMIC_COLOR,
    IMPROVING_COLOR,
    PRAGMATIC_COLOR,
)

from alphacogant.statistics import compare_regimes  # noqa: E402


def main() -> int:
    import matplotlib.pyplot as plt

    apply_style()
    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    comparison = compare_regimes(n=128)

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

    ax1.bar(x - w/2, create_means, w, yerr=create_errs, color=CREATE_COLOR,
            edgecolor="black", linewidth=0.8, capsize=5,
            label="create-rate", alpha=0.85)
    ax1.bar(x + w/2, decay_means, w, yerr=decay_errs, color=DECAY_COLOR,
            edgecolor="black", linewidth=0.8, capsize=5,
            label="decay-rate", alpha=0.85)

    # Annotate t-RSI values
    for i, reg in enumerate(regimes):
        ax1.text(
            x[i], max(create_means[i], decay_means[i]) + 0.08,
            f"t-RSI = {reg.t_rsi:.2f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=regime_colors[i],
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_names)
    ax1.set_ylabel("Rate (nats / cycle)")
    ax1.set_title("(a) Create vs decay rates with 95% bootstrap CIs")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.legend(fontsize=8)
    ax1.set_xlim(-0.6, 1.6)

    # ── Panel (b): EFE decomposition for funded channel ──
    funded_labels = [r.funded_channel for r in regimes]
    prags = [r.funded_pragmatic for r in regimes]
    epis = [r.funded_epistemic for r in regimes]

    ax2.bar(x - w/2, prags, w, color=PRAGMATIC_COLOR, edgecolor="black",
            linewidth=0.8, alpha=0.85, label="pragmatic value")
    ax2.bar(x + w/2, epis, w, color=EPISTEMIC_COLOR, edgecolor="black",
            linewidth=0.8, alpha=0.85, label="epistemic value")

    for i, reg in enumerate(regimes):
        ax2.text(
            x[i] - w/2, prags[i] - 0.15, f"{prags[i]:.3f}",
            ha="center", va="top", fontsize=7, color="white", fontweight="bold",
        )
        ax2.text(
            x[i] + w/2, epis[i] + 0.01, f"{epis[i]:.3f}",
            ha="center", va="bottom", fontsize=7, color="black",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{name}\nfunds {label}" for name, label in zip(regime_names, funded_labels)])
    ax2.set_ylabel("Value (nats)")
    ax2.set_title("(b) EFE decomposition of the funded channel")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Regime comparison: bootstrap confidence intervals and EFE decomposition",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = out_fig / "regime_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
