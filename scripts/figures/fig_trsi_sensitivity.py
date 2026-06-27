#!/usr/bin/env python3
"""Figure: t-RSI sensitivity to belief precision and Theta freshness.

Thin-orchestrator figure script. All engine math comes from
``src/alphacogant.sensitivity`` (which wraps ``t_rsi.bootstrap_t_rsi``).

Two-panel figure:
  (a) t-RSI vs Dirichlet concentration — shows how the headline number
      responds to the firm's belief precision. At low concentration the
      perturbations are wide and the standardized distance shrinks; at high
      concentration they collapse to a point mass and the distance inflates.
  (b) t-RSI vs Theta-freshness prior — shows how the certificate responds
      as the firm's belief about its own parameter quality moves from stale
      to fresh. This is the belief axis the certificate is designed to
      discriminate along.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.generative_model import default_model  # noqa: E402
from alphacogant.model.operating_points import IMPROVING  # noqa: E402
from alphacogant.stats.sensitivity import sweep_concentration, sweep_theta_freshness  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    CREATE_COLOR,
    DECAY_COLOR,
    EPISTEMIC_COLOR,
    apply_style,
)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()
    conc_result = sweep_concentration(model, IMPROVING, n=128)
    theta_result = sweep_theta_freshness(model, IMPROVING, n=128)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # Panel (a): t-RSI vs concentration with shaded regions
    concentrations = conc_result["concentrations"]
    t_rsi_conc = conc_result["t_rsi"]
    create_ses = conc_result["create_se"]
    decay_ses = conc_result["decay_se"]

    ax1.fill_between(
        concentrations,
        t_rsi_conc - create_ses,
        t_rsi_conc + decay_ses,
        alpha=0.15,
        color=EPISTEMIC_COLOR,
        label="±1 SE band",
    )
    ax1.plot(
        concentrations,
        t_rsi_conc,
        marker="o",
        color=EPISTEMIC_COLOR,
        linewidth=2.5,
        markersize=6,
        label="t-RSI",
        zorder=3,
    )
    ax1.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.2, label="break-even")
    ax1.set_xlabel("Dirichlet concentration (belief precision)")
    ax1.set_ylabel("t-RSI (standardized units)")
    ax1.set_title("(a) t-RSI vs belief precision")
    ax1.set_xscale("log")
    ax1.legend(fontsize=7, loc="lower right")

    # Panel (b): t-RSI vs Theta freshness
    theta_vals = theta_result["theta_p_fresh"]
    t_rsi_theta = theta_result["t_rsi"]
    create_means = theta_result["create_mean"]
    decay_means = theta_result["decay_mean"]
    ax2.plot(
        theta_vals,
        t_rsi_theta,
        marker="s",
        color=EPISTEMIC_COLOR,
        linewidth=2.5,
        markersize=6,
        label="t-RSI",
        zorder=3,
    )
    ax2.plot(
        theta_vals,
        create_means,
        marker="^",
        color=CREATE_COLOR,
        linewidth=1.8,
        alpha=0.8,
        label="create-rate mean",
    )
    ax2.plot(
        theta_vals,
        decay_means,
        marker="v",
        color=DECAY_COLOR,
        linewidth=1.8,
        alpha=0.8,
        label="decay-rate mean",
    )
    ax2.fill_between(
        theta_vals,
        create_means,
        decay_means,
        where=(create_means > decay_means),
        alpha=0.10,
        color=CREATE_COLOR,
        label="create > decay",
    )
    ax2.fill_between(
        theta_vals,
        create_means,
        decay_means,
        where=(create_means <= decay_means),
        alpha=0.10,
        color=DECAY_COLOR,
        label="decay > create",
    )
    ax2.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax2.set_xlabel(r"$P(\Theta = \mathrm{fresh})$ prior")
    ax2.set_ylabel("Rate / t-RSI")
    ax2.set_title(r"(b) t-RSI vs $\Theta$-freshness prior")
    ax2.legend(fontsize=7)

    fig.suptitle(
        "Sensitivity of the t-RSI certificate to belief precision and parameter freshness",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = out_fig / "trsi_sensitivity.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
