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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.generative_model import default_model  # noqa: E402
from alphacogant.operating_points import IMPROVING  # noqa: E402
from alphacogant.sensitivity import sweep_concentration, sweep_theta_freshness  # noqa: E402


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_fig = PROJECT_ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()

    conc_result = sweep_concentration(model, IMPROVING, n=128)
    theta_result = sweep_theta_freshness(model, IMPROVING, n=128)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Panel (a): t-RSI vs concentration
    concentrations = conc_result["concentrations"]
    t_rsi_conc = conc_result["t_rsi"]
    ax1.plot(concentrations, t_rsi_conc, marker="o", color="#7c3aed", linewidth=2.0)
    ax1.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0, label="break-even")
    ax1.set_xlabel("Dirichlet concentration (belief precision)")
    ax1.set_ylabel("t-RSI (standardized units)")
    ax1.set_title("(a) t-RSI vs belief precision")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Panel (b): t-RSI vs Theta freshness
    theta_vals = theta_result["theta_p_fresh"]
    t_rsi_theta = theta_result["t_rsi"]
    create_means = theta_result["create_mean"]
    decay_means = theta_result["decay_mean"]
    ax2.plot(theta_vals, t_rsi_theta, marker="s", color="#0f766e", linewidth=2.0, label="t-RSI")
    ax2.plot(theta_vals, create_means, marker="^", color="#2563eb", linewidth=1.5, alpha=0.7, label="create-rate mean")
    ax2.plot(theta_vals, decay_means, marker="v", color="#b91c1c", linewidth=1.5, alpha=0.7, label="decay-rate mean")
    ax2.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0)
    ax2.set_xlabel(r"$P(\Theta = \mathrm{fresh})$ prior")
    ax2.set_ylabel("Rate / t-RSI")
    ax2.set_title(r"(b) t-RSI vs $\Theta$-freshness prior")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)

    fig.suptitle("Sensitivity of the t-RSI certificate to belief precision and parameter freshness", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = out_fig / "trsi_sensitivity.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
