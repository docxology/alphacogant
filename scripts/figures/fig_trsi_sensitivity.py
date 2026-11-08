#!/usr/bin/env python3
"""Figure: t-RSI sensitivity to belief precision and Theta freshness.

Engine sweeps use synthetic Dirichlet perturbations. Concentration changes
both dispersion and center because alpha = concentration * belief + 1.
The dimensionless statistic and the rate means occupy separate panels;
no rate standard error is presented as uncertainty of t-RSI.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.generative_model import default_model  # noqa: E402
from alphacogant.model.operating_points import (  # noqa: E402
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_SEED,
    IMPROVING,
)
from alphacogant.stats.sensitivity import sweep_concentration, sweep_theta_freshness  # noqa: E402
from alphacogant.trsi.t_rsi import DEFAULT_HORIZON  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    BOOTSTRAP_FOOTER,
    CREATE_COLOR,
    DECAY_COLOR,
    DPI,
    EPISTEMIC_COLOR,
    add_provenance_footer,
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
    n = 128
    conc_result = sweep_concentration(model, IMPROVING, seed=BOOTSTRAP_SEED, n=n)
    theta_result = sweep_theta_freshness(
        model, IMPROVING, seed=BOOTSTRAP_SEED, n=n,
        concentration=BOOTSTRAP_CONCENTRATION,
    )

    fig, (ax1, ax2, ax_rate) = plt.subplots(1, 3, figsize=(15.0, 5.2))

    # Panel (a): dimensionless t-RSI vs concentration, without an invalid SE band.
    concentrations = conc_result["concentrations"]
    t_rsi_conc = conc_result["t_rsi"]
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
    ax1.set_xlabel("Dirichlet concentration (center and dispersion)")
    ax1.set_ylabel("t-RSI (dimensionless)")
    ax1.set_title("(a) t-RSI vs concentration")
    ax1.set_xscale("log")
    ax1.set_xticks(concentrations, labels=[f"{value:g}" for value in concentrations])
    ax1.minorticks_off()
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
    ax_rate.plot(
        theta_vals,
        create_means,
        marker="^",
        color=CREATE_COLOR,
        linewidth=1.8,
        alpha=0.8,
        label="create-rate mean",
    )
    ax_rate.plot(
        theta_vals,
        decay_means,
        marker="v",
        color=DECAY_COLOR,
        linewidth=1.8,
        alpha=0.8,
        label="decay-rate mean",
    )
    ax_rate.fill_between(
        theta_vals,
        create_means,
        decay_means,
        where=(create_means > decay_means),
        alpha=0.10,
        color=CREATE_COLOR,
        label="create > decay",
    )
    ax_rate.fill_between(
        theta_vals,
        create_means,
        decay_means,
        where=(create_means <= decay_means),
        alpha=0.10,
        color=DECAY_COLOR,
        label="decay >= create",
    )
    ax2.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax2.set_xlabel(r"$P(\Theta = \mathrm{fresh})$ prior")
    ax2.set_ylabel("t-RSI (dimensionless)")
    ax2.set_title("(b) t-RSI vs Theta freshness")
    ax2.legend(fontsize=7)
    ax_rate.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax_rate.set_xlabel(r"$P(\Theta = \mathrm{fresh})$ prior")
    ax_rate.set_ylabel("Rate mean (nats / cycle)")
    ax_rate.set_title("(c) Rate means vs Theta freshness")
    ax_rate.legend(fontsize=7)

    fig.suptitle(
        "Synthetic perturbation sensitivity (not t-RSI confidence bands)\n"
        f"n={n} per point; seed={BOOTSTRAP_SEED} restarted per point; "
        f"horizon={DEFAULT_HORIZON}; freshness-sweep concentration={BOOTSTRAP_CONCENTRATION:g}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    add_provenance_footer(fig, BOOTSTRAP_FOOTER)  # bootstrap sweeps (n=128)

    out_path = out_fig / "trsi_sensitivity.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
