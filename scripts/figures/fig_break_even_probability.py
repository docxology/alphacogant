#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.model.generative_model import default_model, validate_belief_map  # noqa: E402
from alphacogant.model.operating_points import (  # noqa: E402
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_SEED,
    IMPROVING,
)
from alphacogant.stats.statistics import break_even_profile  # noqa: E402
from alphacogant.viz.plot_style import (  # noqa: E402
    DECAY_COLOR,
    EPISTEMIC_COLOR,
    POSITIVE_COLOR,
    apply_style,
)


def belief_with_theta_freshness(p_fresh: float) -> dict[str, np.ndarray]:
    belief = {channel: vector.copy() for channel, vector in IMPROVING.items()}
    belief["Theta"] = np.array([1.0 - p_fresh, p_fresh], dtype=float)
    return validate_belief_map(belief, context="theta_freshness_sweep")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    output_dir = PROJECT_ROOT / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = default_model()
    theta_values = np.linspace(0.05, 0.95, 10)
    profiles = [
        break_even_profile(
            model,
            belief_with_theta_freshness(float(theta_value)),
            seed=BOOTSTRAP_SEED + index,
            n=96,
            concentration=BOOTSTRAP_CONCENTRATION,
        )
        for index, theta_value in enumerate(theta_values)
    ]

    probabilities = np.array([profile.probability for profile in profiles], dtype=float)
    margin_means = np.array([profile.margin_mean for profile in profiles], dtype=float)
    margin_lowers = np.array([profile.margin_ci_lower for profile in profiles], dtype=float)
    margin_uppers = np.array([profile.margin_ci_upper for profile in profiles], dtype=float)

    fig, (ax_prob, ax_margin) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    ax_prob.plot(
        theta_values,
        probabilities,
        marker="o",
        color=EPISTEMIC_COLOR,
        linewidth=2.4,
        label="P(create > decay)",
    )
    ax_prob.axhline(0.5, color="#64748b", linestyle="--", linewidth=1.2, label="coin-flip line")
    ax_prob.fill_between(
        theta_values,
        probabilities,
        0.5,
        where=probabilities >= 0.5,
        color=POSITIVE_COLOR,
        alpha=0.12,
    )
    ax_prob.fill_between(
        theta_values,
        probabilities,
        0.5,
        where=probabilities < 0.5,
        color=DECAY_COLOR,
        alpha=0.12,
    )
    ax_prob.set_xlabel(r"$P(\Theta=\mathrm{fresh})$ prior")
    ax_prob.set_ylabel("paired bootstrap probability")
    ax_prob.set_title("(a) Break-even event probability")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.legend(loc="best", fontsize=8)

    ax_margin.plot(
        theta_values,
        margin_means,
        marker="s",
        color=EPISTEMIC_COLOR,
        linewidth=2.4,
        label="mean(create - decay)",
    )
    ax_margin.fill_between(
        theta_values,
        margin_lowers,
        margin_uppers,
        color=EPISTEMIC_COLOR,
        alpha=0.16,
        label="95% margin interval",
    )
    ax_margin.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.2)
    ax_margin.set_xlabel(r"$P(\Theta=\mathrm{fresh})$ prior")
    ax_margin.set_ylabel("rate margin (nats / cycle)")
    ax_margin.set_title("(b) Paired create-minus-decay margin")
    ax_margin.legend(loc="best", fontsize=8)

    fig.suptitle("Break-even robustness across parameter-freshness beliefs", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = output_dir / "break_even_probability.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(str(output_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
