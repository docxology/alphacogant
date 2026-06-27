"""Statistical analysis: bootstrap confidence intervals, effect sizes, and regime comparisons.

This module wraps the engine's t-RSI machinery with higher-level statistical
reporting that the manuscript references: confidence intervals for create/decay
rates, Cohen's d effect sizes between regimes, and a structured regime
comparison table.  All computation comes from the engine; this module only
orchestrates and reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from alphacogant.efe.free_energy import expected_free_energy, marginal_return_vector
from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import (
    EconomicWorldModel,
    belief_prior,
    default_model,
)
from alphacogant.model.operating_points import (
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COASTING,
    IMPROVING,
)
from alphacogant.trsi.t_rsi import (
    DEFAULT_HORIZON,
    paired_bootstrap_samples,
    t_rsi,
)


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for a rate."""

    mean: float
    std: float
    se: float
    ci_lower: float
    ci_upper: float
    n: int
    confidence: float

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower


@dataclass(frozen=True)
class BreakEvenProfile:
    probability: float
    margin_mean: float
    margin_std: float
    margin_ci_lower: float
    margin_ci_upper: float
    create_mean: float
    decay_mean: float
    n: int
    horizon: int
    concentration: float
    confidence: float

    @property
    def margin_ci_width(self) -> float:
        return self.margin_ci_upper - self.margin_ci_lower

    def as_dict(self) -> dict[str, float | int]:
        return {
            "probability": self.probability,
            "margin_mean": self.margin_mean,
            "margin_std": self.margin_std,
            "margin_ci_lower": self.margin_ci_lower,
            "margin_ci_upper": self.margin_ci_upper,
            "create_mean": self.create_mean,
            "decay_mean": self.decay_mean,
            "n": self.n,
            "horizon": self.horizon,
            "concentration": self.concentration,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RegimeStatistics:
    """Complete statistical profile of one operating regime."""

    name: str
    belief: dict[str, np.ndarray]
    t_rsi: float
    create_ci: BootstrapCI
    decay_ci: BootstrapCI
    efe_pragmatic_per_action: dict[str, float]
    efe_epistemic_per_action: dict[str, float]
    funded_channel: str
    funded_pragmatic: float
    funded_epistemic: float
    exploration_ratio: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "t_rsi": self.t_rsi,
            "create_mean": self.create_ci.mean,
            "create_ci": [self.create_ci.ci_lower, self.create_ci.ci_upper],
            "decay_mean": self.decay_ci.mean,
            "decay_ci": [self.decay_ci.ci_lower, self.decay_ci.ci_upper],
            "funded_channel": self.funded_channel,
            "funded_pragmatic": self.funded_pragmatic,
            "funded_epistemic": self.funded_epistemic,
            "exploration_ratio": self.exploration_ratio,
        }


@dataclass(frozen=True)
class RegimeComparison:
    """Side-by-side comparison of two regimes."""

    improving: RegimeStatistics
    coasting: RegimeStatistics

    @property
    def t_rsi_delta(self) -> float:
        return self.improving.t_rsi - self.coasting.t_rsi

    @property
    def create_rate_delta(self) -> float:
        return self.improving.create_ci.mean - self.coasting.create_ci.mean

    @property
    def decay_rate_delta(self) -> float:
        return self.improving.decay_ci.mean - self.coasting.decay_ci.mean

    @property
    def cohen_d_create(self) -> float:
        """Cohen's d effect size for the create-rate difference between regimes."""
        pooled_std = np.sqrt((self.improving.create_ci.std**2 + self.coasting.create_ci.std**2) / 2)
        if pooled_std == 0:
            return 0.0
        return float(self.create_rate_delta / pooled_std)

    @property
    def cohen_d_decay(self) -> float:
        """Cohen's d effect size for the decay-rate difference."""
        pooled_std = np.sqrt((self.improving.decay_ci.std**2 + self.coasting.decay_ci.std**2) / 2)
        if pooled_std == 0:
            return 0.0
        return float(self.decay_rate_delta / pooled_std)

    def as_table(self) -> str:
        """Render the comparison as a markdown table."""
        rows = [
            "| Metric | Improving | Coasting | Delta |",
            "|--------|-----------|----------|-------|",
            (
                f"| t-RSI | {self.improving.t_rsi:.4f} | "
                f"{self.coasting.t_rsi:.4f} | {self.t_rsi_delta:.4f} |"
            ),
            (
                f"| Create-rate mean | {self.improving.create_ci.mean:.4f} | "
                f"{self.coasting.create_ci.mean:.4f} | {self.create_rate_delta:.4f} |"
            ),
            (
                f"| Create-rate 95% CI | [{self.improving.create_ci.ci_lower:.4f}, "
                f"{self.improving.create_ci.ci_upper:.4f}] | "
                f"[{self.coasting.create_ci.ci_lower:.4f}, "
                f"{self.coasting.create_ci.ci_upper:.4f}] | — |"
            ),
            (
                f"| Decay-rate mean | {self.improving.decay_ci.mean:.4f} | "
                f"{self.coasting.decay_ci.mean:.4f} | {self.decay_rate_delta:.4f} |"
            ),
            (
                f"| Decay-rate 95% CI | [{self.improving.decay_ci.ci_lower:.4f}, "
                f"{self.improving.decay_ci.ci_upper:.4f}] | "
                f"[{self.coasting.decay_ci.ci_lower:.4f}, "
                f"{self.coasting.decay_ci.ci_upper:.4f}] | — |"
            ),
            f"| Cohen's d (create) | — | — | {self.cohen_d_create:.4f} |",
            f"| Cohen's d (decay) | — | — | {self.cohen_d_decay:.4f} |",
            (
                f"| Funded channel | {self.improving.funded_channel} | "
                f"{self.coasting.funded_channel} | — |"
            ),
            (
                f"| Exploration ratio | {self.improving.exploration_ratio:.4f} | "
                f"{self.coasting.exploration_ratio:.4f} | — |"
            ),
        ]
        return "\n".join(rows)


def bootstrap_ci(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Compute a bootstrap confidence interval from a sample array."""
    if samples.size < 2:
        raise ValueError("Need at least 2 samples for a confidence interval.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    se = std / np.sqrt(samples.size)
    alpha = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(samples, 100 * alpha))
    ci_upper = float(np.percentile(samples, 100 * (1 - alpha)))
    return BootstrapCI(
        mean=mean,
        std=std,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=samples.size,
        confidence=confidence,
    )


def break_even_profile(
    model: EconomicWorldModel | None = None,
    belief: Mapping[str, np.ndarray] | None = None,
    *,
    seed: int = BOOTSTRAP_SEED,
    n: int = BOOTSTRAP_N,
    horizon: int = DEFAULT_HORIZON,
    concentration: float = BOOTSTRAP_CONCENTRATION,
    confidence: float = 0.95,
) -> BreakEvenProfile:
    if n < 2:
        raise ValueError("Need at least 2 samples for a break-even profile.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")

    model = model or default_model()
    if belief is None:
        belief = belief_prior(model)

    create_samples, decay_samples = paired_bootstrap_samples(
        model,
        belief,
        np.random.default_rng(seed),
        n,
        horizon,
        concentration,
    )
    margins = create_samples - decay_samples
    alpha = (1.0 - confidence) / 2.0
    return BreakEvenProfile(
        probability=float(np.mean(margins > 0.0)),
        margin_mean=float(np.mean(margins)),
        margin_std=float(np.std(margins, ddof=1)),
        margin_ci_lower=float(np.percentile(margins, 100 * alpha)),
        margin_ci_upper=float(np.percentile(margins, 100 * (1 - alpha))),
        create_mean=float(np.mean(create_samples)),
        decay_mean=float(np.mean(decay_samples)),
        n=n,
        horizon=horizon,
        concentration=concentration,
        confidence=confidence,
    )


def compute_regime_statistics(
    model: EconomicWorldModel | None = None,
    belief: Mapping[str, np.ndarray] | None = None,
    name: str = "operating_point",
    *,
    seed: int = BOOTSTRAP_SEED,
    n: int = BOOTSTRAP_N,
    horizon: int = DEFAULT_HORIZON,
    concentration: float = BOOTSTRAP_CONCENTRATION,
    confidence: float = 0.95,
) -> RegimeStatistics:
    """Compute the full statistical profile of one operating regime."""
    model = model or default_model()
    if belief is None:
        belief = belief_prior(model)

    # One paired bootstrap (same seed, same draw order) yields the t-RSI statistic
    # and the create/decay CIs from a single deterministic sample set — identical to
    # the previous three separate same-seed bootstraps, at a third of the cost.
    create_samples, decay_samples = paired_bootstrap_samples(
        model, belief, np.random.default_rng(seed), n, horizon, concentration
    )
    boot = {"t_rsi": float(t_rsi(create_samples, decay_samples))}

    create_ci = bootstrap_ci(create_samples, confidence)
    decay_ci = bootstrap_ci(decay_samples, confidence)

    # EFE per action
    returns = marginal_return_vector(model, belief)
    funded_action = max(returns, key=lambda a: returns[a])
    funded_result = expected_free_energy(model, belief, funded_action)

    from alphacogant.model.channels import ACTIONS

    efe_prag: dict[str, float] = {}
    efe_epis: dict[str, float] = {}
    for action_idx in range(len(ACTIONS)):
        efe = expected_free_energy(model, belief, action_idx)
        efe_prag[ACTIONS[action_idx]] = efe.pragmatic
        efe_epis[ACTIONS[action_idx]] = efe.epistemic

    # Exploration ratio: the fraction of greedy-trajectory CYCLES this regime spends
    # funding an epistemic channel (S or Z) — a behavioral, regime-discriminating
    # quantity, not the structural constant (epistemic actions / total actions).
    from alphacogant.stats.simulation import simulate_trajectory, summarize_trajectory

    trajectory = simulate_trajectory(model, belief, horizon=horizon, policy="greedy")
    exploration_ratio = float(summarize_trajectory(trajectory)["exploration_ratio"])

    return RegimeStatistics(
        name=name,
        belief={c: belief[c].copy() for c in CHANNELS},
        t_rsi=boot["t_rsi"],
        create_ci=create_ci,
        decay_ci=decay_ci,
        efe_pragmatic_per_action=efe_prag,
        efe_epistemic_per_action=efe_epis,
        funded_channel=ACTIONS[funded_action],
        funded_pragmatic=funded_result.pragmatic,
        funded_epistemic=funded_result.epistemic,
        exploration_ratio=exploration_ratio,
    )


def compare_regimes(
    model: EconomicWorldModel | None = None,
    *,
    seed: int = BOOTSTRAP_SEED,
    n: int = BOOTSTRAP_N,
    horizon: int = DEFAULT_HORIZON,
    concentration: float = BOOTSTRAP_CONCENTRATION,
) -> RegimeComparison:
    """Compare the IMPROVING and COASTING regimes side by side."""
    model = model or default_model()
    improving = compute_regime_statistics(
        model,
        IMPROVING,
        "Improving",
        seed=seed,
        n=n,
        horizon=horizon,
        concentration=concentration,
    )
    coasting = compute_regime_statistics(
        model,
        COASTING,
        "Coasting",
        seed=seed,
        n=n,
        horizon=horizon,
        concentration=concentration,
    )
    return RegimeComparison(improving=improving, coasting=coasting)


__all__ = [
    "BreakEvenProfile",
    "BootstrapCI",
    "RegimeComparison",
    "RegimeStatistics",
    "break_even_profile",
    "bootstrap_ci",
    "compare_regimes",
    "compute_regime_statistics",
]
