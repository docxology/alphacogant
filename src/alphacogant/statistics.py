"""Statistical analysis: bootstrap confidence intervals, effect sizes, and regime comparisons.

This module wraps the engine's t-RSI machinery with higher-level statistical
reporting that the manuscript references: confidence intervals for create/decay
rates, Cohen's d effect sizes between regimes, and a structured regime
comparison table.  All computation comes from the engine; this module only
orchestrates and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from alphacogant.channels import CHANNELS
from alphacogant.free_energy import expected_free_energy, marginal_return_vector
from alphacogant.generative_model import EconomicWorldModel, belief_prior, default_model
from alphacogant.operating_points import (
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COASTING,
    IMPROVING,
)
from alphacogant.t_rsi import (
    DEFAULT_HORIZON,
    bootstrap_t_rsi,
    create_rate,
    decay_rate,
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
        pooled_std = np.sqrt(
            (self.improving.create_ci.std**2 + self.coasting.create_ci.std**2) / 2
        )
        if pooled_std == 0:
            return 0.0
        return float(self.create_rate_delta / pooled_std)

    @property
    def cohen_d_decay(self) -> float:
        """Cohen's d effect size for the decay-rate difference."""
        pooled_std = np.sqrt(
            (self.improving.decay_ci.std**2 + self.coasting.decay_ci.std**2) / 2
        )
        if pooled_std == 0:
            return 0.0
        return float(self.decay_rate_delta / pooled_std)

    def as_table(self) -> str:
        """Render the comparison as a markdown table."""
        return (
            f"| Metric | Improving | Coasting | Delta |\n"
            f"|--------|-----------|----------|-------|\n"
            f"| t-RSI | {self.improving.t_rsi:.4f} | {self.coasting.t_rsi:.4f} | {self.t_rsi_delta:.4f} |\n"
            f"| Create-rate mean | {self.improving.create_ci.mean:.4f} | {self.coasting.create_ci.mean:.4f} | {self.create_rate_delta:.4f} |\n"
            f"| Create-rate 95% CI | [{self.improving.create_ci.ci_lower:.4f}, {self.improving.create_ci.ci_upper:.4f}] | [{self.coasting.create_ci.ci_lower:.4f}, {self.coasting.create_ci.ci_upper:.4f}] | — |\n"
            f"| Decay-rate mean | {self.improving.decay_ci.mean:.4f} | {self.coasting.decay_ci.mean:.4f} | {self.decay_rate_delta:.4f} |\n"
            f"| Decay-rate 95% CI | [{self.improving.decay_ci.ci_lower:.4f}, {self.improving.decay_ci.ci_upper:.4f}] | [{self.coasting.decay_ci.ci_lower:.4f}, {self.coasting.decay_ci.ci_upper:.4f}] | — |\n"
            f"| Cohen's d (create) | — | — | {self.cohen_d_create:.4f} |\n"
            f"| Cohen's d (decay) | — | — | {self.cohen_d_decay:.4f} |\n"
            f"| Funded channel | {self.improving.funded_channel} | {self.coasting.funded_channel} | — |\n"
            f"| Exploration ratio | {self.improving.exploration_ratio:.4f} | {self.coasting.exploration_ratio:.4f} | — |"
        )


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


def _bootstrap_rate_samples(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    rate_fn,
    n: int = BOOTSTRAP_N,
    horizon: int = DEFAULT_HORIZON,
    concentration: float = BOOTSTRAP_CONCENTRATION,
) -> np.ndarray:
    """Bootstrap per-perturbation samples of a rate function."""
    from alphacogant.generative_model import validate_belief_map

    normalized = validate_belief_map(belief, context="belief")
    samples = np.empty(n, dtype=float)
    for i in range(n):
        perturbed: dict[str, np.ndarray] = {}
        for channel, vector in normalized.items():
            alpha = np.clip(vector, 1e-9, None) * concentration + 1.0
            perturbed[channel] = rng.dirichlet(alpha)
        samples[i] = rate_fn(model, perturbed, horizon)
    return samples


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

    rng = np.random.default_rng(seed)
    boot = bootstrap_t_rsi(model, belief, rng, n=n, horizon=horizon, concentration=concentration)

    # Separate CIs from fresh bootstrap
    rng2 = np.random.default_rng(seed)
    create_samples = _bootstrap_rate_samples(model, belief, rng2, create_rate, n, horizon, concentration)
    rng3 = np.random.default_rng(seed)
    decay_samples = _bootstrap_rate_samples(model, belief, rng3, decay_rate, n, horizon, concentration)

    create_ci = bootstrap_ci(create_samples, confidence)
    decay_ci = bootstrap_ci(decay_samples, confidence)

    # EFE per action
    returns = marginal_return_vector(model, belief)
    funded_action = max(returns, key=lambda a: returns[a])
    funded_result = expected_free_energy(model, belief, funded_action)

    from alphacogant.channels import ACTIONS

    efe_prag: dict[str, float] = {}
    efe_epis: dict[str, float] = {}
    for action_idx in range(len(ACTIONS)):
        efe = expected_free_energy(model, belief, action_idx)
        efe_prag[ACTIONS[action_idx]] = efe.pragmatic
        efe_epis[ACTIONS[action_idx]] = efe.epistemic

    # Exploration ratio: fraction of actions that are epistemic
    epistemic_actions = {"fund_S", "fund_Z"}
    epistemic_count = sum(1 for a in ACTIONS if a in epistemic_actions)
    exploration_ratio = epistemic_count / len(ACTIONS)

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
        model, IMPROVING, "Improving",
        seed=seed, n=n, horizon=horizon, concentration=concentration,
    )
    coasting = compute_regime_statistics(
        model, COASTING, "Coasting",
        seed=seed, n=n, horizon=horizon, concentration=concentration,
    )
    return RegimeComparison(improving=improving, coasting=coasting)


__all__ = [
    "BootstrapCI",
    "RegimeComparison",
    "RegimeStatistics",
    "bootstrap_ci",
    "compare_regimes",
    "compute_regime_statistics",
]
