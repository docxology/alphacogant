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
from fractions import Fraction
from itertools import chain
from math import fsum, hypot, sqrt

import numpy as np

from alphacogant.efe.free_energy import efe_vector
from alphacogant.model.channels import ACTIONS, CHANNELS
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
from alphacogant.stats.simulation import simulate_trajectory, summarize_trajectory
from alphacogant.trsi.t_rsi import (
    DEFAULT_HORIZON,
    _exact_mean,
    _sample_statistics,
    _scaled_sample_dispersion,
    _stable_mean,
    paired_bootstrap_samples,
    t_rsi,
)


def cohen_d(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Degrees-of-freedom-pooled Cohen's d for finite 1D groups of size >=2.

    Identical constants have d=0; distinct constants have signed infinite d.
    Both groups share a scale so finite extreme inputs need not have
    representable unscaled variances (or even standard deviations).
    """
    groups = []
    for name, values in (("samples_a", samples_a), ("samples_b", samples_b)):
        group = np.asarray(values, dtype=float)
        if group.ndim != 1 or group.size < 2 or not np.all(np.isfinite(group)):
            raise ValueError(
                f"{name} must be a finite one-dimensional group of at least 2 samples."
            )
        groups.append(group)
    a, b = groups
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if scale == 0.0:
        return 0.0
    std_a, dispersion_a, outer_a = _scaled_sample_dispersion(a)
    std_b, dispersion_b, outer_b = _scaled_sample_dispersion(b)
    unit_a = (dispersion_a / scale) * outer_a
    unit_b = (dispersion_b / scale) * outer_b
    if a.size == b.size:
        try:
            difference = fsum(chain(a, (-value for value in b))) / scale / a.size
        except OverflowError:
            difference = float((_exact_mean(a) - _exact_mean(b)) / Fraction.from_float(scale))
    else:
        # Do not round two large nearby means before taking their difference,
        # or independently round unequal-count weights before cancellation.
        difference = float((_exact_mean(a) - _exact_mean(b)) / Fraction.from_float(scale))
    return _cohen_d_from_moments(
        difference, std_a * unit_a, a.size, 0.0, std_b * unit_b, b.size
    )


def _cohen_d_from_moments(
    mean_a: float, std_a: float, n_a: int, mean_b: float, std_b: float, n_b: int
) -> float:
    if n_a < 2 or n_b < 2:
        raise ValueError("Cohen's d requires at least 2 samples in each group.")
    if std_a == std_b == 0.0:
        if mean_a == mean_b:
            return 0.0
        return float("inf") if mean_a > mean_b else -float("inf")
    scale = max(std_a, std_b)
    degrees = n_a + n_b - 2
    pooled_std = hypot(
        (std_a / scale) * sqrt((n_a - 1) / degrees),
        (std_b / scale) * sqrt((n_b - 1) / degrees),
    )
    difference = float(mean_a) - float(mean_b)
    difference = (
        difference / scale
        if np.isfinite(difference)
        else mean_a / scale - mean_b / scale
    )
    if pooled_std == 0.0:
        return 0.0 if difference == 0.0 else float(np.copysign(np.inf, difference))
    return difference / pooled_std


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


def _cohen_d_from_intervals(a: BootstrapCI, b: BootstrapCI) -> float:
    if np.isfinite(a.std) and np.isfinite(b.std):
        return _cohen_d_from_moments(a.mean, a.std, a.n, b.mean, b.std, b.n)
    # Finite samples have representable SE even when their sample SD overflows.
    # Normalize before reconstructing SD, retaining the stored sample counts.
    scale = max(abs(a.mean), abs(b.mean), a.se, b.se)
    return _cohen_d_from_moments(
        a.mean / scale, (a.se / scale) * sqrt(a.n), a.n,
        b.mean / scale, (b.se / scale) * sqrt(b.n), b.n,
    )


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
            "create_ci_confidence": self.create_ci.confidence,
            "create_n": self.create_ci.n,
            "decay_mean": self.decay_ci.mean,
            "decay_ci": [self.decay_ci.ci_lower, self.decay_ci.ci_upper],
            "decay_ci_confidence": self.decay_ci.confidence,
            "decay_n": self.decay_ci.n,
            "funded_channel": self.funded_channel,
            "funded_pragmatic": self.funded_pragmatic,
            "funded_epistemic": self.funded_epistemic,
            "exploration_ratio": self.exploration_ratio,
        }


@dataclass(frozen=True)
class RegimeComparison:
    """Compare stored binary64 regime summaries, not the original sample clouds."""

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
        """Pooled d from stored create moments; rounding can erase a raw-sample effect."""
        return _cohen_d_from_intervals(self.improving.create_ci, self.coasting.create_ci)

    @property
    def cohen_d_decay(self) -> float:
        """Pooled d from stored decay moments; use raw ``cohen_d`` when samples remain."""
        return _cohen_d_from_intervals(self.improving.decay_ci, self.coasting.decay_ci)

    def as_table(self) -> str:
        """Render intervals with each cell's stored confidence, even when levels differ."""
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
                f"| Create-rate CI | {100 * self.improving.create_ci.confidence:g}% CI "
                f"[{self.improving.create_ci.ci_lower:.4f}, "
                f"{self.improving.create_ci.ci_upper:.4f}] | "
                f"{100 * self.coasting.create_ci.confidence:g}% CI "
                f"[{self.coasting.create_ci.ci_lower:.4f}, "
                f"{self.coasting.create_ci.ci_upper:.4f}] | — |"
            ),
            (
                f"| Decay-rate mean | {self.improving.decay_ci.mean:.4f} | "
                f"{self.coasting.decay_ci.mean:.4f} | {self.decay_rate_delta:.4f} |"
            ),
            (
                f"| Decay-rate CI | {100 * self.improving.decay_ci.confidence:g}% CI "
                f"[{self.improving.decay_ci.ci_lower:.4f}, "
                f"{self.improving.decay_ci.ci_upper:.4f}] | "
                f"{100 * self.coasting.decay_ci.confidence:g}% CI "
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


def _percentile_bounds(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    """Central linear quantiles with exact ranks and one final endpoint rounding."""
    confidence_ratio = Fraction.from_float(float(confidence))
    last = samples.size - 1
    lower_position = last * (1 - confidence_ratio) / 2
    positions = (lower_position, last - lower_position)
    brackets = []
    for position in positions:
        left, remainder = divmod(position.numerator, position.denominator)
        brackets.append((left, left + bool(remainder), remainder, position.denominator))
    indices = sorted({index for left, right, _, _ in brackets for index in (left, right)})
    partitioned = np.partition(samples, indices)
    bounds = []
    for left, right, remainder, denominator in brackets:
        lo, hi = float(partitioned[left]), float(partitioned[right])
        if left == right or lo == hi:
            bounds.append(lo)
            continue
        weight = Fraction(remainder, denominator)
        exact_lo, exact_hi = Fraction.from_float(lo), Fraction.from_float(hi)
        bounds.append(float(exact_lo + weight * (exact_hi - exact_lo)))
    return bounds[0], bounds[1]


def bootstrap_ci(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Compute a bootstrap confidence interval from a sample array."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("samples must be a finite one-dimensional sequence.")
    if samples.size < 2:
        raise ValueError("Need at least 2 samples for a confidence interval.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")
    mean, std, se = _sample_statistics(samples)
    ci_lower, ci_upper = _percentile_bounds(samples, confidence)
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
    margin_ci_lower, margin_ci_upper = _percentile_bounds(margins, confidence)
    margin_mean, margin_std, _ = _sample_statistics(margins)
    return BreakEvenProfile(
        probability=float(np.mean(margins > 0.0)),
        margin_mean=margin_mean,
        margin_std=margin_std,
        margin_ci_lower=margin_ci_lower,
        margin_ci_upper=margin_ci_upper,
        create_mean=_stable_mean(create_samples),
        decay_mean=_stable_mean(decay_samples),
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

    # One validated EFE sweep covers the funded-channel tokens and the per-action
    # decomposition tables.
    efe_results = efe_vector(model, belief)
    funded_action = max(range(len(ACTIONS)), key=lambda a: -efe_results[a].total)
    funded_result = efe_results[funded_action]

    efe_prag: dict[str, float] = {}
    efe_epis: dict[str, float] = {}
    for action_idx, efe in enumerate(efe_results):
        efe_prag[ACTIONS[action_idx]] = efe.pragmatic
        efe_epis[ACTIONS[action_idx]] = efe.epistemic

    # Exploration ratio: the fraction of greedy-trajectory CYCLES this regime spends
    # funding an epistemic channel (S or Z) — a behavioral, regime-discriminating
    # quantity, not the structural constant (epistemic actions / total actions).
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
    "BootstrapCI",
    "BreakEvenProfile",
    "RegimeComparison",
    "RegimeStatistics",
    "break_even_profile",
    "bootstrap_ci",
    "cohen_d",
    "compare_regimes",
    "compute_regime_statistics",
]
