"""Synthetic create/decay diagnostics, paired t-RSI and a scalar predicate.

These rates describe the supplied finite model, not observed corporate books.
The predicate does not enforce allocation admission or coefficient learning.
Both rates average the H post-action beliefs b1…bH; b0 is not an action outcome.

- ``create_rate``: horizon-mean pragmatic value under greedy EFE allocation
  minus that under passive holding.
- ``decay_rate``: residual Theta-staleness preference gap along the same greedy
  predicted trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import fsum, sqrt

import numpy as np

from alphacogant.efe.free_energy import (
    _efe_vector_from_normalized,
    _static_pragmatic_from_normalized,
    predicted_belief,
)
from alphacogant.model.channels import CHANNELS, action_index
from alphacogant.model.generative_model import EconomicWorldModel, validate_belief_map
from alphacogant.model.operating_points import BOOTSTRAP_CONCENTRATION
from alphacogant.validation import positive_integer

DEFAULT_HORIZON = 12


def _as_sample_array(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional numeric sequence.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array


def _scaled_sample_dispersion(samples: np.ndarray) -> tuple[float, float, float]:
    """Return normalized sample SD, dispersion and outer scale.

    Inputs are admitted nonempty samples. Center before scaling to preserve
    close differences; scale first only when subtraction would overflow.
    """
    origin = float(samples[0])
    outer_scale = 1.0
    opposite = np.signbit(samples) != np.signbit(origin)
    if np.any(opposite & (np.abs(samples) > np.finfo(float).max - abs(origin))):
        outer_scale = float(np.max(np.abs(samples)))
        shifted = samples / outer_scale - origin / outer_scale
    else:
        shifted = samples - origin
    dispersion_scale = float(np.max(np.abs(shifted)))
    if dispersion_scale == 0.0:
        return 0.0, 0.0, outer_scale
    scaled = shifted / dispersion_scale
    std = float(np.std(scaled, ddof=1))
    return std, dispersion_scale, outer_scale


def _exact_mean(samples: Sequence[float] | np.ndarray) -> Fraction:
    """Exact mean of admitted float samples, used when float sums cannot suffice."""
    return sum((Fraction.from_float(float(value)) for value in samples), Fraction()) / len(samples)


def _stable_mean(samples: Sequence[float] | np.ndarray) -> float:
    try:
        return fsum(samples) / len(samples)
    except OverflowError:
        # A finite mean can exist even when an intermediate or total sum overflows.
        return float(_exact_mean(samples))


def _sample_statistics(samples: np.ndarray) -> tuple[float, float, float]:
    """Mean, sample SD and SE; rescale SD and SE independently."""
    std, scale, outer_scale = _scaled_sample_dispersion(samples)
    mean = _stable_mean(samples)
    se = (std / sqrt(samples.size)) * scale * outer_scale if samples.size > 1 else 0.0
    return mean, std * scale * outer_scale, se


def _normalize_internal(belief: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Renormalize admitted internal beliefs in the same order as public validation."""
    return {channel: belief[channel] / float(belief[channel].sum()) for channel in CHANNELS}


def _greedy_trajectory(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    horizon: int,
) -> list[dict[str, np.ndarray]]:
    current = {channel: belief[channel].copy() for channel in CHANNELS}
    trajectory: list[dict[str, np.ndarray]] = []
    for _ in range(horizon):
        normalized = _normalize_internal(current)
        scores = _efe_vector_from_normalized(model, normalized)
        action = min(range(len(scores)), key=lambda a: scores[a].total)
        current = predicted_belief(model, current, action)
        trajectory.append({channel: current[channel].copy() for channel in CHANNELS})
    return trajectory


def _hold_mean_pragmatic(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    horizon: int,
) -> float:
    hold = action_index("hold")
    current = {channel: belief[channel].copy() for channel in CHANNELS}
    values: list[float] = []
    for _ in range(horizon):
        current = predicted_belief(model, current, hold)
        values.append(_static_pragmatic_from_normalized(model, _normalize_internal(current)))
    return _stable_mean(values)


def _create_from_trajectory(
    model: EconomicWorldModel,
    normalized: Mapping[str, np.ndarray],
    trajectory: list[dict[str, np.ndarray]],
    horizon: int,
) -> float:
    greedy_mean = _stable_mean(
        [_static_pragmatic_from_normalized(model, _normalize_internal(b)) for b in trajectory]
    )
    return greedy_mean - _hold_mean_pragmatic(model, normalized, horizon)


def _decay_from_trajectory(
    model: EconomicWorldModel,
    trajectory: list[dict[str, np.ndarray]],
) -> float:
    refresh = action_index("fund_Theta")
    erosions: list[float] = []
    for b in trajectory:
        refreshed = {**b, "Theta": model.B["Theta"][:, :, refresh] @ b["Theta"]}
        erosions.append(
            max(
                0.0,
                _static_pragmatic_from_normalized(model, _normalize_internal(refreshed))
                - _static_pragmatic_from_normalized(model, _normalize_internal(b)),
            )
        )
    return _stable_mean(erosions)


def create_rate(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    horizon: int = DEFAULT_HORIZON,
) -> float:
    """Pragmatic value the greedy policy creates over passive holding (signed)."""
    positive_integer(horizon, "horizon")
    normalized = validate_belief_map(belief, context="belief")
    trajectory = _greedy_trajectory(model, normalized, horizon)
    return _create_from_trajectory(model, normalized, trajectory, horizon)


def decay_rate(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    horizon: int = DEFAULT_HORIZON,
) -> float:
    """Residual Theta-staleness erosion remaining along the greedy trajectory (>=0)."""
    positive_integer(horizon, "horizon")
    normalized = validate_belief_map(belief, context="belief")
    trajectory = _greedy_trajectory(model, normalized, horizon)
    return _decay_from_trajectory(model, trajectory)


def t_rsi(
    create_samples: Sequence[float] | np.ndarray,
    decay_samples: Sequence[float] | np.ndarray,
) -> float:
    """Return the standardized create-minus-decay separation.

    The two sample arrays are **paired**: entry ``i`` of each holds the create-
    and decay-rate read off the *same* perturbed trajectory, as
    ``paired_bootstrap_samples`` produces. The denominator is therefore the
    standard error of the elementwise differences ``create - decay`` — the
    pairing-aware SE. Synthetic belief perturbations do not supply empirical
    calibration; the statistic's magnitude depends on the number of draws.
    Common scaling preserves a dimensionless statistic without underflowing
    squared deviations or overflowing a finite sample's subtraction.
    """
    create = _as_sample_array("create_samples", create_samples)
    decay = _as_sample_array("decay_samples", decay_samples)
    if create.size != decay.size:
        raise ValueError(
            "create_samples and decay_samples must be paired (same length); "
            f"got {create.size} and {decay.size}."
        )

    if create.size < 2:
        return 0.0

    limit = np.finfo(float).max
    opposite_sign = np.signbit(create) != np.signbit(decay)
    if np.any(opposite_sign & (np.abs(create) > limit - np.abs(decay))):
        # Only scale before subtraction when the true margin exceeds float64.
        # Otherwise subtract first to retain tiny differences in paired data.
        scale = max(float(np.max(np.abs(create))), float(np.max(np.abs(decay))))
        margins = create / scale - decay / scale
    else:
        margins = create - decay
    scaled_std, scale, outer_scale = _scaled_sample_dispersion(margins)
    if scaled_std == 0.0:
        return 0.0 if margins[0] == 0.0 else float(np.sign(margins[0]) * np.inf)
    # Sum original values in dispersion units, not rounded centered offsets:
    # the latter can erase a small nonzero mean through cancellation.
    scaled_mean = fsum(float(value) / outer_scale / scale for value in margins) / margins.size
    return scaled_mean / (scaled_std / sqrt(margins.size))


def paired_bootstrap_samples(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    horizon: int = DEFAULT_HORIZON,
    concentration: float = BOOTSTRAP_CONCENTRATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-perturbation ``(create, decay)`` rate samples sharing one greedy trajectory.

    Both rates are read off the *same* greedy trajectory per perturbation, so a
    caller that needs both (the t-RSI statistic and its create/decay CIs) draws one
    deterministic sample set instead of re-bootstrapping the identical draws. The
    draw order (Dirichlet per channel, in ``CHANNELS`` order) is preserved so the
    samples are byte-identical to the previous create-then-decay implementation.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator instance.")
    positive_integer(n, "n")
    if not np.isfinite(concentration) or concentration <= 0.0:
        raise ValueError("concentration must be a positive number.")
    positive_integer(horizon, "horizon")
    normalized_belief = validate_belief_map(belief, context="belief")
    create_samples = np.empty(n, dtype=float)
    decay_samples = np.empty(n, dtype=float)
    alphas = {c: np.clip(v, 1e-9, None) * concentration + 1.0 for c, v in normalized_belief.items()}
    for index in range(n):
        perturbed: dict[str, np.ndarray] = {}
        for channel, alpha in alphas.items():
            perturbed[channel] = rng.dirichlet(alpha)
        normalized = validate_belief_map(perturbed, context="belief")
        trajectory = _greedy_trajectory(model, normalized, horizon)
        create_samples[index] = _create_from_trajectory(model, normalized, trajectory, horizon)
        decay_samples[index] = _decay_from_trajectory(model, trajectory)
    return create_samples, decay_samples


def bootstrap_t_rsi(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    n: int = 500,
    horizon: int = DEFAULT_HORIZON,
    concentration: float = BOOTSTRAP_CONCENTRATION,
) -> dict[str, float]:
    """Bootstrap t-RSI from deterministic belief perturbations drawn from ``rng``.

    ``concentration`` sets the Dirichlet tightness around the operating belief — it
    is the firm's belief precision, a modelling choice, not a tuned knob; the
    reported t-RSI is whatever this honest comparator yields at that precision.
    """
    create_samples, decay_samples = paired_bootstrap_samples(
        model, belief, rng, n, horizon, concentration
    )
    create_mean, _, create_se = _sample_statistics(create_samples)
    decay_mean, _, decay_se = _sample_statistics(decay_samples)

    return {
        "t_rsi": float(t_rsi(create_samples, decay_samples)),
        "create_mean": create_mean,
        "decay_mean": decay_mean,
        "create_se": create_se,
        "decay_se": decay_se,
    }


def certificate(t_rsi_value: float, delta: float) -> bool:
    """Evaluate a scalar threshold; no action, model update or deployment is gated."""
    if np.isnan(t_rsi_value) or np.isnan(delta):
        raise ValueError("t_rsi_value and delta must not be NaN.")
    return bool(t_rsi_value >= delta)


__all__ = [
    "DEFAULT_HORIZON",
    "bootstrap_t_rsi",
    "certificate",
    "create_rate",
    "decay_rate",
    "paired_bootstrap_samples",
    "t_rsi",
]
