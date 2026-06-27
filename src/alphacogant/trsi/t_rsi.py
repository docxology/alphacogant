"""t-RSI utilities: create-rate, decay-rate, standardized distance, and certificate.

t-RSI is the standardized distance between two posteriors over **genuinely
different processes** (AlphaFund's own framing): the alpha *created* per cycle by
the firm's active allocation policy, and the alpha *decayed* from the deployed book
because parameters age. Crucially the two are NOT ordered by construction, so t-RSI
can be positive (the firm self-improves) or negative (it bleeds), and the
certificate can honestly fire or fail.

Both rates are path integrals over the planning horizon, matching the whitepaper's
"path integral along the planned allocation path":

- ``create_rate`` = horizon-mean pragmatic value of the greedy Expected-Free-Energy
  policy MINUS horizon-mean pragmatic value of passive holding. The value the
  firm's optimal allocation creates over standing still.
- ``decay_rate`` = horizon-mean residual parameter-staleness erosion that REMAINS
  along the greedy trajectory. A policy that keeps refreshing Theta drives this
  toward zero; a policy that neglects it pays the full Theta-freshness gap each
  cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from alphacogant.efe.free_energy import marginal_return_vector, static_pragmatic_value
from alphacogant.model.channels import ACTIONS, CHANNELS, action_index
from alphacogant.model.generative_model import EconomicWorldModel, validate_belief_map

DEFAULT_HORIZON = 12


def _as_sample_array(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional numeric sequence.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array


def _standard_error(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    return float(np.std(samples, ddof=1) / np.sqrt(samples.size))


def _advance(
    model: EconomicWorldModel, belief: Mapping[str, np.ndarray], action: int
) -> dict[str, np.ndarray]:
    return {channel: model.B[channel][:, :, action] @ belief[channel] for channel in CHANNELS}


def _greedy_action(model: EconomicWorldModel, belief: Mapping[str, np.ndarray]) -> int:
    values = marginal_return_vector(model, belief)
    return max(range(len(ACTIONS)), key=lambda action: values[action])


def _greedy_trajectory(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    horizon: int,
) -> list[dict[str, np.ndarray]]:
    current = {channel: belief[channel].copy() for channel in CHANNELS}
    trajectory: list[dict[str, np.ndarray]] = []
    for _ in range(horizon):
        trajectory.append({channel: current[channel].copy() for channel in CHANNELS})
        current = _advance(model, current, _greedy_action(model, current))
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
        values.append(static_pragmatic_value(model, current))
        current = _advance(model, current, hold)
    return float(np.mean(values))


def _create_from_trajectory(
    model: EconomicWorldModel,
    normalized: Mapping[str, np.ndarray],
    trajectory: list[dict[str, np.ndarray]],
    horizon: int,
) -> float:
    greedy_mean = float(np.mean([static_pragmatic_value(model, b) for b in trajectory]))
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
            max(0.0, static_pragmatic_value(model, refreshed) - static_pragmatic_value(model, b))
        )
    return float(np.mean(erosions))


def create_rate(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    horizon: int = DEFAULT_HORIZON,
) -> float:
    """Pragmatic value the greedy policy creates over passive holding (signed)."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    normalized = validate_belief_map(belief, context="belief")
    trajectory = _greedy_trajectory(model, normalized, horizon)
    return _create_from_trajectory(model, normalized, trajectory, horizon)


def decay_rate(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    horizon: int = DEFAULT_HORIZON,
) -> float:
    """Residual Theta-staleness erosion remaining along the greedy trajectory (>=0)."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    normalized = validate_belief_map(belief, context="belief")
    trajectory = _greedy_trajectory(model, normalized, horizon)
    return _decay_from_trajectory(model, trajectory)


def t_rsi(
    create_samples: Sequence[float] | np.ndarray,
    decay_samples: Sequence[float] | np.ndarray,
) -> float:
    """Return the standardized create-minus-decay separation."""
    create = _as_sample_array("create_samples", create_samples)
    decay = _as_sample_array("decay_samples", decay_samples)
    if create.size < 2 or decay.size < 2:
        return 0.0

    create_mean = float(np.mean(create))
    decay_mean = float(np.mean(decay))
    create_se = _standard_error(create)
    decay_se = _standard_error(decay)
    pooled_se = float(np.sqrt(create_se**2 + decay_se**2))
    if pooled_se == 0.0:
        if create_mean == decay_mean:
            return 0.0
        return float(np.inf)
    return float((create_mean - decay_mean) / pooled_se)


def paired_bootstrap_samples(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    horizon: int = DEFAULT_HORIZON,
    concentration: float = 12.0,
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
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if concentration <= 0.0:
        raise ValueError("concentration must be a positive number.")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    normalized_belief = validate_belief_map(belief, context="belief")
    create_samples = np.empty(n, dtype=float)
    decay_samples = np.empty(n, dtype=float)
    for index in range(n):
        perturbed: dict[str, np.ndarray] = {}
        for channel, vector in normalized_belief.items():
            alpha = np.clip(vector, 1e-9, None) * concentration + 1.0
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
    concentration: float = 12.0,
) -> dict[str, float]:
    """Bootstrap t-RSI from deterministic belief perturbations drawn from ``rng``.

    ``concentration`` sets the Dirichlet tightness around the operating belief — it
    is the firm's belief precision, a modelling choice, not a tuned knob; the
    reported t-RSI is whatever this honest comparator yields at that precision.
    """
    create_samples, decay_samples = paired_bootstrap_samples(
        model, belief, rng, n, horizon, concentration
    )

    return {
        "t_rsi": float(t_rsi(create_samples, decay_samples)),
        "create_mean": float(np.mean(create_samples)),
        "decay_mean": float(np.mean(decay_samples)),
        "create_se": _standard_error(create_samples),
        "decay_se": _standard_error(decay_samples),
    }


def certificate(t_rsi_value: float, delta: float) -> bool:
    """Return whether a t-RSI value clears the required margin (the commit gate)."""
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
