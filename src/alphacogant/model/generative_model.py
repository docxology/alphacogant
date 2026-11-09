"""Generative-model primitives for the AlphaFund Economic World Model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from alphacogant.model.channels import ACTIONS, CHANNELS

ArrayMap = Mapping[str, np.ndarray]


@dataclass
class EconomicWorldModel:
    """Validated float coefficients, with observation/next-state axes first.

    Construction converts lists and integer arrays without normalizing values.
    B and D get new mappings; compatible float arrays (including views) remain
    shared with their original writeability. Callers must preserve invariants
    whenever those arrays are mutable.
    """

    A_R: np.ndarray
    A_L: np.ndarray
    B: dict[str, np.ndarray]
    C_R: np.ndarray
    C_L: np.ndarray
    D: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        self.A_R = np.asarray(self.A_R, dtype=float)
        self.A_L = np.asarray(self.A_L, dtype=float)
        self.B = {channel: np.asarray(values, dtype=float) for channel, values in self.B.items()}
        self.C_R = np.asarray(self.C_R, dtype=float)
        self.C_L = np.asarray(self.C_L, dtype=float)
        self.D = {channel: np.asarray(values, dtype=float) for channel, values in self.D.items()}
        _validate_model(self)


def _as_array(name: str, values: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array


def _normalize_probability_columns(name: str, values: object, shape: tuple[int, ...]) -> np.ndarray:
    array = _as_array(name, values, shape)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must not contain negative probabilities.")
    column_sums = array.sum(axis=0, keepdims=True)
    if not np.all(np.isfinite(column_sums)) or np.any(column_sums <= 0.0):
        raise ValueError(f"{name} has a zero-sum probability column.")
    return array / column_sums


def _normalize_probability_vector(name: str, values: object) -> np.ndarray:
    vector = _as_array(name, values, (2,))
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must not contain negative probabilities.")
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive total probability mass.")
    return vector / total


def _validate_probability_columns(name: str, array: np.ndarray) -> None:
    if np.any(array < 0.0):
        raise ValueError(f"{name} must not contain negative probabilities.")
    sums = array.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} columns must sum to 1.")


def _validate_probability_vector(name: str, vector: np.ndarray) -> None:
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must not contain negative probabilities.")
    if not np.allclose(vector.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} must sum to 1.")


def _validate_belief_map(
    prior: Mapping[str, np.ndarray],
    *,
    context: str,
) -> dict[str, np.ndarray]:
    if not isinstance(prior, Mapping):
        raise ValueError(f"{context} must be a channel mapping.")
    if set(prior) - set(CHANNELS):
        raise ValueError(f"{context} contains unknown channel keys.")
    missing = [channel for channel in CHANNELS if channel not in prior]
    if missing:
        raise ValueError(f"{context} is missing channel priors for {missing}.")
    validated: dict[str, np.ndarray] = {}
    for channel in CHANNELS:
        vector = _as_array(f"{context}[{channel}]", prior[channel], (2,))
        if np.any(vector < 0.0):
            raise ValueError(f"{context}[{channel}] must not contain negative probabilities.")
        total = float(vector.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"{context}[{channel}] must have positive total probability mass.")
        validated[channel] = vector / total
    return validated


def _build_default_B() -> dict[str, np.ndarray]:
    raw_B = {
        "I": np.array(
            [
                [
                    (0.60, 0.05),
                    (0.95, 0.95),
                    (0.95, 0.95),
                    (0.95, 0.95),
                    (0.95, 0.95),
                    (0.90, 0.10),
                ],
                [
                    (0.40, 0.95),
                    (0.05, 0.05),
                    (0.05, 0.05),
                    (0.05, 0.05),
                    (0.05, 0.05),
                    (0.10, 0.90),
                ],
            ],
            dtype=float,
        ),
        "S": np.array(
            [
                [
                    (0.90, 0.90),
                    (0.55, 0.05),
                    (0.90, 0.90),
                    (0.90, 0.90),
                    (0.90, 0.90),
                    (0.90, 0.10),
                ],
                [
                    (0.10, 0.10),
                    (0.45, 0.95),
                    (0.10, 0.10),
                    (0.10, 0.10),
                    (0.10, 0.10),
                    (0.10, 0.90),
                ],
            ],
            dtype=float,
        ),
        "U": np.array(
            [
                [
                    (0.90, 0.90),
                    (0.90, 0.90),
                    (0.55, 0.05),
                    (0.90, 0.90),
                    (0.90, 0.90),
                    (0.90, 0.10),
                ],
                [
                    (0.10, 0.10),
                    (0.10, 0.10),
                    (0.45, 0.95),
                    (0.10, 0.10),
                    (0.10, 0.10),
                    (0.10, 0.90),
                ],
            ],
            dtype=float,
        ),
        "Theta": np.array(
            [
                [
                    (0.85, 0.35),
                    (0.85, 0.35),
                    (0.85, 0.35),
                    (0.50, 0.02),
                    (0.85, 0.35),
                    (0.85, 0.40),
                ],
                [
                    (0.15, 0.65),
                    (0.15, 0.65),
                    (0.15, 0.65),
                    (0.50, 0.98),
                    (0.15, 0.65),
                    (0.15, 0.60),
                ],
            ],
            dtype=float,
        ),
        "Z": np.array(
            [
                [
                    (0.90, 0.90),
                    (0.90, 0.90),
                    (0.90, 0.90),
                    (0.90, 0.90),
                    (0.55, 0.05),
                    (0.90, 0.10),
                ],
                [
                    (0.10, 0.10),
                    (0.10, 0.10),
                    (0.10, 0.10),
                    (0.10, 0.10),
                    (0.45, 0.95),
                    (0.10, 0.90),
                ],
            ],
            dtype=float,
        ),
    }
    normalized: dict[str, np.ndarray] = {}
    for channel, values in raw_B.items():
        normalized[channel] = _normalize_probability_columns(
            f"B[{channel}]",
            values.transpose(0, 2, 1),
            (2, 2, len(ACTIONS)),
        )
    return normalized


def _validate_model(model: EconomicWorldModel) -> None:
    if model.B.keys() != set(CHANNELS):
        raise ValueError(f"B must contain exactly the channel keys {CHANNELS}.")
    if model.D.keys() != set(CHANNELS):
        raise ValueError(f"D must contain exactly the channel keys {CHANNELS}.")
    _validate_probability_columns("A_R", _as_array("A_R", model.A_R, (3, 2, 2, 2)))
    _validate_probability_columns("A_L", _as_array("A_L", model.A_L, (3, 2, 2)))
    for channel in CHANNELS:
        _validate_probability_columns(
            f"B[{channel}]",
            _as_array(f"B[{channel}]", model.B[channel], (2, 2, len(ACTIONS))),
        )
        _validate_probability_vector(
            f"D[{channel}]",
            _as_array(f"D[{channel}]", model.D[channel], (2,)),
        )
    _as_array("C_R", model.C_R, (3,))
    _as_array("C_L", model.C_L, (3,))


def default_model(
    *,
    A_R: object | None = None,
    A_L: object | None = None,
    B: Mapping[str, object] | None = None,
    C_R: object | None = None,
    C_L: object | None = None,
    D: Mapping[str, object] | None = None,
) -> EconomicWorldModel:
    """Return the normalized default AlphaFund Economic World Model.

    Optional overrides are accepted to validate alternate arrays against the same
    contract while preserving the default GNN values when no overrides are passed.
    """
    default_A_R = np.array(
        [
            [[[0.80, 0.55], [0.55, 0.30]], [[0.55, 0.30], [0.30, 0.10]]],
            [[[0.15, 0.30], [0.30, 0.40]], [[0.30, 0.40], [0.40, 0.30]]],
            [[[0.05, 0.15], [0.15, 0.30]], [[0.15, 0.30], [0.30, 0.60]]],
        ],
        dtype=float,
    )
    default_A_L = np.array(
        [
            [[0.70, 0.45], [0.45, 0.15]],
            [[0.22, 0.35], [0.35, 0.30]],
            [[0.08, 0.20], [0.20, 0.55]],
        ],
        dtype=float,
    )
    default_D = {
        "I": np.array([0.6, 0.4], dtype=float),
        "S": np.array([0.5, 0.5], dtype=float),
        "U": np.array([0.6, 0.4], dtype=float),
        "Theta": np.array([0.4, 0.6], dtype=float),
        "Z": np.array([0.5, 0.5], dtype=float),
    }
    normalized_A_R = _normalize_probability_columns(
        "A_R", A_R if A_R is not None else default_A_R, (3, 2, 2, 2)
    )
    normalized_A_L = _normalize_probability_columns(
        "A_L", A_L if A_L is not None else default_A_L, (3, 2, 2)
    )
    normalized_B = _build_default_B()
    if B is not None:
        if set(B) - set(CHANNELS):
            raise ValueError("B override contains unknown channel keys.")
        missing = [channel for channel in CHANNELS if channel not in B]
        if missing:
            raise ValueError(f"B override is missing channels {missing}.")
        normalized_B = {
            channel: _normalize_probability_columns(
                f"B[{channel}]",
                B[channel],
                (2, 2, len(ACTIONS)),
            )
            for channel in CHANNELS
        }
    if D is not None:
        if set(D) - set(CHANNELS):
            raise ValueError("D override contains unknown channel keys.")
        missing = [channel for channel in CHANNELS if channel not in D]
        if missing:
            raise ValueError(f"D override is missing channels {missing}.")
    normalized_D = {
        channel: _normalize_probability_vector(
            f"D[{channel}]",
            (D[channel] if D is not None else default_D[channel]),
        )
        for channel in CHANNELS
    }
    return EconomicWorldModel(
        A_R=normalized_A_R,
        A_L=normalized_A_L,
        B=normalized_B,
        C_R=_as_array(
            "C_R", C_R if C_R is not None else np.array([-2.0, 0.0, 3.0], dtype=float), (3,)
        ),
        C_L=_as_array(
            "C_L", C_L if C_L is not None else np.array([-2.0, 0.0, 2.0], dtype=float), (3,)
        ),
        D=normalized_D,
    )


def belief_prior(model: EconomicWorldModel | None = None) -> dict[str, np.ndarray]:
    """Return a copy of the prior belief over all channel factors."""
    active_model = default_model() if model is None else model
    return {channel: active_model.D[channel].copy() for channel in CHANNELS}


def _log_probabilities(values: np.ndarray) -> np.ndarray:
    result = np.full_like(values, -np.inf)
    np.log(values, out=result, where=values > 0.0)
    return result


def _log_marginal(terms: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Sum nonnegative factors without losing rare positive contractions."""
    maximum = np.max(terms, axis=axes, keepdims=True)
    shifted = np.full_like(terms, -np.inf)
    np.subtract(terms, maximum, out=shifted, where=np.isfinite(maximum))
    with np.errstate(under="ignore"):
        mass = np.sum(np.exp(shifted), axis=axes)
    return _log_probabilities(mass) + np.squeeze(maximum, axis=axes)


def _posterior_from_log_likelihoods(
    name: str, log_prior: np.ndarray, *log_likelihoods: np.ndarray
) -> np.ndarray:
    """Normalize rare evidence; retain exact zeros as impossible support."""
    log_weight = log_prior.copy()
    for likelihood in log_likelihoods:
        log_weight += likelihood
    maximum = float(np.max(log_weight))
    if not np.isfinite(maximum):
        raise ValueError(f"{name} must have positive total probability mass.")
    # Posterior mass below representable precision legitimately rounds to zero.
    with np.errstate(under="ignore"):
        weights = np.exp(log_weight - maximum)
    return weights / weights.sum()


def infer_states(
    model: EconomicWorldModel,
    obs_R: int,
    obs_L: int,
    prior: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Perform a single mean-field Bayesian state update from reward and loss observations."""
    if (
        isinstance(obs_R, (bool, np.bool_))
        or not isinstance(obs_R, (int, np.integer))
        or obs_R not in {0, 1, 2}
    ):
        raise ValueError("obs_R must be one of the reward buckets {0, 1, 2}.")
    if (
        isinstance(obs_L, (bool, np.bool_))
        or not isinstance(obs_L, (int, np.integer))
        or obs_L not in {0, 1, 2}
    ):
        raise ValueError("obs_L must be one of the loss buckets {0, 1, 2}.")
    beliefs = _validate_belief_map(prior, context="prior")
    posterior: dict[str, np.ndarray] = {}

    reward = _log_probabilities(model.A_R[obs_R])
    loss = _log_probabilities(model.A_L[obs_L])
    log_beliefs = {
        channel: _log_probabilities(beliefs[channel]) for channel in ("I", "S", "U", "Theta")
    }
    reward_given_I = _log_marginal(
        reward + log_beliefs["U"][None, :, None] + log_beliefs["Theta"][None, None, :],
        (1, 2),
    )
    reward_given_U = _log_marginal(
        reward + log_beliefs["I"][:, None, None] + log_beliefs["Theta"][None, None, :],
        (0, 2),
    )
    reward_given_Theta = _log_marginal(
        reward + log_beliefs["I"][:, None, None] + log_beliefs["U"][None, :, None],
        (0, 1),
    )
    loss_given_S = _log_marginal(loss + log_beliefs["Theta"][None, :], (1,))
    loss_given_Theta = _log_marginal(loss + log_beliefs["S"][:, None], (0,))

    posterior["I"] = _posterior_from_log_likelihoods(
        "posterior[I]", log_beliefs["I"], reward_given_I
    )
    posterior["S"] = _posterior_from_log_likelihoods(
        "posterior[S]", log_beliefs["S"], loss_given_S
    )
    posterior["U"] = _posterior_from_log_likelihoods(
        "posterior[U]", log_beliefs["U"], reward_given_U
    )
    posterior["Theta"] = _posterior_from_log_likelihoods(
        "posterior[Theta]", log_beliefs["Theta"], reward_given_Theta, loss_given_Theta
    )
    posterior["Z"] = beliefs["Z"].copy()
    return posterior


def validate_belief_map(
    belief: Mapping[str, np.ndarray], *, context: str = "belief"
) -> dict[str, np.ndarray]:
    """Validate and normalize a belief map for downstream modules."""
    return _validate_belief_map(belief, context=context)


__all__ = [
    "EconomicWorldModel",
    "belief_prior",
    "default_model",
    "infer_states",
    "validate_belief_map",
]
