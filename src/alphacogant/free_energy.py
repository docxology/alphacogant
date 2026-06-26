"""Expected Free Energy computations for AlphaCOGANT.

Active Inference minimizes Expected Free Energy ``G``. This module reports
``pragmatic`` and ``epistemic`` as positive values, and ``EFEResult.total`` is
the minimized quantity itself: ``G = -(pragmatic + epistemic)``. The negative-EFE
value used for action ranking is therefore ``-G = pragmatic + epistemic``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from alphacogant.channels import ACTIONS, CHANNELS
from alphacogant.generative_model import EconomicWorldModel, validate_belief_map


@dataclass(frozen=True)
class EFEResult:
    """Expected Free Energy decomposition for one action."""

    pragmatic: float
    epistemic: float
    total: float


def _validate_action(action: int) -> None:
    if action < 0 or action >= len(ACTIONS):
        raise ValueError(f"action must be in [0, {len(ACTIONS) - 1}], got {action}.")


def _predict_belief(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    action: int,
) -> dict[str, np.ndarray]:
    _validate_action(action)
    normalized = validate_belief_map(belief, context="belief")
    predicted: dict[str, np.ndarray] = {}
    for channel in CHANNELS:
        predicted[channel] = model.B[channel][:, :, action] @ normalized[channel]
    return predicted


def _reward_distribution(model: EconomicWorldModel, belief: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.einsum("riut,i,u,t->r", model.A_R, belief["I"], belief["U"], belief["Theta"])


def _loss_distribution(model: EconomicWorldModel, belief: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.einsum("lst,s,t->l", model.A_L, belief["S"], belief["Theta"])


def _theta_prior_likelihood_terms(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    reward_given_theta = np.einsum("riut,i,u->rt", model.A_R, belief["I"], belief["U"])
    loss_given_theta = np.einsum("lst,s->lt", model.A_L, belief["S"])
    return reward_given_theta, loss_given_theta


def _theta_posterior_for_observation(
    theta_prior: np.ndarray,
    reward_given_theta: np.ndarray,
    loss_given_theta: np.ndarray,
    reward_index: int,
    loss_index: int,
) -> tuple[np.ndarray, float]:
    unnormalized = theta_prior * reward_given_theta[reward_index] * loss_given_theta[loss_index]
    observation_prob = float(unnormalized.sum())
    if observation_prob <= 0.0:
        return theta_prior.copy(), 0.0
    posterior = unnormalized / observation_prob
    return posterior, observation_prob


def _kl_divergence(posterior: np.ndarray, prior: np.ndarray) -> float:
    mask = posterior > 0.0
    return float(np.sum(posterior[mask] * np.log(posterior[mask] / prior[mask])))


def expected_free_energy(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    action: int,
) -> EFEResult:
    """Compute the pragmatic value, epistemic value, and total EFE for one action."""
    predicted = _predict_belief(model, belief, action)
    reward_distribution = _reward_distribution(model, predicted)
    loss_distribution = _loss_distribution(model, predicted)
    pragmatic = float(reward_distribution @ model.C_R + loss_distribution @ model.C_L)

    theta_prior = predicted["Theta"]
    reward_given_theta, loss_given_theta = _theta_prior_likelihood_terms(model, predicted)
    epistemic = 0.0
    for reward_index in range(reward_distribution.shape[0]):
        for loss_index in range(loss_distribution.shape[0]):
            posterior, observation_prob = _theta_posterior_for_observation(
                theta_prior,
                reward_given_theta,
                loss_given_theta,
                reward_index,
                loss_index,
            )
            if observation_prob == 0.0:
                continue
            kl_value = _kl_divergence(posterior, theta_prior)
            if kl_value < -1e-12:
                raise ValueError("Epistemic value must be non-negative because KL divergence is non-negative.")
            epistemic += observation_prob * max(0.0, kl_value)

    epistemic = float(np.clip(epistemic, 0.0, None))
    total = -(pragmatic + epistemic)
    return EFEResult(pragmatic=pragmatic, epistemic=epistemic, total=total)


def static_pragmatic_value(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
) -> float:
    """Pragmatic value (expected log-preference) at the *current* belief.

    No action is applied and no transition is rolled forward — this is the
    standing-still baseline against which active funding gains (create-rate) and
    passive erosion (decay-rate) are both measured, so the two are commensurable
    pragmatic deltas of genuinely different processes rather than a level versus a
    delta.
    """
    normalized = validate_belief_map(belief, context="belief")
    reward_distribution = _reward_distribution(model, normalized)
    loss_distribution = _loss_distribution(model, normalized)
    return float(reward_distribution @ model.C_R + loss_distribution @ model.C_L)


def marginal_return_vector(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
) -> dict[int, float]:
    """Return the negative-EFE value for every action."""
    values: dict[int, float] = {}
    for action in range(len(ACTIONS)):
        efe = expected_free_energy(model, belief, action)
        values[action] = -efe.total
    return values


def policy_posterior(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    gamma: float = 1.0,
) -> np.ndarray:
    """Return the precision-weighted softmax posterior over actions."""
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite precision value.")
    values = np.array(list(marginal_return_vector(model, belief).values()), dtype=float)
    logits = gamma * values
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()


__all__ = [
    "EFEResult",
    "expected_free_energy",
    "marginal_return_vector",
    "policy_posterior",
    "static_pragmatic_value",
]
