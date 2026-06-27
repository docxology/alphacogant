"""Expected Free Energy computations for AlphaCOGANT.

Active Inference minimizes Expected Free Energy ``G``. This module reports
``pragmatic`` and ``epistemic`` as positive values, and ``EFEResult.total`` is
the minimized quantity itself: ``G = -(pragmatic + epistemic)``. The negative-EFE
value used for action ranking is therefore ``-G = pragmatic + epistemic``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from alphacogant.model.channels import ACTIONS, CHANNELS
from alphacogant.model.generative_model import EconomicWorldModel, validate_belief_map


@dataclass(frozen=True)
class EFEResult:
    """Expected Free Energy decomposition for one action."""

    pragmatic: float
    epistemic: float
    total: float


def _validate_action(action: int) -> None:
    if action < 0 or action >= len(ACTIONS):
        raise ValueError(f"action must be in [0, {len(ACTIONS) - 1}], got {action}.")


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


def _kl_divergence(posterior: np.ndarray, prior: np.ndarray) -> float:
    mask = posterior > 0.0
    return float(np.sum(posterior[mask] * np.log(posterior[mask] / prior[mask])))


def _epistemic_value(
    theta_prior: np.ndarray,
    reward_given_theta: np.ndarray,
    loss_given_theta: np.ndarray,
) -> float:
    """Expected Bayesian surprise over Theta across all reward/loss observations.

    Vectorized equivalent of the per-observation loop: for every (reward, loss)
    outcome pair it forms the unnormalized Theta posterior, its observation
    probability, and the KL(posterior || prior), then takes the observation-weighted
    sum. Observations with zero probability contribute nothing (their weight is 0).
    """
    # joint[r, l, t] = theta_prior[t] * P(reward=r | theta=t) * P(loss=l | theta=t)
    joint = (
        theta_prior[None, None, :] * reward_given_theta[:, None, :] * loss_given_theta[None, :, :]
    )
    observation_prob = joint.sum(axis=2)  # (R, L)
    valid = observation_prob > 0.0
    safe_denominator = np.where(valid[..., None], joint.sum(axis=2, keepdims=True), 1.0)
    posterior = joint / safe_denominator  # (R, L, T); rows for invalid obs are meaningless
    positive = posterior > 0.0
    ratio = np.where(positive, posterior / theta_prior[None, None, :], 1.0)
    kl = np.sum(np.where(positive, posterior * np.log(ratio), 0.0), axis=2)  # (R, L)
    if np.any(kl < -1e-12):
        raise ValueError(
            "Epistemic value must be non-negative because KL divergence is non-negative."
        )
    contributions = np.where(valid, observation_prob * np.clip(kl, 0.0, None), 0.0)
    return float(np.clip(contributions.sum(), 0.0, None))


def _efe_from_validated(
    model: EconomicWorldModel,
    validated_belief: Mapping[str, np.ndarray],
    action: int,
) -> EFEResult:
    """EFE for an action given an already-validated/normalized belief map.

    Hot-path entry that skips re-validation of the input belief; callers that
    evaluate many actions against the same belief (e.g. ``marginal_return_vector``)
    validate once and reuse, which removes redundant validation from the bootstrap.
    """
    _validate_action(action)
    predicted = {
        channel: model.B[channel][:, :, action] @ validated_belief[channel] for channel in CHANNELS
    }
    reward_distribution = _reward_distribution(model, predicted)
    loss_distribution = _loss_distribution(model, predicted)
    pragmatic = float(reward_distribution @ model.C_R + loss_distribution @ model.C_L)

    theta_prior = predicted["Theta"]
    reward_given_theta, loss_given_theta = _theta_prior_likelihood_terms(model, predicted)
    epistemic = _epistemic_value(theta_prior, reward_given_theta, loss_given_theta)
    return EFEResult(pragmatic=pragmatic, epistemic=epistemic, total=-(pragmatic + epistemic))


def expected_free_energy(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    action: int,
) -> EFEResult:
    """Compute the pragmatic value, epistemic value, and total EFE for one action."""
    return _efe_from_validated(model, validate_belief_map(belief, context="belief"), action)


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
    validated = validate_belief_map(belief, context="belief")
    values: dict[int, float] = {}
    for action in range(len(ACTIONS)):
        efe = _efe_from_validated(model, validated, action)
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
