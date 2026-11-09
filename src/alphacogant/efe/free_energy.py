"""Expected Free Energy computations for AlphaCOGANT.

Active Inference minimizes Expected Free Energy ``G``. This module reports
``pragmatic`` as signed expected log preference and ``epistemic`` as nonnegative
information gain. ``EFEResult.total`` is
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
    if (
        isinstance(action, (bool, np.bool_))
        or not isinstance(action, (int, np.integer))
        or action < 0
        or action >= len(ACTIONS)
    ):
        raise ValueError(f"action must be in [0, {len(ACTIONS) - 1}], got {action}.")


def predicted_belief(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    action: int,
) -> dict[str, np.ndarray]:
    """One-step predicted belief under ``action``: ``B_k[:, :, action] @ belief_k``.

    Hot-path primitive shared by inference, simulation, and the bootstrap: the
    caller is responsible for passing a validated/normalized belief (this
    function performs no validation).
    """
    _validate_action(action)
    return {channel: model.B[channel][:, :, action] @ belief[channel] for channel in CHANNELS}


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
    """Scalar masked KL divergence — the independent reference used by the
    differential epistemic-value test (not the production hot path)."""
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
    return float(_epistemic_values(theta_prior, reward_given_theta, loss_given_theta))


def _epistemic_values(
    theta_prior: np.ndarray, reward_given_theta: np.ndarray, loss_given_theta: np.ndarray
) -> np.ndarray:
    """Broadcast the same masked KL calculation over optional leading batch axes."""
    joint = (
        theta_prior[..., None, None, :]
        * reward_given_theta[..., :, None, :]
        * loss_given_theta[..., None, :, :]
    )
    observation_prob = joint.sum(axis=-1)
    valid = observation_prob > 0.0
    posterior = joint / np.where(valid[..., None], observation_prob[..., None], 1.0)
    positive = posterior > 0.0
    # log(posterior) - log(prior) remains finite when posterior/prior would
    # overflow for a valid subnormal prior. Zero posterior terms contribute 0.
    log_posterior = np.zeros_like(posterior)
    log_prior = np.zeros_like(theta_prior)
    np.log(posterior, out=log_posterior, where=positive)
    np.log(theta_prior, out=log_prior, where=theta_prior > 0.0)
    kl = np.sum(posterior * (log_posterior - log_prior[..., None, None, :]), axis=-1)
    if np.any(kl < -1e-12):
        raise ValueError(
            "Epistemic value must be non-negative because KL divergence is non-negative."
        )
    contributions = np.where(valid, observation_prob * np.clip(kl, 0.0, None), 0.0)
    return np.clip(contributions.sum(axis=(-2, -1)), 0.0, None)


def _preference_values(
    model: EconomicWorldModel, reward: np.ndarray, loss: np.ndarray
) -> np.ndarray | np.float64:
    """Evaluate scalar/batched preferences before any policy can rank them."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = reward @ model.C_R + loss @ model.C_L
    except FloatingPointError as exc:
        raise ValueError("Computed action scores must be finite.") from exc
    # BLAS reductions need not report their floating-point status to NumPy.
    if not np.all(np.isfinite(values)):
        raise ValueError("Computed action scores must be finite.")
    return values


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
    predicted = predicted_belief(model, validated_belief, action)
    reward_distribution = _reward_distribution(model, predicted)
    loss_distribution = _loss_distribution(model, predicted)
    pragmatic = float(_preference_values(model, reward_distribution, loss_distribution))

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
    return _static_pragmatic_from_normalized(model, normalized)


def _static_pragmatic_from_normalized(
    model: EconomicWorldModel, normalized: Mapping[str, np.ndarray]
) -> float:
    """Private evaluator for validated, normalized internal beliefs."""
    reward_distribution = _reward_distribution(model, normalized)
    loss_distribution = _loss_distribution(model, normalized)
    return float(_preference_values(model, reward_distribution, loss_distribution))


def efe_vector(model: EconomicWorldModel, belief: Mapping[str, np.ndarray]) -> list[EFEResult]:
    """EFE decomposition for every action, validating the belief exactly once.

    Hot-path entry for callers that score several actions against one belief
    (simulation cycles, policy scoring): one validation, one B-rollout sweep.
    """
    validated = validate_belief_map(belief, context="belief")
    return _efe_vector_from_normalized(model, validated)


def _efe_vector_from_normalized(
    model: EconomicWorldModel, belief: Mapping[str, np.ndarray]
) -> list[EFEResult]:
    """Score all six actions in one batch; callers own belief admission."""
    predicted = {c: np.einsum("ija,j->ai", model.B[c], belief[c]) for c in CHANNELS}
    reward = np.einsum(
        "riut,ai,au,at->ar", model.A_R, predicted["I"], predicted["U"], predicted["Theta"]
    )
    loss = np.einsum("lst,as,at->al", model.A_L, predicted["S"], predicted["Theta"])
    pragmatic = _preference_values(model, reward, loss)
    reward_theta = np.einsum("riut,ai,au->art", model.A_R, predicted["I"], predicted["U"])
    loss_theta = np.einsum("lst,as->alt", model.A_L, predicted["S"])
    epistemic = _epistemic_values(predicted["Theta"], reward_theta, loss_theta)
    return [
        EFEResult(float(p), float(e), float(-(p + e)))
        for p, e in zip(pragmatic, epistemic, strict=True)
    ]


def marginal_return_vector(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
) -> dict[int, float]:
    """Return the negative-EFE value for every action."""
    return {action: -result.total for action, result in enumerate(efe_vector(model, belief))}


def greedy_action(model: EconomicWorldModel, belief: Mapping[str, np.ndarray]) -> int:
    """Return the action with the highest marginal return (lowest total EFE)."""
    returns = marginal_return_vector(model, belief)
    return max(returns, key=lambda action: returns[action])


def policy_posterior(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    gamma: float = 1.0,
    *,
    values: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Return the precision-weighted softmax posterior over actions.

    ``values`` may carry precomputed marginal returns (``-EFE`` per action) so a
    caller that already scored every action against this belief does not pay a
    second EFE sweep.
    """
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite precision value.")
    if values is None:
        values = marginal_return_vector(model, belief)
    if set(values) != set(range(len(ACTIONS))):
        raise ValueError("values must contain exactly the action indices.")
    vals = np.array([values[action] for action in range(len(ACTIONS))], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("values must be finite.")
    # Center first to preserve close scores sharing a large common offset.
    with np.errstate(over="ignore", under="ignore"):
        maximum = np.max(vals)
        gaps = vals - maximum
        logits = gamma * gaps
        if gamma < 1.0:
            overflowing = np.isneginf(gaps)
            if np.any(overflowing):
                # These opposite-sign gaps may become finite after scaling.
                # Scale only them: scaling all scores first can erase close gaps.
                logits[overflowing] = gamma * vals[overflowing] - gamma * maximum
        weights = np.exp(logits)
    return weights / weights.sum()


__all__ = [
    "EFEResult",
    "efe_vector",
    "expected_free_energy",
    "greedy_action",
    "marginal_return_vector",
    "policy_posterior",
    "predicted_belief",
    "static_pragmatic_value",
]
