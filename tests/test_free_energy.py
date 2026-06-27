"""Tests for Expected Free Energy and policy evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.efe.free_energy import (
    _epistemic_value,
    _kl_divergence,
    expected_free_energy,
    marginal_return_vector,
    policy_posterior,
)
from alphacogant.model.channels import action_index
from alphacogant.model.generative_model import default_model


def _reference_epistemic(
    theta_prior: np.ndarray,
    reward_given_theta: np.ndarray,
    loss_given_theta: np.ndarray,
) -> float:
    """Naive per-observation epistemic value — the pre-vectorization reference.

    Independent implementation (explicit double loop over reward/loss outcomes,
    using the scalar ``_kl_divergence``) that pins the vectorized ``_epistemic_value``
    so a value-altering vectorization bug cannot pass silently.
    """
    total = 0.0
    for r in range(reward_given_theta.shape[0]):
        for loss in range(loss_given_theta.shape[0]):
            unnormalized = theta_prior * reward_given_theta[r] * loss_given_theta[loss]
            observation_prob = float(unnormalized.sum())
            if observation_prob <= 0.0:
                continue
            posterior = unnormalized / observation_prob
            total += observation_prob * max(0.0, _kl_divergence(posterior, theta_prior))
    return total


def test_epistemic_value_matches_naive_per_observation_loop() -> None:
    """Vectorized _epistemic_value equals the explicit per-observation KL loop."""
    rng = np.random.default_rng(20240623)
    max_diff = 0.0
    for _ in range(200):
        theta_states = int(rng.integers(2, 5))
        reward_outcomes = int(rng.integers(2, 5))
        loss_outcomes = int(rng.integers(2, 5))
        theta_prior = rng.dirichlet(np.ones(theta_states))
        # Columns over outcomes sum to 1 per theta-state, matching A column-stochasticity.
        reward_given_theta = rng.dirichlet(np.ones(reward_outcomes), size=theta_states).T
        loss_given_theta = rng.dirichlet(np.ones(loss_outcomes), size=theta_states).T
        vectorized = _epistemic_value(theta_prior, reward_given_theta, loss_given_theta)
        reference = _reference_epistemic(theta_prior, reward_given_theta, loss_given_theta)
        assert vectorized >= 0.0
        max_diff = max(max_diff, abs(vectorized - reference))
    assert max_diff < 1e-12, f"vectorized epistemic drifted from reference by {max_diff}"


def test_expected_free_energy_sign_and_value_consistency(model, prior) -> None:
    result = expected_free_energy(model, prior, action_index("fund_I"))
    assert result.total == pytest.approx(-(result.pragmatic + result.epistemic))

    returns = marginal_return_vector(model, prior)
    assert returns[action_index("fund_I")] == pytest.approx(-result.total)

    posterior = policy_posterior(model, prior, gamma=1.0)
    assert posterior.shape == (6,)
    assert np.all(posterior > 0.0)
    assert posterior.sum() == pytest.approx(1.0)


def test_epistemic_value_is_non_negative_across_actions_and_beliefs(model, prior) -> None:
    beliefs = []
    for theta_strong in np.linspace(0.05, 0.95, 5):
        beliefs.append(
            {
                "I": np.array([0.25, 0.75], dtype=float),
                "S": np.array([0.6, 0.4], dtype=float),
                "U": np.array([0.55, 0.45], dtype=float),
                "Theta": np.array([1.0 - theta_strong, theta_strong], dtype=float),
                "Z": prior["Z"].copy(),
            }
        )
    for belief in beliefs:
        for action in range(6):
            result = expected_free_energy(model, belief, action)
            assert result.epistemic >= 0.0


def test_funding_theta_beats_hold_on_epistemic_value_when_theta_is_stale(model, prior) -> None:
    stale_belief = {channel: vector.copy() for channel, vector in prior.items()}
    stale_belief["Theta"] = np.array([0.95, 0.05], dtype=float)
    theta_result = expected_free_energy(model, stale_belief, action_index("fund_Theta"))
    hold_result = expected_free_energy(model, stale_belief, action_index("hold"))
    assert theta_result.epistemic > hold_result.epistemic


def test_pragmatic_value_rises_with_stronger_production_channels(model, prior) -> None:
    weak = {channel: vector.copy() for channel, vector in prior.items()}
    weak["I"] = np.array([0.9, 0.1], dtype=float)
    weak["U"] = np.array([0.9, 0.1], dtype=float)
    weak["Theta"] = np.array([0.9, 0.1], dtype=float)

    strong = {channel: vector.copy() for channel, vector in prior.items()}
    strong["I"] = np.array([0.1, 0.9], dtype=float)
    strong["U"] = np.array([0.1, 0.9], dtype=float)
    strong["Theta"] = np.array([0.1, 0.9], dtype=float)

    weak_result = expected_free_energy(model, weak, action_index("hold"))
    strong_result = expected_free_energy(model, strong, action_index("hold"))
    assert strong_result.pragmatic > weak_result.pragmatic


def test_policy_posterior_rejects_invalid_gamma(model, prior) -> None:
    with pytest.raises(ValueError, match="gamma"):
        policy_posterior(model, prior, gamma=0.0)


def test_expected_free_energy_rejects_invalid_action(model, prior) -> None:
    with pytest.raises(ValueError, match="action must be in"):
        expected_free_energy(model, prior, -1)
    with pytest.raises(ValueError, match="action must be in"):
        expected_free_energy(model, prior, 6)


def test_expected_free_energy_handles_zero_probability_observation_paths(prior) -> None:
    deterministic_model = default_model(
        A_R=np.array(
            [
                np.ones((2, 2, 2), dtype=float),
                np.zeros((2, 2, 2), dtype=float),
                np.zeros((2, 2, 2), dtype=float),
            ]
        ),
        A_L=np.array(
            [
                np.ones((2, 2), dtype=float),
                np.zeros((2, 2), dtype=float),
                np.zeros((2, 2), dtype=float),
            ]
        ),
    )
    result = expected_free_energy(deterministic_model, prior, action_index("hold"))
    assert result.epistemic == pytest.approx(0.0)
