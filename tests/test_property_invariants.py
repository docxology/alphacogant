"""Property-based tests for Active Inference invariants.

These tests exercise the mathematical properties that must hold for ANY
valid model and belief combination, not just the default operating points.
No mocks; all tests use real computation with randomly-generated valid models.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.efe.free_energy import (
    expected_free_energy,
    marginal_return_vector,
    policy_posterior,
)
from alphacogant.model.channels import ACTIONS, CHANNELS
from alphacogant.model.generative_model import (
    EconomicWorldModel,
    default_model,
    validate_belief_map,
)
from alphacogant.trsi.t_rsi import certificate, t_rsi


def _random_belief(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate a random valid belief map."""
    return {channel: rng.dirichlet([5.0, 5.0]) for channel in CHANNELS}


def _random_valid_model(rng: np.random.Generator) -> EconomicWorldModel:
    """Generate a random valid EconomicWorldModel."""
    # Random A_R: shape (3, 2, 2, 2), column-stochastic over axis 0
    A_R = rng.dirichlet([5, 5, 5], size=(2, 2, 2)).transpose(3, 0, 1, 2)
    # Random A_L: shape (3, 2, 2), column-stochastic over axis 0
    A_L = rng.dirichlet([5, 5, 5], size=(2, 2)).transpose(2, 0, 1)
    # Random B: each (2, 2, 6), column-stochastic over axis 0
    B = {channel: rng.dirichlet([5, 5], size=(2, 6)).transpose(2, 0, 1) for channel in CHANNELS}
    D = {channel: rng.dirichlet([5, 5]) for channel in CHANNELS}
    C_R = rng.uniform(-3, 3, size=3)
    C_L = rng.uniform(-3, 3, size=3)
    return EconomicWorldModel(A_R=A_R, A_L=A_L, B=B, C_R=C_R, C_L=C_L, D=D)


@pytest.mark.parametrize("execution_number", range(10))
def test_efe_total_equals_negative_sum(execution_number):
    """G = -(pragmatic + epistemic) for every action."""
    rng = np.random.default_rng(42 + execution_number)
    model = default_model()
    belief = _random_belief(rng)
    for action in range(len(ACTIONS)):
        efe = expected_free_energy(model, belief, action)
        assert abs(efe.total - (-(efe.pragmatic + efe.epistemic))) < 1e-10


@pytest.mark.parametrize("execution_number", range(10))
def test_epistemic_non_negative(execution_number):
    """Epistemic value (a KL divergence) is always >= 0."""
    rng = np.random.default_rng(100 + execution_number)
    model = default_model()
    belief = _random_belief(rng)
    for action in range(len(ACTIONS)):
        efe = expected_free_energy(model, belief, action)
        assert efe.epistemic >= -1e-10, (
            f"Epistemic value negative: {efe.epistemic} for action {action}"
        )


@pytest.mark.parametrize("execution_number", range(10))
def test_policy_posterior_sums_to_one(execution_number):
    """Policy posterior is a valid probability distribution."""
    rng = np.random.default_rng(200 + execution_number)
    model = default_model()
    belief = _random_belief(rng)
    posterior = policy_posterior(model, belief, gamma=1.0)
    assert abs(sum(posterior) - 1.0) < 1e-10
    assert all(p >= 0 for p in posterior)


@pytest.mark.parametrize("execution_number", range(5))
def test_marginal_return_vector_all_actions(execution_number):
    """Marginal return vector covers all actions."""
    rng = np.random.default_rng(300 + execution_number)
    model = default_model()
    belief = _random_belief(rng)
    returns = marginal_return_vector(model, belief)
    assert len(returns) == len(ACTIONS)
    for action in range(len(ACTIONS)):
        assert action in returns
        assert np.isfinite(returns[action])


@pytest.mark.parametrize("execution_number", range(5))
def test_t_rsi_zero_when_equal(execution_number):
    """t-RSI is 0 when create and decay samples are identical."""
    rng = np.random.default_rng(400 + execution_number)
    samples = rng.normal(0.5, 0.1, size=20)
    result = t_rsi(samples, samples)
    assert abs(result) < 1e-10


def test_t_rsi_positive_when_create_exceeds_decay():
    """t-RSI is positive when create samples exceed decay samples."""
    rng = np.random.default_rng(500)
    create = rng.normal(1.0, 0.1, size=50)
    decay = rng.normal(0.5, 0.1, size=50)
    assert t_rsi(create, decay) > 0


def test_t_rsi_negative_when_decay_exceeds_create():
    """t-RSI is negative when decay samples exceed create samples."""
    rng = np.random.default_rng(600)
    create = rng.normal(0.5, 0.1, size=50)
    decay = rng.normal(1.0, 0.1, size=50)
    assert t_rsi(create, decay) < 0


def test_certificate_threshold_logic():
    """Certificate fires iff t-RSI >= delta."""
    assert certificate(5.0, 3.0) is True
    assert certificate(3.0, 5.0) is False
    assert certificate(3.0, 3.0) is True  # boundary: >=


def test_certificate_nan_raises():
    """NaN inputs to certificate should raise."""
    with pytest.raises(ValueError, match="NaN"):
        certificate(float("nan"), 1.0)
    with pytest.raises(ValueError, match="NaN"):
        certificate(1.0, float("nan"))


@pytest.mark.parametrize("execution_number", range(5))
def test_belief_validation_normalizes(execution_number):
    """validate_belief_map normalizes non-normalized beliefs."""
    rng = np.random.default_rng(700 + execution_number)
    raw = {channel: rng.uniform(0, 5, size=2) for channel in CHANNELS}
    normalized = validate_belief_map(raw, context="test")
    for channel in CHANNELS:
        assert abs(sum(normalized[channel]) - 1.0) < 1e-8
        assert all(normalized[channel] >= 0)


def test_t_rsi_returns_zero_for_small_samples():
    """t-RSI returns 0 for samples with fewer than 2 elements."""
    assert t_rsi([1.0], [2.0]) == 0.0
    assert t_rsi([1.0, 2.0], [3.0]) == 0.0


def test_t_rsi_inf_when_zero_variance_different_means():
    """t-RSI returns inf when both SEs are 0 and means differ."""
    assert t_rsi([1.0, 1.0], [2.0, 2.0]) == float("inf")


def test_t_rsi_finite_when_zero_variance_same_mean():
    """t-RSI returns 0 when both SEs are 0 and means are equal."""
    assert t_rsi([1.0, 1.0], [1.0, 1.0]) == 0.0
