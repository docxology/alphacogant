"""Numerical boundaries that previously leaked NaNs or invalid indexing."""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.efe.free_energy import (
    _epistemic_value,
    expected_free_energy,
    policy_posterior,
    predicted_belief,
)
from alphacogant.model.generative_model import default_model, infer_states, validate_belief_map
from alphacogant.stats.simulation import simulate_trajectory
from alphacogant.stats.statistics import bootstrap_ci
from alphacogant.trsi.t_rsi import bootstrap_t_rsi, create_rate, decay_rate, t_rsi


def test_degenerate_theta_information_gain_has_no_invalid_arithmetic():
    with np.errstate(all="raise"):
        information = _epistemic_value(
            np.array([0.0, 1.0]),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        )
    assert information == 0.0


def test_extreme_finite_policy_scores_are_stable(model, prior):
    values = {0: 1e308, 1: -1e308, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    with np.errstate(all="raise"):
        posterior = policy_posterior(model, prior, gamma=1e308, values=values)
    np.testing.assert_array_equal(posterior, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("value", [-1, 6, 1.0, True, float("nan")])
def test_invalid_actions_fail_at_public_boundaries(model, prior, value):
    for function in (expected_free_energy, predicted_belief):
        with pytest.raises(ValueError, match="action"):
            function(model, prior, value)


@pytest.mark.parametrize("value", [1.0, True, -1, 3, float("nan")])
def test_observations_require_integer_buckets(model, prior, value):
    with pytest.raises(ValueError, match="obs_R"):
        infer_states(model, value, 0, prior)
    with pytest.raises(ValueError, match="obs_L"):
        infer_states(model, 0, value, prior)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), float("inf")])
def test_loop_bounds_require_positive_integers(model, prior, value):
    for function in (create_rate, decay_rate, simulate_trajectory):
        with pytest.raises(ValueError, match="horizon"):
            function(model, prior, horizon=value)
    with pytest.raises(ValueError, match="n"):
        bootstrap_t_rsi(model, prior, np.random.default_rng(1), n=value)


@pytest.mark.parametrize("concentration", [float("nan"), float("inf"), -1.0, 0.0])
def test_nonfinite_bootstrap_precision_rejected(model, prior, concentration):
    with pytest.raises(ValueError, match="concentration"):
        bootstrap_t_rsi(model, prior, np.random.default_rng(1), n=2, concentration=concentration)


def test_pairing_is_checked_before_single_sample_fallback():
    with pytest.raises(ValueError, match="paired"):
        t_rsi([1.0], [1.0, 2.0])


@pytest.mark.parametrize("samples", [[1.0, float("nan")], [1.0, float("inf")], [[1.0, 2.0]]])
def test_ci_rejects_nonfinite_and_nonvector_samples(samples):
    with pytest.raises(ValueError):
        bootstrap_ci(samples)


def test_unknown_channel_keys_rejected(model, prior):
    with pytest.raises(ValueError, match="unknown"):
        validate_belief_map({**prior, "typo": [0.5, 0.5]})
    with pytest.raises(ValueError, match="unknown"):
        default_model(B={**model.B, "typo": model.B["I"]})
    with pytest.raises(ValueError, match="unknown"):
        default_model(D={**model.D, "typo": model.D["I"]})
    with pytest.raises(ValueError, match="mapping"):
        validate_belief_map(None)


@pytest.mark.parametrize("values", [{}, {i: float("nan") for i in range(6)}])
def test_precomputed_policy_scores_are_complete_and_finite(model, prior, values):
    with pytest.raises(ValueError):
        policy_posterior(model, prior, values=values)
