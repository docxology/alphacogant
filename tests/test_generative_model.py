"""Tests for the AlphaCOGANT generative model."""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.model.generative_model import (
    EconomicWorldModel,
    belief_prior,
    default_model,
    infer_states,
)


def test_default_model_probabilities_and_belief_prior_copy(model) -> None:
    assert model.A_R.shape == (3, 2, 2, 2)
    assert model.A_L.shape == (3, 2, 2)
    assert np.allclose(model.A_R.sum(axis=0), 1.0)
    assert np.allclose(model.A_L.sum(axis=0), 1.0)
    for matrix in model.B.values():
        assert matrix.shape == (2, 2, 6)
        assert np.allclose(matrix.sum(axis=0), 1.0)
    for vector in model.D.values():
        assert vector.shape == (2,)
        assert np.allclose(vector.sum(), 1.0)

    detached_prior = belief_prior()
    detached_prior["I"][0] = 0.0
    assert model.D["I"][0] == pytest.approx(0.6)


def test_default_model_rejects_malformed_distributions(model) -> None:
    bad_a_r = model.A_R.copy()
    bad_a_r[0, 0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="A_R"):
        default_model(A_R=bad_a_r)

    bad_b = {channel: matrix.copy() for channel, matrix in model.B.items()}
    bad_b["Theta"][:, 0, 0] = 0.0
    with pytest.raises(ValueError, match="B\\[Theta\\]"):
        default_model(B=bad_b)

    bad_d = {channel: vector.copy() for channel, vector in model.D.items()}
    bad_d["Z"] = np.array([0.0, 0.0], dtype=float)
    with pytest.raises(ValueError, match="D\\[Z\\]"):
        default_model(D=bad_d)


def test_infer_states_moves_belief_toward_strong_on_good_observation(model, prior) -> None:
    posterior = infer_states(model, obs_R=2, obs_L=2, prior=prior)
    assert posterior["I"][1] > prior["I"][1]
    assert posterior["S"][1] > prior["S"][1]
    assert posterior["U"][1] > prior["U"][1]
    assert posterior["Theta"][1] > prior["Theta"][1]
    np.testing.assert_array_equal(posterior["Z"], prior["Z"])


def test_infer_states_rejects_bad_inputs(model, prior) -> None:
    with pytest.raises(ValueError, match="obs_R"):
        infer_states(model, obs_R=4, obs_L=2, prior=prior)
    with pytest.raises(ValueError, match="obs_L"):
        infer_states(model, obs_R=2, obs_L=-1, prior=prior)

    malformed_prior = {channel: vector.copy() for channel, vector in prior.items()}
    malformed_prior["I"] = np.array([0.0, 0.0], dtype=float)
    with pytest.raises(ValueError, match="prior\\[I\\]"):
        infer_states(model, obs_R=2, obs_L=2, prior=malformed_prior)

    missing_channel_prior = {channel: vector.copy() for channel, vector in prior.items()}
    missing_channel_prior.pop("Z")
    with pytest.raises(ValueError, match="missing channel priors"):
        infer_states(model, obs_R=2, obs_L=2, prior=missing_channel_prior)


def test_default_model_rejects_missing_keys_bad_shapes_and_non_finite_values(model) -> None:
    incomplete_b = {channel: matrix.copy() for channel, matrix in model.B.items() if channel != "I"}
    with pytest.raises(ValueError, match="missing channels"):
        default_model(B=incomplete_b)

    incomplete_d = {channel: vector.copy() for channel, vector in model.D.items() if channel != "I"}
    with pytest.raises(ValueError, match="missing channels"):
        default_model(D=incomplete_d)

    with pytest.raises(ValueError, match="shape"):
        default_model(C_R=np.array([1.0, 2.0], dtype=float))

    non_finite_a_l = model.A_L.copy()
    non_finite_a_l[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        default_model(A_L=non_finite_a_l)


def test_economic_world_model_constructor_rejects_missing_keys(model) -> None:
    with pytest.raises(ValueError, match="B must contain exactly"):
        EconomicWorldModel(
            A_R=model.A_R,
            A_L=model.A_L,
            B={channel: matrix.copy() for channel, matrix in model.B.items() if channel != "I"},
            C_R=model.C_R,
            C_L=model.C_L,
            D={channel: vector.copy() for channel, vector in model.D.items()},
        )

    with pytest.raises(ValueError, match="D must contain exactly"):
        EconomicWorldModel(
            A_R=model.A_R,
            A_L=model.A_L,
            B={channel: matrix.copy() for channel, matrix in model.B.items()},
            C_R=model.C_R,
            C_L=model.C_L,
            D={channel: vector.copy() for channel, vector in model.D.items() if channel != "I"},
        )


def test_rare_independent_observations_preserve_prior_instead_of_underflow(model, prior):
    reward = np.zeros_like(model.A_R)
    loss = np.zeros_like(model.A_L)
    reward[0] = loss[0] = 1e-200
    reward[1] = loss[1] = 1.0
    rare = default_model(A_R=reward, A_L=loss)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        posterior = infer_states(rare, 0, 0, prior)
    for channel in prior:
        np.testing.assert_allclose(posterior[channel], prior[channel], atol=2e-14, rtol=2e-14)


def test_rare_nonuniform_contractions_preserve_positive_support(model, prior):
    reward = np.zeros_like(model.A_R)
    reward[0, :, 1, 1] = [0.25, 0.5]
    reward[1] = 1.0 - reward[0]
    loss = np.zeros_like(model.A_L)
    loss[0] = 1.0
    rare = default_model(A_R=reward, A_L=loss)
    prior["I"] = np.array([0.5, 0.5])
    prior["U"] = np.array([1.0, 1e-200])
    prior["Theta"] = np.array([1.0, 1e-200])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        posterior = infer_states(rare, 0, 0, prior)
    # Reward requires rare U=Theta=1, with likelihood odds 1:2 over I.
    np.testing.assert_allclose(posterior["I"], [1 / 3, 2 / 3], rtol=2e-13, atol=0)
    np.testing.assert_array_equal(posterior["U"], [0.0, 1.0])
    np.testing.assert_array_equal(posterior["Theta"], [0.0, 1.0])
    np.testing.assert_allclose(posterior["S"], prior["S"], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(posterior["Z"], prior["Z"])


def test_truly_impossible_observation_is_not_replaced_with_prior(model, prior):
    reward = np.zeros_like(model.A_R)
    reward[1] = 1.0
    impossible = default_model(A_R=reward)
    with pytest.raises(ValueError, match="positive total probability"):
        infer_states(impossible, 0, 0, prior)
