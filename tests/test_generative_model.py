"""Tests for the AlphaCOGANT generative model."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from alphacogant.bridge.cogant_bridge import model_to_gnn_summary
from alphacogant.bridge.gnn import joint_matrices, model_from_gnn, model_parameters
from alphacogant.efe.free_energy import predicted_belief
from alphacogant.model.channels import ACTIONS, CHANNELS
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

    original = {channel: vector.copy() for channel, vector in model.D.items()}
    detached_prior = belief_prior(model)
    for channel in CHANNELS:
        detached_prior[channel][:] = [1.0, 0.0]
        np.testing.assert_array_equal(model.D[channel], original[channel])


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


@pytest.mark.parametrize("representation", ["integer", "list"])
def test_constructor_canonicalizes_coefficients_for_prediction_filtering_and_joint(representation):
    reward = np.zeros((3, 2, 2, 2), dtype=int)
    loss = np.zeros((3, 2, 2), dtype=int)
    for i, u, theta in product(range(2), repeat=3):
        reward[(i + u + theta) % 3, i, u, theta] = 1
    for s, theta in product(range(2), repeat=2):
        loss[s + theta, s, theta] = 1
    transitions = {}
    priors = {}
    for k, channel in enumerate(CHANNELS):
        matrix = np.zeros((2, 2, len(ACTIONS)), dtype=int)
        for current, action in product(range(2), range(len(ACTIONS))):
            matrix[(current + action + k) % 2, current, action] = 1
        transitions[channel] = matrix
        priors[channel] = np.eye(2, dtype=int)[k % 2]
    reward_preferences = np.array([-3, 0, 5])
    loss_preferences = np.array([-2, 1, 4])

    def supplied(array):
        return array.tolist() if representation == "list" else array

    caller_b = {channel: supplied(array) for channel, array in transitions.items()}
    caller_d = {channel: supplied(array) for channel, array in priors.items()}
    original_b = caller_b.copy()
    original_d = caller_d.copy()
    constructed = EconomicWorldModel(
        A_R=supplied(reward),
        A_L=supplied(loss),
        B=caller_b,
        C_R=supplied(reward_preferences),
        C_L=supplied(loss_preferences),
        D=caller_d,
    )
    assert constructed.B is not caller_b
    assert constructed.D is not caller_d
    for channel in CHANNELS:
        assert caller_b[channel] is original_b[channel]
        assert caller_d[channel] is original_d[channel]
        np.testing.assert_array_equal(caller_b[channel], transitions[channel])
        np.testing.assert_array_equal(caller_d[channel], priors[channel])

    explicit_float = EconomicWorldModel(
        A_R=reward.astype(float),
        A_L=loss.astype(float),
        B={channel: array.astype(float) for channel, array in transitions.items()},
        C_R=reward_preferences.astype(float),
        C_L=loss_preferences.astype(float),
        D={channel: array.astype(float) for channel, array in priors.items()},
    )
    restored = model_from_gnn(model_to_gnn_summary(constructed))
    expected_coefficients = {
        "A_R": reward,
        "A_L": loss,
        **{f"B_{channel}": array for channel, array in transitions.items()},
        "C_R": reward_preferences,
        "C_L": loss_preferences,
        **{f"D_{channel}": array for channel, array in priors.items()},
    }
    prior = dict(zip(
        CHANNELS,
        map(np.array, ([0.25, 0.75], [0.6, 0.4], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7])),
        strict=True,
    ))
    # R=0 permits only (I,U,Theta)=(0,0,0) or (1,1,1);
    # L=1 requires S != Theta. These are one-step mean-field marginals.
    expected_posterior = dict(zip(
        CHANNELS,
        ([8 / 17, 9 / 17], [3 / 11, 8 / 11], [8 / 17, 9 / 17],
         [16 / 43, 27 / 43], [0.3, 0.7]),
        strict=True,
    ))
    states = list(product(range(2), repeat=len(CHANNELS)))
    expected_a = np.zeros((9, 32))
    expected_b = np.zeros((32, 32, len(ACTIONS)))
    expected_d = np.zeros(32)
    expected_d[states.index((0, 1, 0, 1, 0))] = 1.0
    for current, (i, s, u, theta, _z) in enumerate(states):
        expected_a[3 * ((i + u + theta) % 3) + s + theta, current] = 1.0
        for action in range(len(ACTIONS)):
            successor = tuple((bit + action + k) % 2 for k, bit in enumerate(states[current]))
            expected_b[states.index(successor), current, action] = 1.0

    for candidate in (constructed, explicit_float, restored):
        for name, array in model_parameters(candidate).items():
            assert array.dtype == np.dtype(float)
            np.testing.assert_array_equal(array, expected_coefficients[name])
        for action in range(len(ACTIONS)):
            prediction = predicted_belief(candidate, prior, action)
            for k, channel in enumerate(CHANNELS):
                expected = prior[channel][::-1] if (action + k) % 2 else prior[channel]
                np.testing.assert_array_equal(prediction[channel], expected)
        posterior = infer_states(candidate, 0, 1, prior)
        for channel in CHANNELS:
            np.testing.assert_allclose(posterior[channel], expected_posterior[channel], rtol=2e-14)
        joint = joint_matrices(candidate)
        np.testing.assert_array_equal(joint["A"], expected_a)
        np.testing.assert_array_equal(joint["B"], expected_b)
        np.testing.assert_array_equal(joint["C"], [-5, -2, 1, -2, 1, 4, 3, 6, 9])
        np.testing.assert_array_equal(joint["D"], expected_d)


def test_constructor_shares_float_views_but_not_caller_mappings(model):
    reward_view = model.A_R.view()
    caller_b = model.B.copy()
    caller_d = model.D.copy()
    constructed = EconomicWorldModel(
        A_R=reward_view,
        A_L=model.A_L,
        B=caller_b,
        C_R=model.C_R,
        C_L=model.C_L,
        D=caller_d,
    )
    assert constructed.A_R is reward_view
    assert constructed.A_L is model.A_L
    assert constructed.C_R is model.C_R
    assert constructed.C_L is model.C_L
    assert constructed.B is not caller_b
    assert constructed.D is not caller_d
    for channel in CHANNELS:
        assert constructed.B[channel] is caller_b[channel]
        assert constructed.D[channel] is caller_d[channel]
    caller_b["I"] = np.zeros_like(model.B["I"])
    caller_d["I"] = np.zeros(2)
    np.testing.assert_array_equal(constructed.B["I"], model.B["I"])
    np.testing.assert_array_equal(constructed.D["I"], model.D["I"])
    constructed.C_R[0] = -7.0
    assert model.C_R[0] == -7.0


@pytest.mark.parametrize("as_lists", [False, True])
def test_integer_constant_likelihoods_leave_filtering_prior_unchanged(model, prior, as_lists):
    reward = np.zeros((3, 2, 2, 2), dtype=int)
    loss = np.zeros((3, 2, 2), dtype=int)
    reward[0] = 1
    loss[0] = 1
    constructed = EconomicWorldModel(
        A_R=reward.tolist() if as_lists else reward,
        A_L=loss.tolist() if as_lists else loss,
        B=model.B,
        C_R=model.C_R,
        C_L=model.C_L,
        D=model.D,
    )
    posterior = infer_states(constructed, 0, 0, prior)
    for channel in CHANNELS:
        np.testing.assert_allclose(posterior[channel], prior[channel], rtol=2e-14)
