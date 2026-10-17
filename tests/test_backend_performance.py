"""Differential numerical checks for batched scoring and validation-free internal paths."""

import subprocess
import sys

import numpy as np
import pytest

from alphacogant.efe.free_energy import efe_vector, expected_free_energy
from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import default_model
from alphacogant.trsi.t_rsi import create_rate, decay_rate, paired_bootstrap_samples


@pytest.mark.parametrize("seed", [7, 19, 83])
def test_batched_scores_agree_with_independent_action_evaluation(seed):
    rng = np.random.default_rng(seed)
    model = default_model(
        A_R=rng.random((3, 2, 2, 2)),
        A_L=rng.random((3, 2, 2)),
        B={c: rng.random((2, 2, 6)) for c in CHANNELS},
        C_R=rng.normal(size=3),
        C_L=rng.normal(size=3),
    )
    for _ in range(10):
        belief = {c: rng.dirichlet([0.3, 0.3]) for c in CHANNELS}
        actual = efe_vector(model, belief)
        expected = [expected_free_energy(model, belief, a) for a in range(6)]
        np.testing.assert_allclose(
            [[r.pragmatic, r.epistemic, r.total] for r in actual],
            [[r.pragmatic, r.epistemic, r.total] for r in expected],
            atol=2e-14,
            rtol=1e-13,
        )
        assert np.argmin([r.total for r in actual]) == np.argmin([r.total for r in expected])


@pytest.mark.parametrize("theta", [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
def test_batched_kl_accepts_sparse_likelihoods_and_priors(theta):
    reward = np.zeros((3, 2, 2, 2))
    loss = np.zeros((3, 2, 2))
    reward[0, :, :, 0] = 1
    reward[1, :, :, 1] = 1
    loss[1, :, 0] = 1
    loss[0, :, 1] = 1
    model = default_model(A_R=reward, A_L=loss)
    belief = {**model.D, "Theta": np.array(theta)}
    with np.errstate(all="raise"):
        actual = efe_vector(model, belief)
        expected = [expected_free_energy(model, belief, a) for a in range(6)]
    np.testing.assert_allclose([r.total for r in actual], [r.total for r in expected], atol=1e-14)


def test_bootstrap_preserves_draw_order_and_reference_rates(model, prior):
    rng = np.random.default_rng(7)
    actual = paired_bootstrap_samples(model, prior, rng, n=8, horizon=4)
    reference = np.random.default_rng(7)
    create, decay = [], []
    for _ in range(8):
        belief = {c: reference.dirichlet(np.clip(v, 1e-9, None) * 12 + 1) for c, v in prior.items()}
        create.append(create_rate(model, belief, 4))
        decay.append(decay_rate(model, belief, 4))
    np.testing.assert_array_equal(actual[0], create)
    np.testing.assert_array_equal(actual[1], decay)
    assert rng.random() == reference.random()


def test_backend_import_defers_plotting_and_keeps_public_plot_api():
    code = """import sys
import alphacogant
assert "matplotlib" not in sys.modules
assert "plot_style" in dir(alphacogant)
assert callable(alphacogant.plot_style.apply_style)
assert alphacogant.plot_style is alphacogant.plot_style
try:
    alphacogant.missing_attribute
except AttributeError:
    pass
else:
    raise AssertionError("Unknown attribute must fail")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_information_gain_stays_finite_for_subnormal_prior():
    from alphacogant.efe.free_energy import _epistemic_value

    theta = np.array([1e-310, 1.0])
    reward = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    loss = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        information = _epistemic_value(theta, reward, loss)
    # Reward reveals Theta exactly, so expected KL equals the prior entropy.
    expected = float(-np.sum(theta * np.log(theta)))
    assert information > 0
    np.testing.assert_allclose(information, expected, atol=0, rtol=1e-12)
