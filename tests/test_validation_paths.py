"""Tests for validation error paths in the generative-model module.

These cover the ``_validate_probability_*`` helpers that are reached when the
``default_model`` path has already normalized input but direct construction of
``EconomicWorldModel`` bypasses normalization and must reject malformed arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.generative_model import EconomicWorldModel, default_model


def _valid_matrices(model):
    """Return a deep copy of every matrix from a valid model."""
    return (
        model.A_R.copy(),
        model.A_L.copy(),
        {k: v.copy() for k, v in model.B.items()},
        model.C_R.copy(),
        model.C_L.copy(),
        {k: v.copy() for k, v in model.D.items()},
    )


def test_validate_probability_columns_rejects_negative(model) -> None:
    bad_b = {k: v.copy() for k, v in model.B.items()}
    bad_b["I"][0, 0, 0] = -0.5
    with pytest.raises(ValueError, match="B\\[I\\]"):
        default_model(B=bad_b)


def test_validate_probability_columns_rejects_unnormalized(model) -> None:
    bad_b = {k: v.copy() for k, v in model.B.items()}
    bad_b["S"][0, 0, 0] = 0.0
    bad_b["S"][1, 0, 0] = 0.0
    with pytest.raises(ValueError, match="B\\[S\\]"):
        default_model(B=bad_b)


def test_validate_probability_vector_rejects_negative(model) -> None:
    bad_d = {k: v.copy() for k, v in model.D.items()}
    bad_d["U"] = np.array([-0.1, 1.1], dtype=float)
    with pytest.raises(ValueError, match="D\\[U\\]"):
        default_model(D=bad_d)


def test_validate_probability_vector_rejects_zero_sum(model) -> None:
    bad_d = {k: v.copy() for k, v in model.D.items()}
    bad_d["Z"] = np.array([0.0, 0.0], dtype=float)
    with pytest.raises(ValueError, match="D\\[Z\\]"):
        default_model(D=bad_d)


def test_validate_belief_map_rejects_negative(model, prior) -> None:
    from alphacogant.generative_model import infer_states

    bad_prior = {k: v.copy() for k, v in prior.items()}
    bad_prior["Theta"] = np.array([-0.1, 1.1], dtype=float)
    with pytest.raises(ValueError, match="prior\\[Theta\\]"):
        infer_states(model, obs_R=1, obs_L=1, prior=bad_prior)


def test_constructor_rejects_non_column_stochastic_a_r(model) -> None:
    a_r, a_l, b, c_r, c_l, d = _valid_matrices(model)
    bad_a_r = a_r.copy()
    bad_a_r[0, 0, 0, 0] = 10.0
    with pytest.raises(ValueError, match="A_R"):
        EconomicWorldModel(
            A_R=bad_a_r, A_L=a_l, B=b, C_R=c_r, C_L=c_l, D=d,
        )


def test_constructor_rejects_non_column_stochastic_a_l(model) -> None:
    a_r, a_l, b, c_r, c_l, d = _valid_matrices(model)
    bad_a_l = a_l.copy()
    bad_a_l[0, 0, 0] = 10.0
    with pytest.raises(ValueError, match="A_L"):
        EconomicWorldModel(
            A_R=a_r, A_L=bad_a_l, B=b, C_R=c_r, C_L=c_l, D=d,
        )


def test_constructor_rejects_non_normalized_d(model) -> None:
    a_r, a_l, b, c_r, c_l, d = _valid_matrices(model)
    bad_d = {k: v.copy() for k, v in d.items()}
    bad_d["I"] = np.array([0.3, 0.3], dtype=float)
    with pytest.raises(ValueError, match="D\\[I\\]"):
        EconomicWorldModel(
            A_R=a_r, A_L=a_l, B=b, C_R=c_r, C_L=c_l, D=bad_d,
        )
