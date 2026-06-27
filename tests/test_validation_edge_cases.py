"""Tests for defensive validation paths in the engine.

These tests exercise the error-handling code paths that the main suite
(the happy-path tests) does not cover: malformed matrices, negative
probabilities, and the KL-nonnegativity guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.efe.free_energy import _kl_divergence
from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import (
    _validate_probability_columns,
    _validate_probability_vector,
    default_model,
    infer_states,
)


def test_validate_probability_columns_rejects_negative():
    """_validate_probability_columns raises on negative probabilities."""
    bad = np.array([[1.2, 0.5], [-0.2, 0.5]])  # column 0 sums to 1, but has -0.2
    with pytest.raises(ValueError, match="negative probabilities"):
        _validate_probability_columns("test", bad)


def test_validate_probability_columns_rejects_non_normalized():
    """_validate_probability_columns raises on columns that don't sum to 1."""
    bad = np.array([[0.5, 0.5], [0.3, 0.5]])  # column 0 sums to 0.8
    with pytest.raises(ValueError, match="columns must sum to 1"):
        _validate_probability_columns("test", bad)


def test_validate_probability_vector_rejects_negative():
    """_validate_probability_vector raises on negative probabilities."""
    bad = np.array([1.5, -0.5])
    with pytest.raises(ValueError, match="negative probabilities"):
        _validate_probability_vector("test", bad)


def test_validate_probability_vector_rejects_non_normalized():
    """_validate_probability_vector raises on vectors that don't sum to 1."""
    bad = np.array([0.3, 0.3])
    with pytest.raises(ValueError, match="must sum to 1"):
        _validate_probability_vector("test", bad)


def test_kl_divergence_zero_for_identical():
    """KL divergence is 0 for identical distributions."""
    p = np.array([0.5, 0.5])
    assert abs(_kl_divergence(p, p)) < 1e-10


def test_kl_divergence_positive_for_different():
    """KL divergence is positive for different distributions."""
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    assert _kl_divergence(p, q) > 0


def test_kl_divergence_handles_zeros():
    """KL divergence handles zero entries in posterior (masked out)."""
    p = np.array([0.0, 1.0])  # zero in first entry
    q = np.array([0.5, 0.5])
    # Should not raise; the zero is masked
    result = _kl_divergence(p, q)
    assert result > 0  # KL(1.0 || 0.5) = ln(2) > 0


def test_infer_states_rejects_bad_observation():
    """infer_states rejects observation indices outside {0, 1, 2}."""
    model = default_model()
    prior = {channel: model.D[channel].copy() for channel in CHANNELS}
    with pytest.raises(ValueError, match="obs_R"):
        infer_states(model, 5, 0, prior)
    with pytest.raises(ValueError, match="obs_L"):
        infer_states(model, 0, 5, prior)
