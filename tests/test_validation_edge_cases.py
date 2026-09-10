"""Tests for defensive validation paths in the engine.

These tests exercise error-handling code paths that the main suite
(the happy-path tests) does not cover: malformed observations and the
KL-nonnegativity guard. The probability-validator matrix lives in
test_validation_paths.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.efe.free_energy import _kl_divergence
from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import infer_states


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


def test_infer_states_rejects_bad_observation(model):
    """infer_states rejects observation indices outside {0, 1, 2}."""
    prior = {channel: model.D[channel].copy() for channel in CHANNELS}
    with pytest.raises(ValueError, match="obs_R"):
        infer_states(model, 5, 0, prior)
    with pytest.raises(ValueError, match="obs_L"):
        infer_states(model, 0, 5, prior)
