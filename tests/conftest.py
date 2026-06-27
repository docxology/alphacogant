"""Shared fixtures for deterministic AlphaCOGANT tests."""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.model.generative_model import belief_prior, default_model


@pytest.fixture()
def model():
    """Return the default economic world model."""
    return default_model()


@pytest.fixture()
def prior(model):
    """Return the default belief prior."""
    return belief_prior(model)


@pytest.fixture()
def seeded_rng():
    """Return a deterministic generator for synthetic perturbations."""
    return np.random.default_rng(7)
