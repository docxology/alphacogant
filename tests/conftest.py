"""Shared fixtures for deterministic AlphaCOGANT tests.

Seeding convention: tests that need randomness construct their own
``np.random.default_rng(<explicit seed>)`` at the call site so each test's
stream is independent and auditable; the ``seeded_rng`` fixture exists for
tests that want one shared fixed-seed generator without picking a seed.
"""

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
