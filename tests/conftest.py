"""Shared fixtures for deterministic AlphaCOGANT tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from alphacogant.generative_model import belief_prior, default_model


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
