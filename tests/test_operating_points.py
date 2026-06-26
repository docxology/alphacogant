"""Tests for the shared operating-points module and its cross-file consistency."""

from __future__ import annotations

import numpy as np

from alphacogant.channels import CHANNELS
from alphacogant.operating_points import (
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COASTING,
    COASTING_RAW,
    IMPROVING,
    IMPROVING_RAW,
    as_belief,
)


def test_improving_and_coasting_have_all_channels() -> None:
    for belief in (IMPROVING, COASTING):
        assert set(belief.keys()) == set(CHANNELS)
        for channel in CHANNELS:
            assert belief[channel].shape == (2,)
            assert np.allclose(belief[channel].sum(), 1.0)


def test_raw_specs_match_belief_maps() -> None:
    for raw, belief in (
        (IMPROVING_RAW, IMPROVING),
        (COASTING_RAW, COASTING),
    ):
        for channel in CHANNELS:
            assert np.allclose(belief[channel], np.array(raw[channel], dtype=float))


def test_as_belief_returns_independent_arrays() -> None:
    belief = as_belief(IMPROVING_RAW)
    belief["I"][0] = 999.0
    # The module-level constant must not be mutated.
    assert IMPROVING["I"][0] != 999.0


def test_bootstrap_constants_are_canonical() -> None:
    assert BOOTSTRAP_SEED == 20240623
    assert BOOTSTRAP_N == 256
    assert BOOTSTRAP_CONCENTRATION == 12.0


def test_improving_is_weak_coasting_is_strong() -> None:
    """The two operating points must be genuinely different regimes."""
    for channel in CHANNELS:
        assert IMPROVING[channel][1] < 0.5  # weak
        assert COASTING[channel][1] > 0.5  # strong
