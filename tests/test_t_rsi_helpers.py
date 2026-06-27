"""Tests for t_rsi internal helpers and edge cases."""

from __future__ import annotations

import numpy as np

from alphacogant.trsi.t_rsi import _standard_error


def test_standard_error_returns_zero_for_single_sample() -> None:
    assert _standard_error(np.array([42.0])) == 0.0


def test_standard_error_returns_zero_for_empty_array() -> None:
    assert _standard_error(np.array([], dtype=float)) == 0.0


def test_standard_error_matches_manual_formula() -> None:
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = float(np.std(samples, ddof=1) / np.sqrt(samples.size))
    assert _standard_error(samples) == expected
