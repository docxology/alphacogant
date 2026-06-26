"""Tests for the statistics module: bootstrap CIs, regime comparison, effect sizes.

No mocks; all tests use real model computation and fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pytest

from alphacogant.generative_model import default_model
from alphacogant.operating_points import IMPROVING, COASTING
from alphacogant.statistics import (
    BootstrapCI,
    RegimeComparison,
    RegimeStatistics,
    bootstrap_ci,
    compare_regimes,
    compute_regime_statistics,
)


# ── bootstrap_ci ─────────────────────────────────────────────────────────────

def test_bootstrap_ci_basic():
    rng = np.random.default_rng(42)
    samples = rng.normal(0.5, 0.1, size=100)
    ci = bootstrap_ci(samples)
    assert isinstance(ci, BootstrapCI)
    assert ci.n == 100
    assert ci.mean > 0.3
    assert ci.mean < 0.7
    assert ci.ci_lower < ci.mean < ci.ci_upper
    assert ci.ci_width > 0
    assert ci.se > 0


def test_bootstrap_ci_confidence_level():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, size=200)
    ci95 = bootstrap_ci(samples, confidence=0.95)
    ci99 = bootstrap_ci(samples, confidence=0.99)
    assert ci99.ci_width >= ci95.ci_width


def test_bootstrap_ci_too_few_samples():
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci(np.array([1.0]))


def test_bootstrap_ci_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(np.array([1.0, 2.0]), confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(np.array([1.0, 2.0]), confidence=1.5)


# ── compute_regime_statistics ────────────────────────────────────────────────

def test_compute_regime_statistics_improving():
    model = default_model()
    stats = compute_regime_statistics(model, IMPROVING, "Improving", n=32)
    assert isinstance(stats, RegimeStatistics)
    assert stats.name == "Improving"
    assert isinstance(stats.t_rsi, float)
    assert np.isfinite(stats.t_rsi)
    assert isinstance(stats.create_ci, BootstrapCI)
    assert isinstance(stats.decay_ci, BootstrapCI)
    assert stats.funded_channel is not None
    assert isinstance(stats.funded_pragmatic, float)
    assert isinstance(stats.funded_epistemic, float)
    assert 0 <= stats.exploration_ratio <= 1


def test_compute_regime_statistics_coasting():
    model = default_model()
    stats = compute_regime_statistics(model, COASTING, "Coasting", n=32)
    assert stats.name == "Coasting"
    assert np.isfinite(stats.t_rsi)


def test_compute_regime_statistics_deterministic():
    model = default_model()
    s1 = compute_regime_statistics(model, IMPROVING, n=32)
    s2 = compute_regime_statistics(model, IMPROVING, n=32)
    assert s1.t_rsi == s2.t_rsi
    assert s1.create_ci.mean == s2.create_ci.mean


def test_compute_regime_statistics_default_model():
    stats = compute_regime_statistics(n=32)
    assert stats.name == "operating_point"
    assert np.isfinite(stats.t_rsi)


def test_compute_regime_statistics_as_dict():
    model = default_model()
    stats = compute_regime_statistics(model, IMPROVING, "test", n=32)
    d = stats.as_dict()
    assert d["name"] == "test"
    assert "t_rsi" in d
    assert "create_mean" in d
    assert "create_ci" in d
    assert len(d["create_ci"]) == 2


def test_compute_regime_statistics_efe_decomposition():
    model = default_model()
    stats = compute_regime_statistics(model, IMPROVING, n=32)
    assert len(stats.efe_pragmatic_per_action) == 6
    assert len(stats.efe_epistemic_per_action) == 6
    for v in stats.efe_epistemic_per_action.values():
        assert v >= -1e-10  # epistemic is always >= 0


# ── compare_regimes ─────────────────────────────────────────────────────────

def test_compare_regimes():
    comparison = compare_regimes(n=32)
    assert isinstance(comparison, RegimeComparison)
    assert comparison.improving.name == "Improving"
    assert comparison.coasting.name == "Coasting"


def test_compare_regimes_deltas():
    comparison = compare_regimes(n=32)
    assert isinstance(comparison.t_rsi_delta, float)
    assert isinstance(comparison.create_rate_delta, float)
    assert isinstance(comparison.decay_rate_delta, float)
    assert np.isfinite(comparison.t_rsi_delta)


def test_compare_regimes_cohen_d():
    comparison = compare_regimes(n=32)
    assert isinstance(comparison.cohen_d_create, float)
    assert isinstance(comparison.cohen_d_decay, float)
    assert np.isfinite(comparison.cohen_d_create)
    assert np.isfinite(comparison.cohen_d_decay)


def test_compare_regimes_as_table():
    comparison = compare_regimes(n=32)
    table = comparison.as_table()
    assert isinstance(table, str)
    assert "t-RSI" in table
    assert "Create-rate" in table
    assert "Decay-rate" in table
    assert "Cohen" in table
    assert "Improving" in table
    assert "Coasting" in table


def test_compare_regimes_deterministic():
    c1 = compare_regimes(n=32)
    c2 = compare_regimes(n=32)
    assert c1.improving.t_rsi == c2.improving.t_rsi
    assert c1.coasting.t_rsi == c2.coasting.t_rsi


def test_bootstrap_ci_property_ci_contains_mean():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, size=500)
    ci = bootstrap_ci(samples, confidence=0.95)
    assert ci.ci_lower < ci.mean < ci.ci_upper
