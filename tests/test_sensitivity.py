"""Tests for the sensitivity module: t-RSI sweeps.

No mocks; all tests use real model computation and fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pytest

from alphacogant.generative_model import default_model
from alphacogant.operating_points import IMPROVING
from alphacogant.sensitivity import sweep_concentration, sweep_theta_freshness


def test_sweep_concentration_returns_arrays():
    model = default_model()
    result = sweep_concentration(model, IMPROVING, n=32)
    assert "concentrations" in result
    assert "t_rsi" in result
    assert "create_mean" in result
    assert "decay_mean" in result
    assert len(result["concentrations"]) == len(result["t_rsi"])
    assert len(result["t_rsi"]) == len(result["create_mean"])
    assert len(result["create_mean"]) == len(result["decay_mean"])


def test_sweep_concentration_default_values():
    model = default_model()
    result = sweep_concentration(model, IMPROVING, n=16)
    concentrations = result["concentrations"]
    assert len(concentrations) == 7
    assert concentrations[0] == 2.0
    assert concentrations[-1] == 80.0


def test_sweep_concentration_deterministic():
    model = default_model()
    r1 = sweep_concentration(model, IMPROVING, n=32)
    r2 = sweep_concentration(model, IMPROVING, n=32)
    assert np.allclose(r1["t_rsi"], r2["t_rsi"])


def test_sweep_concentration_custom_values():
    model = default_model()
    custom = np.array([5.0, 10.0, 15.0])
    result = sweep_concentration(model, IMPROVING, concentrations=custom, n=16)
    assert len(result["t_rsi"]) == 3
    assert result["concentrations"][0] == 5.0


def test_sweep_theta_freshness_returns_arrays():
    model = default_model()
    result = sweep_theta_freshness(model, IMPROVING, n=32)
    assert "theta_p_fresh" in result
    assert "t_rsi" in result
    assert "create_mean" in result
    assert "decay_mean" in result
    assert len(result["theta_p_fresh"]) == len(result["t_rsi"])


def test_sweep_theta_freshness_default_values():
    model = default_model()
    result = sweep_theta_freshness(model, IMPROVING, n=16)
    theta_vals = result["theta_p_fresh"]
    assert len(theta_vals) == 19  # linspace(0.05, 0.95, 19)
    assert pytest.approx(theta_vals[0]) == 0.05
    assert pytest.approx(theta_vals[-1]) == 0.95


def test_sweep_theta_freshness_deterministic():
    model = default_model()
    r1 = sweep_theta_freshness(model, IMPROVING, n=32)
    r2 = sweep_theta_freshness(model, IMPROVING, n=32)
    assert np.allclose(r1["t_rsi"], r2["t_rsi"])


def test_sweep_theta_freshness_custom_values():
    model = default_model()
    custom = np.array([0.1, 0.5, 0.9])
    result = sweep_theta_freshness(model, IMPROVING, theta_values=custom, n=16)
    assert len(result["t_rsi"]) == 3


def test_sweep_concentration_all_finite():
    model = default_model()
    result = sweep_concentration(model, IMPROVING, n=16)
    assert np.all(np.isfinite(result["t_rsi"]))
    assert np.all(np.isfinite(result["create_mean"]))
    assert np.all(np.isfinite(result["decay_mean"]))


def test_sweep_theta_freshness_all_finite():
    model = default_model()
    result = sweep_theta_freshness(model, IMPROVING, n=16)
    assert np.all(np.isfinite(result["t_rsi"]))
    assert np.all(np.isfinite(result["create_mean"]))
    assert np.all(np.isfinite(result["decay_mean"]))
