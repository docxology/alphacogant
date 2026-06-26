"""Tests for t-RSI, bootstrap determinism, and the certificate gate.

The central property under test is that the create/decay comparator is NOT
green-by-construction: t-RSI must be able to change sign across operating points,
so the certificate of monotone improvement can honestly fire AND fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.generative_model import belief_prior, default_model
from alphacogant.operating_points import IMPROVING as _IMPROVING, COASTING as _COASTING
from alphacogant.t_rsi import bootstrap_t_rsi, certificate, create_rate, decay_rate, t_rsi


def test_decay_rate_is_non_negative_everywhere(model, prior) -> None:
    for belief in (prior, _IMPROVING, _COASTING):
        assert decay_rate(model, belief) >= 0.0


def test_comparator_is_not_green_by_construction(model) -> None:
    """create vs decay must order BOTH ways across beliefs (the negative control).

    If create were >= decay by construction, t-RSI could never go negative and the
    certificate could never honestly fail. The self-improving operating point and
    the coasting operating point must therefore order oppositely.
    """
    improving_create = create_rate(model, _IMPROVING)
    improving_decay = decay_rate(model, _IMPROVING)
    coasting_create = create_rate(model, _COASTING)
    coasting_decay = decay_rate(model, _COASTING)

    assert improving_create > improving_decay  # active policy out-creates decay
    assert coasting_create < coasting_decay  # greedy stops creating; decay continues
    assert (improving_create - improving_decay) > 0.0 > (coasting_create - coasting_decay)


def test_create_rate_is_signed_and_horizon_validated(model) -> None:
    assert create_rate(model, _IMPROVING) >= 0.0
    with pytest.raises(ValueError, match="horizon"):
        create_rate(model, _IMPROVING, horizon=0)
    with pytest.raises(ValueError, match="horizon"):
        decay_rate(model, _IMPROVING, horizon=-1)


def test_bootstrap_reports_rates_and_finite_t_rsi(model, seeded_rng) -> None:
    result = bootstrap_t_rsi(model, _IMPROVING, seeded_rng, n=200)
    assert result["create_se"] >= 0.0 and result["decay_se"] >= 0.0
    assert result["create_mean"] >= 0.0 and result["decay_mean"] >= 0.0
    assert np.isfinite(result["t_rsi"])
    with pytest.raises(ValueError, match="concentration"):
        bootstrap_t_rsi(model, _IMPROVING, seeded_rng, n=10, concentration=0.0)


def test_t_rsi_standardization_is_monotone_in_separation() -> None:
    narrow = t_rsi([1.0, 1.1, 1.2], [0.9, 1.0, 1.1])
    wide = t_rsi([1.4, 1.5, 1.6], [0.8, 0.9, 1.0])
    assert wide > narrow


def test_certificate_fires_only_at_or_above_delta() -> None:
    assert certificate(2.0, 2.0) is True
    assert certificate(2.5, 2.0) is True
    assert certificate(1.9, 2.0) is False


def test_bootstrap_is_byte_identical_for_same_seed() -> None:
    model = default_model()
    prior = belief_prior(model)
    first = bootstrap_t_rsi(model, prior, np.random.default_rng(7), n=200)
    second = bootstrap_t_rsi(model, prior, np.random.default_rng(7), n=200)
    assert first == second


def test_t_rsi_edge_cases_and_validation(model, prior) -> None:
    assert t_rsi([1.0], [0.5]) == 0.0
    assert np.isinf(t_rsi([2.0, 2.0], [1.0, 1.0]))
    assert t_rsi([1.0, 1.0], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="one-dimensional"):
        t_rsi([[1.0, 2.0]], [0.5, 0.6])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        t_rsi([1.0, np.nan], [0.5, 0.6])

    with pytest.raises(ValueError, match="rng"):
        bootstrap_t_rsi(model, prior, rng="seeded", n=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_t_rsi(model, prior, rng=np.random.default_rng(3), n=0)
    with pytest.raises(ValueError, match="NaN"):
        certificate(float("nan"), 1.0)
