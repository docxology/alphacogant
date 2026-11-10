"""Tests for the statistics module: bootstrap CIs, regime comparison, effect sizes.

No mocks; all tests use real model computation and fixed seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.model.operating_points import COASTING, IMPROVING
from alphacogant.stats.simulation import simulate_trajectory, summarize_trajectory
from alphacogant.stats.statistics import (
    BootstrapCI,
    BreakEvenProfile,
    RegimeComparison,
    RegimeStatistics,
    bootstrap_ci,
    break_even_profile,
    cohen_d,
    compare_regimes,
    compute_regime_statistics,
)
from alphacogant.trsi.t_rsi import paired_bootstrap_samples


def test_exploration_ratio_is_behavioral_not_a_structural_constant(model):
    """The regime exploration ratio is the greedy-trajectory epistemic-cycle fraction.

    Binds it to the independent simulation module (not the constant epistemic/total
    actions = 2/6), so it is a genuine per-regime statistic.
    """
    for belief in (IMPROVING, COASTING):
        stats = compute_regime_statistics(model, belief, n=16)
        traj = simulate_trajectory(model, belief, horizon=12, policy="greedy")
        expected = float(summarize_trajectory(traj)["exploration_ratio"])
        assert stats.exploration_ratio == pytest.approx(expected)

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


def test_compute_regime_statistics_improving(model):
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


def test_compute_regime_statistics_coasting(model):
    stats = compute_regime_statistics(model, COASTING, "Coasting", n=32)
    assert stats.name == "Coasting"
    assert np.isfinite(stats.t_rsi)


def test_compute_regime_statistics_deterministic(model):
    s1 = compute_regime_statistics(model, IMPROVING, n=32)
    s2 = compute_regime_statistics(model, IMPROVING, n=32)
    assert s1.t_rsi == s2.t_rsi
    assert s1.create_ci.mean == s2.create_ci.mean


def test_compute_regime_statistics_default_model():
    stats = compute_regime_statistics(n=32)
    assert stats.name == "operating_point"
    assert np.isfinite(stats.t_rsi)


def test_compute_regime_statistics_as_dict(model):
    stats = compute_regime_statistics(model, IMPROVING, "test", n=32)
    d = stats.as_dict()
    assert d["name"] == "test"
    assert "t_rsi" in d
    assert "create_mean" in d
    assert "create_ci" in d
    assert len(d["create_ci"]) == 2


def test_compute_regime_statistics_efe_decomposition(model):
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
    for field, actual in (
        ("create_ci", comparison.cohen_d_create),
        ("decay_ci", comparison.cohen_d_decay),
    ):
        a = getattr(comparison.improving, field)
        b = getattr(comparison.coasting, field)
        expected = (a.mean - b.mean) / np.sqrt(
            ((a.n - 1) * a.std**2 + (b.n - 1) * b.std**2) / (a.n + b.n - 2)
        )
        assert actual == pytest.approx(expected)


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


def test_break_even_profile_basic(model):
    profile = break_even_profile(model, IMPROVING, n=64)
    assert isinstance(profile, BreakEvenProfile)
    assert 0.0 <= profile.probability <= 1.0
    assert profile.n == 64
    assert profile.margin_ci_lower <= profile.margin_mean <= profile.margin_ci_upper
    assert np.isfinite(profile.margin_std)


def test_break_even_profile_is_paired_and_deterministic(model):
    p1 = break_even_profile(model, IMPROVING, n=64)
    p2 = break_even_profile(model, IMPROVING, n=64)
    assert p1 == p2


def test_break_even_profile_as_dict():
    data = break_even_profile(n=32).as_dict()
    assert data["n"] == 32
    assert "probability" in data
    assert "margin_mean" in data


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200, -1e200])
def test_cohen_d_unequal_groups_is_scale_invariant(scale):
    a = np.array([0.0, 2.0]) * scale
    b = np.array([2.0, 2.0, 2.0, 2.0]) * scale
    expected = -np.copysign(np.sqrt(2.0), scale)
    assert cohen_d(a, b) == pytest.approx(expected)
    assert cohen_d(b, a) == pytest.approx(-expected)


def test_cohen_d_extreme_groups_need_no_unscaled_variance_or_mean_difference():
    maximum = np.finfo(float).max
    assert cohen_d(
        np.array([-maximum, maximum]), np.full(4, maximum)
    ) == pytest.approx(-np.sqrt(2.0))
    assert cohen_d(
        np.array([maximum / 2, maximum]), np.array([-maximum / 2, -maximum])
    ) == pytest.approx(3 * np.sqrt(2.0))


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([2.0, 2.0], [2.0, 2.0, 2.0], 0.0),
        ([0.0, 0.0], [1.0, 1.0], -np.inf),
        ([1.0, 1.0], [0.0, 0.0], np.inf),
        ([1e308, 1e308], [-1e308, -1e308], np.inf),
        ([1e-200, 1e-200], [2e-200, 2e-200], -np.inf),
    ],
)
def test_cohen_d_constant_groups(a, b, expected):
    assert cohen_d(np.array(a), np.array(b)) == expected


@pytest.mark.parametrize(
    "invalid",
    [[], [1.0], [[1.0, 2.0]], 1.0, [np.nan, 1.0], [np.inf, 1.0], [-np.inf, 1.0]],
)
@pytest.mark.parametrize("invalid_first", [True, False])
def test_cohen_d_rejects_malformed_groups(invalid, invalid_first):
    valid = np.array([0.0, 1.0])
    a, b = (invalid, valid) if invalid_first else (valid, invalid)
    with pytest.raises(ValueError, match="finite one-dimensional group of at least 2"):
        cohen_d(a, b)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_bootstrap_ci_stable_moments_and_quantiles(scale):
    ci = bootstrap_ci(np.array([1.0, 2.0, 3.0]) * scale, confidence=0.8)
    assert ci.mean / scale == pytest.approx(2.0)
    assert ci.std / scale == pytest.approx(1.0)
    assert ci.se / scale == pytest.approx(1 / np.sqrt(3.0))
    assert ci.ci_lower / scale == pytest.approx(1.2)
    assert ci.ci_upper / scale == pytest.approx(2.8)


def test_bootstrap_ci_extreme_opposite_endpoints():
    ci = bootstrap_ci(np.array([-1e308, 1e308]), confidence=0.8)
    assert ci.mean == 0.0
    assert ci.std / 1e308 == pytest.approx(np.sqrt(2.0))
    assert ci.se / 1e308 == pytest.approx(1.0)
    assert ci.ci_lower / 1e308 == pytest.approx(-0.8)
    assert ci.ci_upper / 1e308 == pytest.approx(0.8)


@pytest.mark.parametrize(
    "confidence",
    [float(np.nextafter(0.0, 1.0)), 1e-18, float(np.nextafter(1.0, 0.0))],
)
def test_central_interval_preserves_admitted_confidence(confidence):
    magnitude = 1e308
    samples = np.array([-magnitude, magnitude])
    # The central linear interval of two symmetric endpoints is exactly ±a*c.
    expected = magnitude * confidence
    interval = bootstrap_ci(samples, confidence=confidence)
    assert (interval.ci_lower, interval.ci_upper) == (-expected, expected)
    np.testing.assert_array_equal(samples, [-magnitude, magnitude])


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([100.0, -8.0, 1.0, 7.0, 4.0], (1.0, 7.0)),
        ([-3.0, 0.0, 10.0], (-1.5, 5.0)),
    ],
)
def test_central_interval_exact_and_fractional_order_statistic_ranks(samples, expected):
    interval = bootstrap_ci(np.array(samples), confidence=0.5)
    assert (interval.ci_lower, interval.ci_upper) == expected


@pytest.mark.parametrize("offset", [float(3 * 2**52), -float(3 * 2**52)])
def test_moments_and_effect_size_preserve_large_common_offset(offset):
    a = np.array([offset, offset + 4])
    b = np.array([offset + 4, offset + 8])
    ci = bootstrap_ci(a)
    assert ci.mean == offset + 2
    assert ci.se == pytest.approx(2.0)
    assert cohen_d(a, b) == pytest.approx(-np.sqrt(2.0))
    first = _sample_regime("a", a, a, 0.95, 0.95)
    second = _sample_regime("b", b, b, 0.95, 0.95)
    assert RegimeComparison(first, second).cohen_d_create == pytest.approx(-np.sqrt(2.0))


def test_standard_error_can_be_finite_when_sample_sd_exceeds_float_range():
    maximum = np.finfo(float).max
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ci = bootstrap_ci(np.array([-maximum, maximum]))
    assert ci.mean == 0.0
    assert np.isinf(ci.std)
    assert ci.se == pytest.approx(maximum)


def _sample_regime(name, create_samples, decay_samples, create_confidence, decay_confidence):
    return RegimeStatistics(
        name=name,
        belief={},
        t_rsi=0.0,
        create_ci=bootstrap_ci(np.asarray(create_samples), create_confidence),
        decay_ci=bootstrap_ci(np.asarray(decay_samples), decay_confidence),
        efe_pragmatic_per_action={},
        efe_epistemic_per_action={},
        funded_channel="hold",
        funded_pragmatic=0.0,
        funded_epistemic=0.0,
        exploration_ratio=0.0,
    )


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_regime_comparison_weights_stored_sample_counts(scale):
    a = _sample_regime(
        "Improving", np.array([0.0, 2.0]) * scale, np.array([4.0, 4.0]) * scale, 0.8, 0.9
    )
    b = _sample_regime(
        "Coasting", np.full(4, 2.0) * scale, np.array([0.0, 2.0, 2.0, 4.0]) * scale, 0.9, 0.8
    )
    comparison = RegimeComparison(a, b)
    assert comparison.cohen_d_create == pytest.approx(-np.sqrt(2.0))
    # Decay: means 4 and 2; sample variances 0 and 8/3; pooled variance 2.
    assert comparison.cohen_d_decay == pytest.approx(np.sqrt(2.0))


def test_regime_comparison_constant_summary_policy():
    a = _sample_regime("Improving", [0.0, 0.0], [2.0, 2.0], 0.8, 0.8)
    b = _sample_regime("Coasting", [1.0] * 4, [2.0] * 4, 0.8, 0.8)
    comparison = RegimeComparison(a, b)
    assert comparison.cohen_d_create == -np.inf
    assert comparison.cohen_d_decay == 0.0


@pytest.mark.parametrize(
    ("levels", "create_cells", "decay_cells"),
    [
        ((0.8, 0.8, 0.8, 0.8), ("80%", "80%"), ("80%", "80%")),
        ((0.8, 0.9, 0.95, 0.99), ("80%", "95%"), ("90%", "99%")),
    ],
)
def test_regime_table_and_summary_preserve_each_confidence(levels, create_cells, decay_cells):
    a = _sample_regime("Improving", [0.0, 2.0], [1.0, 3.0], *levels[:2])
    b = _sample_regime("Coasting", [2.0] * 4, [1.0, 2.0, 3.0, 4.0], *levels[2:])
    rows = [
        [cell.strip() for cell in row.strip("|").split("|")]
        for row in RegimeComparison(a, b).as_table().splitlines()
    ]
    create_row = next(row for row in rows if row[0] == "Create-rate CI")
    decay_row = next(row for row in rows if row[0] == "Decay-rate CI")
    for row, labels, intervals in (
        (create_row, create_cells, (a.create_ci, b.create_ci)),
        (decay_row, decay_cells, (a.decay_ci, b.decay_ci)),
    ):
        for cell, label, ci in zip(row[1:3], labels, intervals, strict=True):
            assert cell == f"{label} CI [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"
    for regime in (a, b):
        summary = regime.as_dict()
        assert summary["create_ci_confidence"] == regime.create_ci.confidence
        assert summary["decay_ci_confidence"] == regime.decay_ci.confidence
        assert summary["create_n"] == regime.create_ci.n
        assert summary["decay_n"] == regime.decay_ci.n


def test_regime_and_break_even_summaries_share_paired_draws(model):
    seed, n, horizon, concentration, confidence = 37, 12, 2, 8.0, 0.8
    create, decay = paired_bootstrap_samples(
        model, IMPROVING, np.random.default_rng(seed), n, horizon, concentration
    )
    margins = create - decay
    stats = compute_regime_statistics(
        model, IMPROVING, seed=seed, n=n, horizon=horizon,
        concentration=concentration, confidence=confidence,
    )
    profile = break_even_profile(
        model, IMPROVING, seed=seed, n=n, horizon=horizon,
        concentration=concentration, confidence=confidence,
    )
    assert stats.t_rsi == pytest.approx(
        margins.mean() / (margins.std(ddof=1) / np.sqrt(n))
    )
    for samples, ci in ((create, stats.create_ci), (decay, stats.decay_ci)):
        assert ci.mean == pytest.approx(samples.mean())
        assert ci.std == pytest.approx(samples.std(ddof=1))
        assert ci.confidence == confidence
        assert [ci.ci_lower, ci.ci_upper] == pytest.approx(np.percentile(samples, [10, 90]))
    assert profile.probability == np.count_nonzero(margins > 0.0) / n
    assert profile.margin_mean == pytest.approx(margins.mean())
    assert profile.margin_std == pytest.approx(margins.std(ddof=1))
    assert [profile.margin_ci_lower, profile.margin_ci_upper] == pytest.approx(
        np.percentile(margins, [10, 90])
    )
    assert profile.create_mean == pytest.approx(stats.create_ci.mean)
    assert profile.decay_mean == pytest.approx(stats.decay_ci.mean)
    assert profile.as_dict()["confidence"] == confidence


def test_regime_comparison_extreme_summary_moments():
    a = _sample_regime(
        "Improving", [-1e308, 1e308], [5e307, 1e308], 0.8, 0.8
    )
    b = _sample_regime(
        "Coasting", [1e308] * 4, [-5e307, -1e308], 0.8, 0.8
    )
    comparison = RegimeComparison(a, b)
    assert comparison.cohen_d_create == pytest.approx(-np.sqrt(2.0))
    assert comparison.cohen_d_decay == pytest.approx(3 * np.sqrt(2.0))


def test_opposite_large_offsets_do_not_erase_within_group_dispersion():
    offset = float(3 * 2**52)
    samples = np.array([offset, offset + 4])
    assert cohen_d(samples, -samples) == pytest.approx(
        (2 * offset + 4) / np.sqrt(8.0), rel=1e-15
    )


def test_regime_effect_retains_finite_se_when_raw_sd_overflows():
    maximum = np.finfo(float).max
    first = _sample_regime("a", [-maximum, maximum], [-maximum, maximum], 0.95, 0.95)
    second = _sample_regime("b", [maximum] * 4, [maximum] * 4, 0.95, 0.95)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        comparison = RegimeComparison(first, second)
        assert comparison.cohen_d_create == pytest.approx(-np.sqrt(2.0))


def test_mean_and_effect_size_preserve_cancellation_residual():
    magnitude = float(2**54)
    samples = np.array([magnitude, -magnitude, -1.0])
    assert bootstrap_ci(samples).mean == pytest.approx(-1 / 3)
    assert cohen_d(samples, np.zeros(3)) == pytest.approx(
        -np.sqrt(2) / (3 * magnitude), rel=1e-15, abs=0
    )


def test_unrelated_outlier_does_not_erase_positive_interval_endpoints():
    small = 1e-100
    ci = bootstrap_ci(np.array([small] * 80 + [1e308]))
    assert ci.ci_lower == small
    assert ci.ci_upper == small


def test_finite_mean_survives_intermediate_sum_overflow():
    maximum = np.finfo(float).max
    ci = bootstrap_ci(np.array([maximum, maximum, -maximum]))
    assert ci.mean / maximum == pytest.approx(1 / 3)
