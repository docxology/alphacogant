"""Paired t-RSI, deterministic bootstrap and scalar threshold behavior.

The comparator can change sign; the predicate does not enforce action,
model-update or deployment admission.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import default_model
from alphacogant.model.operating_points import COASTING as _COASTING
from alphacogant.model.operating_points import IMPROVING as _IMPROVING
from alphacogant.stats.simulation import simulate_trajectory
from alphacogant.trsi.t_rsi import (
    bootstrap_t_rsi,
    certificate,
    create_rate,
    decay_rate,
    paired_bootstrap_samples,
    t_rsi,
)


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


def test_t_rsi_standardization_is_pairing_aware() -> None:
    """Same marginal means, tighter paired margins -> larger |t-RSI|.

    create and decay samples are paired (both rates read off the same perturbed
    trajectory), so the denominator is the SE of the elementwise differences:
    correlation between the two rate streams changes the statistic even when the
    marginal distributions are identical. This is exactly the property the pooled
    independence-assuming SE ignores.
    """
    create = [1.0, 1.2]
    noisy_margin_decay = [1.05, 0.95]  # margins [-0.05, +0.25]
    tight_margin_decay = [0.95, 1.05]  # margins [+0.05, +0.15]
    noisy = t_rsi(create, noisy_margin_decay)
    tight = t_rsi(create, tight_margin_decay)
    assert tight > noisy > 0.0


def test_t_rsi_rejects_unpaired_sample_sets() -> None:
    """Non-empty arrays of unequal length cannot be paired."""
    with pytest.raises(ValueError, match="paired"):
        t_rsi([1.0, 1.1, 1.2], [0.9, 1.0])


def test_t_rsi_sign_carries_into_degenerate_zero_variance() -> None:
    """A zero-variance paired margin yields a signed infinity, not +inf always."""
    assert t_rsi([2.0, 2.0], [1.0, 1.0]) == float("inf")
    assert t_rsi([1.0, 1.0], [2.0, 2.0]) == float("-inf")


def test_certificate_fires_only_at_or_above_delta() -> None:
    assert certificate(2.0, 2.0) is True
    assert certificate(2.5, 2.0) is True
    assert certificate(1.9, 2.0) is False


def test_bootstrap_is_byte_identical_for_same_seed(model, prior) -> None:
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


def test_paired_bootstrap_samples_are_byte_identical_and_aligned(model, prior) -> None:
    """paired_bootstrap_samples is deterministic and yields aligned rate arrays."""
    first_create, first_decay = paired_bootstrap_samples(
        model, prior, np.random.default_rng(7), n=24, horizon=4
    )
    second_create, second_decay = paired_bootstrap_samples(
        model, prior, np.random.default_rng(7), n=24, horizon=4
    )
    assert np.array_equal(first_create, second_create)
    assert np.array_equal(first_decay, second_decay)
    assert first_create.shape == first_decay.shape == (24,)
    # Decay samples are residual erosions: never negative.
    assert np.all(first_decay >= 0.0)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_standardization_is_scale_invariant_without_false_certificate(scale):
    statistic = t_rsi([scale, 2 * scale], [0.0, 0.0])
    assert statistic == pytest.approx(3.0)
    assert certificate(statistic, 4.0) is False


def test_standardization_handles_overflowing_unscaled_margins():
    assert t_rsi([1e308, 0.0], [-1e308, 0.0]) == pytest.approx(1.0)
    # Large equal pairs must not erase the nonzero tiny paired margins.
    assert t_rsi([1e308, 1e-200, 2e-200], [1e308, 0.0, 0.0]) == pytest.approx(
        np.sqrt(3.0)
    )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_close_margins_have_finite_signed_statistic(sign):
    magnitude = float(2**1023)
    below = np.nextafter(magnitude, 0.0)
    # For two margins, t = (m1 + m2) / |m1 - m2|. Here their
    # difference is 2**970, giving the exact signed value 2**55 - 1.
    exact = int(sign) * Fraction(2**1025 - 2**970, 2**970)
    with np.errstate(all="raise"):
        statistic = t_rsi([sign * magnitude] * 2, [-sign * magnitude, -sign * below])
    assert statistic == float(exact)
    assert certificate(statistic, 2**56) is False
    assert certificate(statistic, -2**56) is True


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_margins_preserve_small_cancellation_residual(sign):
    magnitude = float(2**1023)
    # Margins [2M, -2M, sign] give t = sign / sqrt(12M² + 1).
    # The +1 cannot affect the representable result; forming M² would overflow.
    expected = (sign / magnitude) / np.sqrt(12.0)
    with np.errstate(all="raise"):
        statistic = t_rsi([magnitude, -magnitude, sign], [-magnitude, magnitude, 0.0])
    assert statistic == pytest.approx(expected, rel=1e-14, abs=0.0)
    assert certificate(statistic, 0.0) is (sign > 0.0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_equal_margins_preserve_genuine_signed_infinity(sign):
    magnitude = sign * float(2**1023)
    assert t_rsi([magnitude, magnitude], [-magnitude, -magnitude]) == sign * float("inf")
    assert t_rsi([magnitude, -magnitude], [-magnitude, magnitude]) == 0.0
    assert t_rsi([magnitude], [-magnitude]) == 0.0


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_unrepresentable_nonzero_overflow_cancellation_is_rejected(sign):
    magnitude = float(2**1023)
    residual = sign * np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="too small to represent"):
        t_rsi([magnitude, -magnitude, residual], [-magnitude, magnitude, 0.0])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_inexact_finite_subtraction_cannot_manufacture_constant_margins(sign):
    create = [sign, sign]
    decay = [sign * 2.0**-54, sign * 2.0**-55]
    # Independent two-pair identity: t = (m1 + m2) / abs(m1 - m2).
    exact = [
        Fraction.from_float(c) - Fraction.from_float(d)
        for c, d in zip(create, decay, strict=True)
    ]
    expected = float(sum(exact) / abs(exact[0] - exact[1]))
    statistic = t_rsi(create, decay)
    assert statistic == expected
    assert certificate(statistic, 2**57) is False


def test_inexact_subtraction_cannot_erase_negative_paired_mean():
    statistic = t_rsi([1.0, -1.0], [2.0**-54, 0.0])
    expected = float(-Fraction(1, 2**55 - 1))
    assert statistic == expected
    assert certificate(statistic, 0.0) is False


def test_mean_scaling_retains_representable_cancellation_after_many_pairs():
    from decimal import Decimal, localcontext

    magnitude = float(2**1023)
    create = np.zeros(1024)
    create[:3] = [magnitude, -magnitude, 1e-13]
    # Exact independent t² = (n-1)r² / (2nM² + (n-1)r²).
    with localcontext() as context:
        context.prec = 80
        m, r = Decimal.from_float(magnitude), Decimal.from_float(1e-13)
        squared = 1023 * r * r / (2048 * m * m + 1023 * r * r)
        expected = float(squared.sqrt())
    statistic = t_rsi(create, np.zeros(1024))
    assert statistic == pytest.approx(expected, rel=0.0, abs=np.nextafter(0.0, 1.0))
    assert statistic > 0.0


def test_nonconstant_unrepresentable_paired_statistic_is_not_degenerate_infinity():
    magnitude = float(2**1023)
    tiny = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="finite float range"):
        t_rsi([magnitude, magnitude], [-tiny, -2 * tiny])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_subnormal_mean_is_not_rounded_before_dimensionless_standardization(sign):
    smallest = np.nextafter(0.0, 1.0)
    # Exact two-pair t = (s + 2s) / abs(s - 2s) = 3, independent of units.
    statistic = t_rsi([sign * smallest, sign * 2 * smallest], [0.0, 0.0])
    assert statistic == sign * 3.0
    assert certificate(statistic, 3.5) is False
    assert certificate(statistic, -3.5) is True


@pytest.mark.parametrize("offset", [float(3 * 2**52), -float(3 * 2**52)])
def test_standardization_preserves_close_large_offset_samples(offset):
    statistic = t_rsi([offset, offset + 4], [0.0, 0.0])
    assert statistic == pytest.approx((offset + 2) / 2, rel=1e-15)
    assert certificate(statistic, 6e15) is (offset > 0)


@pytest.mark.parametrize("horizon", [1, 2, 4])
def test_create_rate_includes_every_action_outcome(horizon):
    reward = np.zeros((3, 2, 2, 2))
    reward[0, 0, :, :] = 1.0
    reward[2, 1, :, :] = 1.0
    loss = np.zeros((3, 2, 2))
    loss[1] = 1.0
    transitions = {
        channel: np.repeat(np.eye(2)[:, :, None], 6, axis=2) for channel in CHANNELS
    }
    transitions["I"][:, :, 0] = [[0.0, 0.0], [1.0, 1.0]]
    model = default_model(
        A_R=reward, A_L=loss, B=transitions, C_R=[0, 0, 1], C_L=[0, 0, 0]
    )
    belief = {channel: np.array([0.5, 0.5]) for channel in CHANNELS}
    belief["I"] = np.array([1.0, 0.0])
    active = simulate_trajectory(model, belief, horizon, policy="greedy")
    passive = simulate_trajectory(model, belief, horizon, policy="hold")
    advantage = (active.cumulative_pragmatic[-1] - passive.cumulative_pragmatic[-1]) / horizon
    assert advantage == pytest.approx(1.0)
    assert create_rate(model, belief, horizon) == pytest.approx(advantage)


def test_subnormal_samples_are_standardized_before_rescaling():
    smallest = np.nextafter(0.0, 1.0)
    assert t_rsi([0.0, smallest], [0.0, 0.0]) == pytest.approx(1.0)
    assert certificate(t_rsi([0.0, smallest], [0.0, 0.0]), 0.5)


def test_single_draw_bootstrap_preserves_zero_statistic_convention(model, prior):
    result = bootstrap_t_rsi(model, prior, np.random.default_rng(7), n=1, horizon=2)
    assert result["t_rsi"] == 0.0
    assert result["create_se"] == result["decay_se"] == 0.0


def test_cancellation_residual_cannot_become_a_passing_zero():
    magnitude = float(2**54)
    statistic = t_rsi([magnitude, -magnitude, -1.0], [0.0, 0.0, 0.0])
    assert statistic == pytest.approx(-1 / (magnitude * np.sqrt(3)), rel=1e-15, abs=0)
    assert not certificate(statistic, 0.0)


def test_identical_policies_have_zero_create_rate_at_large_finite_preferences():
    model = default_model()
    model.C_R[:] = 1e308
    model.C_L[:] = 0.0
    for likelihood in (model.A_R, model.A_L):
        likelihood.fill(0.0)
        likelihood[0, ...] = 1.0
    for transition in model.B.values():
        transition[:] = np.eye(2)[:, :, None]
    belief = {channel: np.array([1.0, 0.0]) for channel in CHANNELS}
    with np.errstate(over="raise", invalid="raise"):
        assert create_rate(model, belief, horizon=4) == 0.0
        assert decay_rate(model, belief, horizon=4) == 0.0
