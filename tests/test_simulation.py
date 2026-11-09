"""Tests for the simulation module: multi-cycle trajectory recording.

No mocks; all tests use real model computation and fixed seeds.
"""

from __future__ import annotations

from dataclasses import replace
from math import fsum, isfinite

import numpy as np
import pytest

from alphacogant.model.channels import ACTIONS, CHANNELS, action_index
from alphacogant.model.generative_model import belief_prior, default_model
from alphacogant.model.operating_points import IMPROVING
from alphacogant.stats.simulation import (
    CycleRecord,
    TrajectoryResult,
    simulate_trajectory,
    summarize_trajectory,
)


def test_trajectory_has_correct_length(model):
    traj = simulate_trajectory(model, IMPROVING, horizon=12, policy="greedy")
    assert len(traj.cycles) == 12
    assert traj.horizon == 12


def test_each_cycle_records_full_state(model):
    traj = simulate_trajectory(model, IMPROVING, horizon=3, policy="greedy")
    for i, record in enumerate(traj.cycles):
        assert record.cycle == i
        assert isinstance(record, CycleRecord)
        assert record.action_name in ACTIONS
        assert 0 <= record.action < len(ACTIONS)
        # Belief is a valid distribution for each channel
        for channel in CHANNELS:
            assert abs(sum(record.belief[channel]) - 1.0) < 1e-8
            assert 0.0 <= record.p_strong[channel] <= 1.0
        # EFE decomposition
        assert record.efe_total == -(record.efe_pragmatic + record.efe_epistemic)
        assert record.efe_epistemic >= -1e-12
        # Marginal returns cover all actions
        assert len(record.marginal_returns) == len(ACTIONS)


def test_greedy_trajectory_deterministic(model):
    t1 = simulate_trajectory(model, IMPROVING, horizon=8, policy="greedy")
    t2 = simulate_trajectory(model, IMPROVING, horizon=8, policy="greedy")
    assert t1.action_history == t2.action_history
    for c1, c2 in zip(t1.cycles, t2.cycles, strict=True):
        assert c1.action == c2.action
        for channel in CHANNELS:
            assert np.allclose(c1.belief[channel], c2.belief[channel])


def test_hold_policy_never_funds(model):
    traj = simulate_trajectory(model, IMPROVING, horizon=6, policy="hold")
    assert all(a == 5 for a in traj.action_history)  # hold = action index 5
    counts = traj.action_counts()
    assert counts["hold"] == 6
    assert all(counts[a] == 0 for a in ACTIONS if a != "hold")


def test_fund_theta_policy_always_funds_theta(model):
    traj = simulate_trajectory(model, IMPROVING, horizon=6, policy="fund_theta")
    assert all(a == 3 for a in traj.action_history)  # fund_Theta = 3
    counts = traj.action_counts()
    assert counts["fund_Theta"] == 6


def test_stochastic_policy_requires_seed(model):
    with pytest.raises(ValueError, match="seed is required"):
        simulate_trajectory(model, IMPROVING, horizon=3, policy="stochastic")


def test_stochastic_policy_deterministic_with_same_seed(model):
    t1 = simulate_trajectory(model, IMPROVING, horizon=6, policy="stochastic", seed=42)
    t2 = simulate_trajectory(model, IMPROVING, horizon=6, policy="stochastic", seed=42)
    assert t1.action_history == t2.action_history


def test_stochastic_policy_different_with_different_seeds(model):
    t1 = simulate_trajectory(model, IMPROVING, horizon=12, policy="stochastic", seed=42)
    t2 = simulate_trajectory(model, IMPROVING, horizon=12, policy="stochastic", seed=99)
    # They should differ (extremely unlikely to be identical)
    assert t1.action_history != t2.action_history


def test_belief_history_tracks_p_strong(model):
    traj = simulate_trajectory(model, IMPROVING, horizon=5, policy="greedy")
    for channel in CHANNELS:
        assert len(traj.belief_history[channel]) == 5
        for val in traj.belief_history[channel]:
            assert 0.0 <= val <= 1.0


def test_cumulative_pragmatic_matches_signed_prefix_totals(model):
    # Rewards depend only on Theta. Funding yields q' = .25 + .6q,
    # so the expected preference 2q' - 1 crosses from negative to positive.
    reward = np.zeros((3, 2, 2, 2))
    reward[0, :, :, 0] = 1.0
    reward[2, :, :, 1] = 1.0
    transitions = {channel: matrix.copy() for channel, matrix in model.B.items()}
    transitions["Theta"][:, :, action_index("fund_Theta")] = [
        [0.75, 0.15], [0.25, 0.85]
    ]
    signed_model = default_model(
        A_R=reward, B=transitions, C_R=[-1.0, 0.0, 1.0], C_L=np.zeros(3)
    )
    traj = simulate_trajectory(signed_model, IMPROVING, horizon=8, policy="fund_theta")
    q = float(IMPROVING["Theta"][1])
    expected_scores = []
    for _ in range(8):
        q = 0.25 + 0.6 * q
        expected_scores.append(2 * q - 1)
    scores = [record.efe_pragmatic for record in traj.cycles]
    assert min(scores) < 0 < max(scores)
    assert scores == pytest.approx(expected_scores, abs=1e-14)
    expected_prefixes = [fsum(scores[:end]) for end in range(1, 9)]
    assert traj.cumulative_pragmatic == pytest.approx(expected_prefixes, abs=1e-14)
    assert summarize_trajectory(traj)["total_pragmatic"] == f"{fsum(scores):.4f}"


@pytest.mark.parametrize("sign", [-1, 1])
def test_large_finite_scores_preserve_representable_aggregate(sign):
    value = sign * 1e307
    model = default_model(C_R=np.full(3, value), C_L=np.zeros(3))
    traj = simulate_trajectory(model, belief_prior(model), horizon=17, policy="hold")
    scores = [record.efe_pragmatic for record in traj.cycles]
    assert scores == pytest.approx([value] * 17)
    expected = [fsum(scores[:end]) for end in range(1, 18)]
    assert all(isfinite(total) for total in traj.cumulative_pragmatic)
    assert traj.cumulative_pragmatic == pytest.approx(expected, rel=1e-14)
    summary = summarize_trajectory(traj)
    assert isfinite(float(summary["total_pragmatic"]))
    assert float(summary["total_pragmatic"]) == pytest.approx(expected[-1], rel=1e-14)
    assert float(summary["total_epistemic"]) == pytest.approx(
        fsum(record.efe_epistemic for record in traj.cycles), abs=5e-5
    )


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("horizon", [18, 32])
def test_trajectory_rejects_first_unrepresentable_prefix(sign, horizon):
    model = default_model(C_R=np.full(3, sign * 1e307), C_L=np.zeros(3))
    with pytest.raises(ValueError, match="cumulative pragmatic.*finite float at cycle 18"):
        simulate_trajectory(model, belief_prior(model), horizon=horizon, policy="hold")


def test_summary_rejects_unrepresentable_total_from_finite_records():
    model = default_model(C_R=np.full(3, 1e307), C_L=np.zeros(3))
    traj = simulate_trajectory(model, belief_prior(model), horizon=17, policy="hold")
    # The public record list may be assembled outside the simulator. Reuse
    # actual finite cycle scores, not nonfinite or fabricated EFE values.
    overflowing = replace(
        traj, cycles=[*traj.cycles, replace(traj.cycles[-1], cycle=17)]
    )
    with pytest.raises(ValueError, match="cumulative pragmatic.*finite float at cycle 18"):
        summarize_trajectory(overflowing)


def test_action_counts_sum_to_horizon(model):
    traj = simulate_trajectory(model, IMPROVING, horizon=10, policy="greedy")
    counts = traj.action_counts()
    assert sum(counts.values()) == 10


def test_greedy_from_improving_funds_theta(model):
    """The greedy policy from IMPROVING should fund Theta (it starts very stale)."""
    traj = simulate_trajectory(model, IMPROVING, horizon=3, policy="greedy")
    assert all(a == 3 for a in traj.action_history[:3])


def test_theta_freshness_increases_under_greedy(model):
    """When Theta starts stale and greedy funds it, P(fresh) should increase."""
    traj = simulate_trajectory(model, IMPROVING, horizon=6, policy="greedy")
    first_p = traj.cycles[0].p_strong["Theta"]
    last_p = traj.cycles[-1].p_strong["Theta"]
    assert last_p > first_p


def test_summarize_trajectory(model):
    traj = simulate_trajectory(model, IMPROVING, horizon=6, policy="greedy")
    summary = summarize_trajectory(traj)
    assert "horizon" in summary
    assert summary["horizon"] == "6"
    assert "first_funded" in summary
    assert "last_funded" in summary
    assert "exploration_ratio" in summary
    assert "dominant_action" in summary
    assert summary["dominant_action"] in ACTIONS


def test_summarize_empty_trajectory_raises():
    empty = TrajectoryResult(
        cycles=[],
        belief_history={c: [] for c in CHANNELS},
        action_history=[],
        cumulative_pragmatic=[],
    )
    with pytest.raises(ValueError, match="empty"):
        summarize_trajectory(empty)


def test_invalid_policy_raises(model):
    with pytest.raises(ValueError, match="Unknown policy"):
        simulate_trajectory(model, IMPROVING, horizon=3, policy="invalid")


def test_zero_horizon_raises(model):
    with pytest.raises(ValueError, match="horizon"):
        simulate_trajectory(model, IMPROVING, horizon=0)


def test_stochastic_actions_valid(model):
    traj = simulate_trajectory(model, IMPROVING, horizon=10, policy="stochastic", seed=123)
    for action in traj.action_history:
        assert 0 <= action < len(ACTIONS)
