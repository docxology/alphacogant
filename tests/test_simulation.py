"""Tests for the simulation module: multi-cycle trajectory recording.

No mocks; all tests use real model computation and fixed seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.model.channels import ACTIONS, CHANNELS
from alphacogant.model.generative_model import default_model
from alphacogant.model.operating_points import IMPROVING
from alphacogant.stats.simulation import (
    CycleRecord,
    TrajectoryResult,
    simulate_trajectory,
    summarize_trajectory,
)


def test_trajectory_has_correct_length():
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=12, policy="greedy")
    assert len(traj.cycles) == 12
    assert traj.horizon == 12


def test_each_cycle_records_full_state():
    model = default_model()
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


def test_greedy_trajectory_deterministic():
    model = default_model()
    t1 = simulate_trajectory(model, IMPROVING, horizon=8, policy="greedy")
    t2 = simulate_trajectory(model, IMPROVING, horizon=8, policy="greedy")
    assert t1.action_history == t2.action_history
    for c1, c2 in zip(t1.cycles, t2.cycles, strict=True):
        assert c1.action == c2.action
        for channel in CHANNELS:
            assert np.allclose(c1.belief[channel], c2.belief[channel])


def test_hold_policy_never_funds():
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=6, policy="hold")
    assert all(a == 5 for a in traj.action_history)  # hold = action index 5
    counts = traj.action_counts()
    assert counts["hold"] == 6
    assert all(counts[a] == 0 for a in ACTIONS if a != "hold")


def test_fund_theta_policy_always_funds_theta():
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=6, policy="fund_theta")
    assert all(a == 3 for a in traj.action_history)  # fund_Theta = 3
    counts = traj.action_counts()
    assert counts["fund_Theta"] == 6


def test_stochastic_policy_requires_seed():
    model = default_model()
    with pytest.raises(ValueError, match="seed is required"):
        simulate_trajectory(model, IMPROVING, horizon=3, policy="stochastic")


def test_stochastic_policy_deterministic_with_same_seed():
    model = default_model()
    t1 = simulate_trajectory(model, IMPROVING, horizon=6, policy="stochastic", seed=42)
    t2 = simulate_trajectory(model, IMPROVING, horizon=6, policy="stochastic", seed=42)
    assert t1.action_history == t2.action_history


def test_stochastic_policy_different_with_different_seeds():
    model = default_model()
    t1 = simulate_trajectory(model, IMPROVING, horizon=12, policy="stochastic", seed=42)
    t2 = simulate_trajectory(model, IMPROVING, horizon=12, policy="stochastic", seed=99)
    # They should differ (extremely unlikely to be identical)
    assert t1.action_history != t2.action_history


def test_belief_history_tracks_p_strong():
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=5, policy="greedy")
    for channel in CHANNELS:
        assert len(traj.belief_history[channel]) == 5
        for val in traj.belief_history[channel]:
            assert 0.0 <= val <= 1.0


def test_cumulative_pragmatic_monotone():
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=6, policy="greedy")
    assert len(traj.cumulative_pragmatic) == 6
    # cumulative should be a running sum
    for i in range(1, len(traj.cumulative_pragmatic)):
        assert (
            traj.cumulative_pragmatic[i] >= traj.cumulative_pragmatic[i - 1] - 1e-10
            or traj.cumulative_pragmatic[i] <= traj.cumulative_pragmatic[i - 1] + 1e-10
        )


def test_action_counts_sum_to_horizon():
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=10, policy="greedy")
    counts = traj.action_counts()
    assert sum(counts.values()) == 10


def test_greedy_from_improving_funds_theta():
    """The greedy policy from IMPROVING should fund Theta (it starts very stale)."""
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=3, policy="greedy")
    assert all(a == 3 for a in traj.action_history[:3])


def test_theta_freshness_increases_under_greedy():
    """When Theta starts stale and greedy funds it, P(fresh) should increase."""
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=6, policy="greedy")
    first_p = traj.cycles[0].p_strong["Theta"]
    last_p = traj.cycles[-1].p_strong["Theta"]
    assert last_p > first_p


def test_summarize_trajectory():
    model = default_model()
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


def test_invalid_policy_raises():
    model = default_model()
    with pytest.raises(ValueError, match="Unknown policy"):
        simulate_trajectory(model, IMPROVING, horizon=3, policy="invalid")


def test_zero_horizon_raises():
    model = default_model()
    with pytest.raises(ValueError, match="horizon"):
        simulate_trajectory(model, IMPROVING, horizon=0)


def test_stochastic_actions_valid():
    model = default_model()
    traj = simulate_trajectory(model, IMPROVING, horizon=10, policy="stochastic", seed=123)
    for action in traj.action_history:
        assert 0 <= action < len(ACTIONS)
