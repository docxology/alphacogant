"""Multi-cycle simulation: the firm running its self-improvement loop.

This module wraps the engine's inference, EFE evaluation, and transition
machinery into a single deterministic trajectory recorder.  It exists so
that the manuscript can show — not just describe — what happens when the
firm runs for ``H`` cycles: beliefs move from weak to strong, the funded
channel shifts from exploration to exploitation, and the t-RSI certificate
gates commits.

All computation comes from the engine modules; this module only orchestrates
and records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from alphacogant.efe.free_energy import expected_free_energy, marginal_return_vector
from alphacogant.model.channels import ACTIONS, CHANNELS, action_index
from alphacogant.model.generative_model import EconomicWorldModel, validate_belief_map
from alphacogant.trsi.t_rsi import DEFAULT_HORIZON, _advance, _greedy_action


@dataclass(frozen=True)
class CycleRecord:
    """One cycle of the simulated corporate trajectory."""

    cycle: int
    belief: dict[str, np.ndarray]
    action: int
    action_name: str
    efe_pragmatic: float
    efe_epistemic: float
    efe_total: float
    marginal_returns: dict[int, float]
    p_strong: dict[str, float]


@dataclass(frozen=True)
class TrajectoryResult:
    """Full recorded trajectory of the firm over ``horizon`` cycles."""

    cycles: list[CycleRecord]
    belief_history: dict[str, list[float]]  # channel -> [p_strong_0, p_strong_1, ...]
    action_history: list[int]
    cumulative_pragmatic: list[float]

    @property
    def horizon(self) -> int:
        return len(self.cycles)

    def action_counts(self) -> dict[str, int]:
        """How many times each action was selected over the trajectory."""
        counts: dict[str, int] = {name: 0 for name in ACTIONS}
        for record in self.cycles:
            counts[record.action_name] += 1
        return counts


def _belief_p_strong(belief: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {channel: float(belief[channel][1]) for channel in CHANNELS}


def simulate_trajectory(
    model: EconomicWorldModel,
    initial_belief: Mapping[str, np.ndarray],
    horizon: int = DEFAULT_HORIZON,
    *,
    policy: str = "greedy",
    seed: int | None = None,
) -> TrajectoryResult:
    """Run the firm for ``horizon`` cycles and record the full trajectory.

    Parameters
    ----------
    model
        The Economic World Model (A/B/C/D matrices).
    initial_belief
        Starting belief over the five channel factors.
    horizon
        Number of cycles to simulate.
    policy
        ``"greedy"`` (default) picks the action with the highest negative-EFE
        each cycle; ``"hold"`` never funds; ``"fund_theta"`` always refreshes
        Parameters; ``"stochastic"`` samples from the policy posterior.
    seed
        Required when ``policy="stochastic"``; ignored otherwise.
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if policy not in {"greedy", "hold", "fund_theta", "stochastic"}:
        raise ValueError(
            f"Unknown policy {policy!r}; expected one of "
            "'greedy', 'hold', 'fund_theta', 'stochastic'."
        )
    if policy == "stochastic" and seed is None:
        raise ValueError("seed is required when policy='stochastic'.")

    rng = np.random.default_rng(seed) if seed is not None else None
    normalized = validate_belief_map(initial_belief, context="initial_belief")
    current = {channel: normalized[channel].copy() for channel in CHANNELS}

    cycles: list[CycleRecord] = []
    belief_history: dict[str, list[float]] = {channel: [] for channel in CHANNELS}
    action_history: list[int] = []
    cumulative_pragmatic: list[float] = []
    running_sum = 0.0

    for cycle_idx in range(horizon):
        for channel in CHANNELS:
            belief_history[channel].append(float(current[channel][1]))

        if policy == "greedy":
            action = _greedy_action(model, current)
        elif policy == "hold":
            action = action_index("hold")
        elif policy == "fund_theta":
            action = action_index("fund_Theta")
        else:  # stochastic
            from alphacogant.efe.free_energy import policy_posterior

            posterior = policy_posterior(model, current)
            action = int(rng.choice(len(ACTIONS), p=posterior))  # type: ignore[union-attr]

        action_history.append(action)
        returns = marginal_return_vector(model, current)
        efe = expected_free_energy(model, current, action)
        running_sum += efe.pragmatic

        record = CycleRecord(
            cycle=cycle_idx,
            belief={channel: current[channel].copy() for channel in CHANNELS},
            action=action,
            action_name=ACTIONS[action],
            efe_pragmatic=efe.pragmatic,
            efe_epistemic=efe.epistemic,
            efe_total=efe.total,
            marginal_returns=dict(returns),
            p_strong=_belief_p_strong(current),
        )
        cycles.append(record)
        cumulative_pragmatic.append(running_sum)

        current = _advance(model, current, action)

    return TrajectoryResult(
        cycles=cycles,
        belief_history=belief_history,
        action_history=action_history,
        cumulative_pragmatic=cumulative_pragmatic,
    )


def summarize_trajectory(trajectory: TrajectoryResult) -> dict[str, str]:
    """Return a compact summary of a trajectory for the manuscript."""
    if not trajectory.cycles:
        raise ValueError("Cannot summarize an empty trajectory.")
    first = trajectory.cycles[0]
    last = trajectory.cycles[-1]
    action_counts = trajectory.action_counts()
    # Exploration ratio: fraction of cycles spent on epistemic channels (S, Z)
    epistemic_cycles = action_counts.get("fund_S", 0) + action_counts.get("fund_Z", 0)
    return {
        "horizon": str(trajectory.horizon),
        "first_funded": first.action_name,
        "last_funded": last.action_name,
        "first_p_strong_theta": f"{first.p_strong['Theta']:.4f}",
        "last_p_strong_theta": f"{last.p_strong['Theta']:.4f}",
        "first_p_strong_I": f"{first.p_strong['I']:.4f}",
        "last_p_strong_I": f"{last.p_strong['I']:.4f}",
        "total_pragmatic": f"{sum(r.efe_pragmatic for r in trajectory.cycles):.4f}",
        "total_epistemic": f"{sum(r.efe_epistemic for r in trajectory.cycles):.4f}",
        "exploration_ratio": f"{epistemic_cycles / trajectory.horizon:.4f}",
        "dominant_action": max(action_counts, key=lambda k: action_counts[k]),
    }


__all__ = [
    "CycleRecord",
    "TrajectoryResult",
    "simulate_trajectory",
    "summarize_trajectory",
]
