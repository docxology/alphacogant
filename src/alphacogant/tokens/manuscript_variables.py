"""Deterministic manuscript-token generation for AlphaCOGANT.

Every numeric the manuscript prose cites as a ``{{TOKEN}}`` is produced here from
the live model and a fixed-seed bootstrap, so the rendered document is proof that
the prose matches the engine. The headline t-RSI is reported at the *self-improving
operating point* (weak channels, stale parameters — where the greedy policy has
cheap epistemic and pragmatic gains to fund); a coasting operating point is also
reported so the manuscript shows the certificate honestly failing rather than
cherry-picking the one regime with a positive number.

Counts that are properties of *this* manuscript (number of numbered Definitions,
number of engine-generated figures) are introspected from the markdown so they
track the document rather than drifting as fixed literals. Everything else is read
straight off ``default_model()`` and the deterministic ``bootstrap_t_rsi``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from alphacogant.efe.free_energy import expected_free_energy, marginal_return_vector
from alphacogant.model.channels import CHANNEL_ROLES, CHANNELS
from alphacogant.model.generative_model import belief_prior, default_model
from alphacogant.model.operating_points import (
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COASTING,
    IMPROVING,
)
from alphacogant.trsi.t_rsi import DEFAULT_HORIZON, bootstrap_t_rsi

PLANNING_HORIZON = DEFAULT_HORIZON

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MANUSCRIPT_DIR = _PROJECT_ROOT / "manuscript"
_DEFINITION_RE = re.compile(r"\*\*Definition\s+D(\d+)\b")
_FIGURE_RE = re.compile(r"\{#fig:[A-Za-z0-9_]+\}")


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _channels_with_role(role: str) -> str:
    """Comma-joined channel names carrying the given role (deterministic order)."""
    return ", ".join(channel for channel in CHANNELS if CHANNEL_ROLES[channel] == role)


def _theta_decay_leak_probability(model) -> float:
    """P(fresh Theta -> stale | currently fresh) under the passive ``hold`` action.

    This is ``B_Theta[stale, fresh, hold]``: the per-cycle freshness mass that
    bleeds away when the firm does not refit, read directly off the normalized
    transition tensor.
    """
    from alphacogant.model.channels import action_index

    theta = model.B["Theta"]
    stale, fresh = 0, 1  # level0 = weak/stale, level1 = fresh
    return float(theta[stale, fresh, action_index("hold")])


def _count_definitions() -> int:
    """Number of distinct numbered Definitions (D1, D2, ...) in the manuscript."""
    ids: set[int] = set()
    for md in sorted(_MANUSCRIPT_DIR.glob("*.md")):
        ids.update(int(match) for match in _DEFINITION_RE.findall(md.read_text(encoding="utf-8")))
    return len(ids)


def _count_figures() -> int:
    """Number of distinct engine-generated figures ({#fig:...}) in the manuscript."""
    labels: set[str] = set()
    for md in sorted(_MANUSCRIPT_DIR.glob("*.md")):
        labels.update(_FIGURE_RE.findall(md.read_text(encoding="utf-8")))
    return len(labels)


def resolve_bootstrap_n(n: int | None) -> int:
    """Resolve the bootstrap resample count, defaulting to the published value."""
    return BOOTSTRAP_N if n is None else n


def generate_variables(n: int | None = None) -> dict[str, str]:
    """Generate every manuscript token from the live deterministic model.

    ``n`` overrides the bootstrap resample count. It defaults to the published
    ``BOOTSTRAP_N`` (the value rendered into the manuscript); tests pass a small
    ``n`` to exercise the full token surface quickly, since the token *contract*
    (determinism, key set, formatting, behaviour) does not depend on the resample
    count — only the bootstrap point values do, and those are reported at the
    published ``BOOTSTRAP_N``.
    """
    from alphacogant.stats.simulation import simulate_trajectory, summarize_trajectory
    from alphacogant.stats.statistics import (
        break_even_profile,
        compare_regimes,
        compute_regime_statistics,
    )

    bootstrap_n = resolve_bootstrap_n(n)
    model = default_model()

    # Exactly two bootstraps (improving + coasting); results reused for every
    # t-RSI / create / decay token below. Fixed seeds keep this deterministic.
    improving = bootstrap_t_rsi(
        model,
        IMPROVING,
        np.random.default_rng(BOOTSTRAP_SEED),
        n=bootstrap_n,
        concentration=BOOTSTRAP_CONCENTRATION,
    )
    coasting = bootstrap_t_rsi(
        model,
        COASTING,
        np.random.default_rng(BOOTSTRAP_SEED),
        n=bootstrap_n,
        concentration=BOOTSTRAP_CONCENTRATION,
    )

    # Full regime statistics (bootstrap CIs + effect sizes)
    stats_improving = compute_regime_statistics(
        model,
        IMPROVING,
        "Improving",
        n=bootstrap_n,
    )
    comparison = compare_regimes(model, n=bootstrap_n)
    improving_break_even = break_even_profile(model, IMPROVING, n=bootstrap_n)
    coasting_break_even = break_even_profile(model, COASTING, n=bootstrap_n)

    # The funded channel is reported at the neutral prior operating point. It comes
    # out epistemic (Sensors) with negative immediate pragmatic value — the firm
    # funds it to learn, not to earn this cycle, which is the explore behaviour the
    # epistemic/pragmatic split predicts.
    prior = belief_prior(model)
    returns = marginal_return_vector(model, prior)
    funded_action = max(range(5), key=lambda action: returns[action])
    funded_result = expected_free_energy(model, prior, funded_action)

    # Trajectory summary: the firm running greedily from the self-improving
    # operating point for the full planning horizon.
    trajectory = simulate_trajectory(model, IMPROVING, horizon=PLANNING_HORIZON, policy="greedy")
    traj_summary = summarize_trajectory(trajectory)

    return {
        "NUM_CHANNELS": str(len(CHANNELS)),
        "NUM_ACTIONS": str(model.B[CHANNELS[0]].shape[2]),
        "PLANNING_HORIZON": str(PLANNING_HORIZON),
        "BOOTSTRAP_N": str(bootstrap_n),
        "EPISTEMIC_CHANNELS": _channels_with_role("epistemic"),
        "PRAGMATIC_CHANNELS": _channels_with_role("pragmatic"),
        "FUNDED_CHANNEL": CHANNELS[funded_action],
        "FUNDED_EPISTEMIC": _format_float(funded_result.epistemic),
        "FUNDED_PRAGMATIC": _format_float(funded_result.pragmatic),
        "HEADLINE_T_RSI": _format_float(improving["t_rsi"]),
        "COASTING_T_RSI": _format_float(coasting["t_rsi"]),
        "CREATE_RATE_MEAN": _format_float(improving["create_mean"]),
        "DECAY_RATE_MEAN": _format_float(improving["decay_mean"]),
        "B_THETA_LEAK_PROB": _format_float(_theta_decay_leak_probability(model)),
        "NUM_DEFINITIONS": str(_count_definitions()),
        "NUM_FIGURES": str(_count_figures()),
        # Trajectory tokens
        "TRAJ_FIRST_FUNDED": traj_summary["first_funded"],
        "TRAJ_LAST_FUNDED": traj_summary["last_funded"],
        "TRAJ_THETA_DELTA": _format_float(
            float(traj_summary["last_p_strong_theta"]) - float(traj_summary["first_p_strong_theta"])
        ),
        "TRAJ_EXPLORATION_RATIO": traj_summary["exploration_ratio"],
        "TRAJ_DOMINANT_ACTION": traj_summary["dominant_action"],
        # Statistics tokens (bootstrap CIs + effect sizes)
        "CREATE_CI_LOWER": _format_float(stats_improving.create_ci.ci_lower),
        "CREATE_CI_UPPER": _format_float(stats_improving.create_ci.ci_upper),
        "DECAY_CI_LOWER": _format_float(stats_improving.decay_ci.ci_lower),
        "DECAY_CI_UPPER": _format_float(stats_improving.decay_ci.ci_upper),
        "COHEN_D_CREATE": _format_float(comparison.cohen_d_create),
        "COHEN_D_DECAY": _format_float(comparison.cohen_d_decay),
        "BREAK_EVEN_PROB": _format_float(improving_break_even.probability),
        "COASTING_BREAK_EVEN_PROB": _format_float(coasting_break_even.probability),
        "BREAK_EVEN_MARGIN_MEAN": _format_float(improving_break_even.margin_mean),
    }


__all__ = [
    "PLANNING_HORIZON",
    "generate_variables",
]
