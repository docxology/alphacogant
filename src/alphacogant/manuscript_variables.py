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

from alphacogant.channels import CHANNEL_ROLES, CHANNELS
from alphacogant.free_energy import expected_free_energy, marginal_return_vector
from alphacogant.generative_model import belief_prior, default_model
from alphacogant.operating_points import (
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    IMPROVING,
    COASTING,
)
from alphacogant.t_rsi import DEFAULT_HORIZON, bootstrap_t_rsi

PLANNING_HORIZON = DEFAULT_HORIZON

_MANUSCRIPT_DIR = Path(__file__).resolve().parents[2] / "manuscript"
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
    from alphacogant.channels import action_index

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


def generate_variables() -> dict[str, str]:
    """Generate every manuscript token from the live deterministic model."""
    model = default_model()

    # Exactly two bootstraps (improving + coasting); results reused for every
    # t-RSI / create / decay token below. Fixed seeds keep this deterministic.
    improving = bootstrap_t_rsi(
        model, IMPROVING, np.random.default_rng(BOOTSTRAP_SEED), n=BOOTSTRAP_N,
        concentration=BOOTSTRAP_CONCENTRATION,
    )
    coasting = bootstrap_t_rsi(
        model, COASTING, np.random.default_rng(BOOTSTRAP_SEED), n=BOOTSTRAP_N,
        concentration=BOOTSTRAP_CONCENTRATION,
    )

    # The funded channel is reported at the neutral prior operating point. It comes
    # out epistemic (Sensors) with negative immediate pragmatic value — the firm
    # funds it to learn, not to earn this cycle, which is the explore behaviour the
    # epistemic/pragmatic split predicts.
    prior = belief_prior(model)
    returns = marginal_return_vector(model, prior)
    funded_action = max(range(5), key=lambda action: returns[action])
    funded_result = expected_free_energy(model, prior, funded_action)

    return {
        "TOKEN": "AlphaCOGANT",
        "NUM_CHANNELS": str(len(CHANNELS)),
        "NUM_ACTIONS": str(model.B[CHANNELS[0]].shape[2]),
        "PLANNING_HORIZON": str(PLANNING_HORIZON),
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
    }


__all__ = [
    "PLANNING_HORIZON",
    "generate_variables",
]
