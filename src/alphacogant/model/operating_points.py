"""Canonical operating-point beliefs and bootstrap constants.

The manuscript, the figure scripts, and the test suite all reference the same
two operating points (self-improving and coasting) and the same bootstrap
parameters (seed, sample count, concentration).  Previously these were
duplicated as literal dictionaries in six+ files, creating a drift risk where
one copy could silently diverge from the others.

This module is the single source of truth.  Every consumer imports from here
instead of re-declaring the literals.
"""

from __future__ import annotations

import numpy as np

from alphacogant.model.channels import CHANNELS

# -- Operating-point beliefs ---------------------------------------------------

#: The self-improving operating point: weak channels, stale parameters.
#: The greedy policy has cheap epistemic and pragmatic gains to fund.
IMPROVING_RAW: dict[str, tuple[float, float]] = {
    "I": (0.85, 0.15),
    "S": (0.85, 0.15),
    "U": (0.85, 0.15),
    "Theta": (0.80, 0.20),
    "Z": (0.60, 0.40),
}

#: The coasting operating point: strong channels, fresh parameters.
#: The greedy policy coasts on ``hold``; decay dominates.
COASTING_RAW: dict[str, tuple[float, float]] = {
    "I": (0.05, 0.95),
    "S": (0.10, 0.90),
    "U": (0.05, 0.95),
    "Theta": (0.10, 0.90),
    "Z": (0.30, 0.70),
}


def as_belief(raw: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """Convert a ``(weak, strong)`` spec into a validated belief map.

    The returned dict is keyed by :data:`alphacogant.channels.CHANNELS` and
    each value is a float64 array of shape ``(2,)``.
    """
    return {channel: np.array(raw[channel], dtype=float) for channel in CHANNELS}


#: Convenience: the improving belief as a ready-to-use channel→array map.
IMPROVING: dict[str, np.ndarray] = as_belief(IMPROVING_RAW)

#: Convenience: the coasting belief as a ready-to-use channel→array map.
COASTING: dict[str, np.ndarray] = as_belief(COASTING_RAW)

# -- Bootstrap constants -------------------------------------------------------

#: Bootstrap seed shared by manuscript variables, statistics, and figure scripts.
BOOTSTRAP_SEED: int = 20240623

#: Canonical bootstrap sample count shared by manuscript variables and figures.
BOOTSTRAP_N: int = 2560

#: Bootstrap Dirichlet concentration — the firm's belief precision.
#: Must match ``t_rsi.bootstrap_t_rsi``'s default.
BOOTSTRAP_CONCENTRATION: float = 12.0


__all__ = [
    "BOOTSTRAP_CONCENTRATION",
    "BOOTSTRAP_N",
    "BOOTSTRAP_SEED",
    "COASTING",
    "COASTING_RAW",
    "IMPROVING",
    "IMPROVING_RAW",
    "as_belief",
]
