"""COGANT-style bridge functions from firm structure to AlphaCOGANT priors."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from alphacogant.channels import ACTIONS, CHANNELS
from alphacogant.generative_model import EconomicWorldModel

_ALIASES: dict[str, tuple[str, ...]] = {
    "I": ("book_size", "book", "aum"),
    "S": ("data_feeds", "feeds"),
    "U": ("venues", "routes"),
    "Theta": ("models", "ewm_models"),
    "Z": ("researchers", "research_staff"),
}


def _resolve_count(spec: Mapping[str, object], channel: str) -> float:
    for key in _ALIASES[channel]:
        if key in spec:
            value = float(spec[key])
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite.")
            if value < 0.0:
                raise ValueError(f"{key} must be non-negative.")
            return value
    raise ValueError(f"Firm structure is missing a count for channel {channel}.")


def _strong_mass(count: float) -> float:
    scaled = np.log1p(count) - 1.0
    return float(1.0 / (1.0 + np.exp(-scaled)))


def firm_structure_to_channels(spec: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Map firm-structure counts onto channel prior beliefs."""
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be a mapping of firm-structure counts.")
    priors: dict[str, np.ndarray] = {}
    for channel in CHANNELS:
        strong_mass = _strong_mass(_resolve_count(spec, channel))
        priors[channel] = np.array([1.0 - strong_mass, strong_mass], dtype=float)
    return priors


def model_to_gnn_summary(model: EconomicWorldModel) -> str:
    """Emit a short GNN-style summary block from an economic world model."""
    lines = [
        "## AlphaCOGANTSummary",
        "Factors: I, S, U, Theta, Z",
        f"Actions: {', '.join(ACTIONS)}",
        f"A_R shape: {tuple(model.A_R.shape)}",
        f"A_L shape: {tuple(model.A_L.shape)}",
        "Objective: ExpectedFreeEnergy",
    ]
    return "\n".join(lines)


__all__ = [
    "firm_structure_to_channels",
    "model_to_gnn_summary",
]
