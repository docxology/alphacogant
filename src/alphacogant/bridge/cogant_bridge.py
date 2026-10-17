"""COGANT-style bridge functions from firm structure to AlphaCOGANT priors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import SupportsFloat, cast

import numpy as np

from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import EconomicWorldModel

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
            value = float(cast(SupportsFloat, spec[key]))
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
    """Serialize every coefficient using the AlphaCOGANT finite GNN profile.

    The parameter axes are observation/next-state first and action last. This
    is a numeric round trip; ``model_from_gnn`` restores the complete model.
    """
    from alphacogant.bridge.gnn import model_to_gnn

    return model_to_gnn(model)


def parse_gnn_summary(text: str) -> dict[str, list[str]]:
    """Parse a GNN-style summary block back into a structured dict.

    This is the inverse direction of ``model_to_gnn_summary`` — given a
    GNN text block, extract the declared shapes, connections, and ontology
    annotations.  Used to verify round-trip fidelity in tests.
    """
    sections: dict[str, list[str]] = {}
    current_section = "Preamble"
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped[3:]
            sections[current_section] = []
        elif stripped:
            sections.setdefault(current_section, []).append(stripped)

    result: dict[str, list[str]] = {"sections": list(sections.keys())}

    # Extract factor names from StateSpaceBlock
    state_block = sections.get("StateSpaceBlock", [])
    factors: list[str] = []
    for line in state_block:
        if line.startswith("s") and "[" in line:
            factors.append(line.split("[")[0])
    result["factors"] = factors

    # Extract connections
    connections = sections.get("Connections", [])
    result["connections"] = connections

    # Extract ontology annotations
    ontology = sections.get("ActInfOntologyAnnotation", [])
    result["ontology"] = ontology

    return result


__all__ = [
    "firm_structure_to_channels",
    "model_to_gnn_summary",
    "parse_gnn_summary",
]
