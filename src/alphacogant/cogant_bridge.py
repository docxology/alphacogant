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
    """Emit a full GNN-style summary block from an economic world model.

    This is the round-trip half of COGANT: ``default_model()`` realises the
    GNN file ``models/alphafund_ewm.md`` as executable NumPy arrays, and this
    function re-emits a GNN-style text block from those arrays.  The output
    is not byte-identical to the source file (it normalises and reformats)
    but carries every section a GNN processor needs: StateSpaceBlock,
    Connections, InitialParameterization, and ActInfOntologyAnnotation.
    """
    lines = [
        "## GNNSection",
        "ActInfPOMDP_AlphaFund_RoundTrip",
        "",
        "## ModelName",
        "AlphaFund Economic World Model (round-trip from EconomicWorldModel)",
        "",
        "## StateSpaceBlock",
    ]
    for channel in CHANNELS:
        lines.append(f"s{channel}[2,1,type=float]")
    lines.append(f"A_R[{','.join(str(d) for d in model.A_R.shape)},type=float]")
    lines.append(f"A_L[{','.join(str(d) for d in model.A_L.shape)},type=float]")
    for channel in CHANNELS:
        lines.append(f"B_{channel}[{','.join(str(d) for d in model.B[channel].shape)},type=float]")
    lines.append(f"C_R[{model.C_R.shape[0]},type=float]")
    lines.append(f"C_L[{model.C_L.shape[0]},type=float]")
    for channel in CHANNELS:
        lines.append(f"D_{channel}[{model.D[channel].shape[0]},type=float]")
    lines.append(f"pi[{len(ACTIONS)},type=float]")
    lines.append("G[pi,type=float]")
    lines.append("tRSI[1,type=float]")
    lines.append("")
    lines.append("## Connections")
    for channel in CHANNELS:
        lines.append(f"D_{channel}>s{channel}")
    lines.append("sI-A_R")
    lines.append("sU-A_R")
    lines.append("sTheta-A_R")
    lines.append("sS-A_L")
    lines.append("sTheta-A_L")
    lines.append("A_R-o_R")
    lines.append("A_L-o_L")
    lines.append("C_R>G")
    lines.append("C_L>G")
    lines.append("G>pi")
    for channel in CHANNELS:
        lines.append(f"B_{channel}>pi")
    lines.append("g_pragmatic>tRSI")
    lines.append("sTheta>tRSI")
    lines.append("")
    lines.append("## InitialParameterization")
    lines.append(f"A_R shape: {tuple(model.A_R.shape)}")
    lines.append(f"A_L shape: {tuple(model.A_L.shape)}")
    for channel in CHANNELS:
        lines.append(f"B_{channel} shape: {tuple(model.B[channel].shape)}")
    lines.append(f"C_R: {np.array2string(model.C_R, separator=', ')}")
    lines.append(f"C_L: {np.array2string(model.C_L, separator=', ')}")
    for channel in CHANNELS:
        lines.append(f"D_{channel}: {np.array2string(model.D[channel], separator=', ')}")
    lines.append("")
    lines.append("## ActInfOntologyAnnotation")
    for channel in CHANNELS:
        lines.append(f"s{channel}=HiddenStateFactor")
    lines.append("A_R=LikelihoodMatrix")
    lines.append("A_L=LikelihoodMatrix")
    for channel in CHANNELS:
        lines.append(f"B_{channel}=TransitionMatrix")
    lines.append("C_R=LogPreferenceVector")
    lines.append("C_L=LogPreferenceVector")
    for channel in CHANNELS:
        lines.append(f"D_{channel}=PriorOverHiddenStates")
    lines.append("pi=PolicyVector")
    lines.append("G=ExpectedFreeEnergy")
    lines.append("tRSI=ImprovementSignalToNoiseRatio")
    lines.append("")
    lines.append("## Footer")
    lines.append("Round-trip GNN summary produced by cogant_bridge.model_to_gnn_summary.")
    lines.append("Objective: ExpectedFreeEnergy")
    return "\n".join(lines)


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
