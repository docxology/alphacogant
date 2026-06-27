"""COGANT bridge subpackage: firm structure → priors → GNN summary round-trip."""

from alphacogant.bridge.cogant_bridge import (
    firm_structure_to_channels,
    model_to_gnn_summary,
    parse_gnn_summary,
)

__all__ = [
    "firm_structure_to_channels",
    "model_to_gnn_summary",
    "parse_gnn_summary",
]
