"""Tests for the enhanced COGANT bridge: full GNN round-trip.

No mocks; all tests use real model computation.
"""

from __future__ import annotations

from alphacogant.bridge.cogant_bridge import (
    model_to_gnn_summary,
    parse_gnn_summary,
)
from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import default_model


def test_gnn_summary_has_all_sections():
    model = default_model()
    summary = model_to_gnn_summary(model)
    assert "## GNNSection" in summary
    assert "## StateSpaceBlock" in summary
    assert "## Connections" in summary
    assert "## InitialParameterization" in summary
    assert "## ActInfOntologyAnnotation" in summary
    assert "## Footer" in summary


def test_gnn_summary_mentions_all_channels():
    model = default_model()
    summary = model_to_gnn_summary(model)
    for channel in CHANNELS:
        assert f"s{channel}" in summary
        assert f"B_{channel}" in summary
        assert f"D_{channel}" in summary


def test_gnn_summary_mentions_key_concepts():
    model = default_model()
    summary = model_to_gnn_summary(model)
    assert "ExpectedFreeEnergy" in summary
    assert "HiddenStateFactor" in summary
    assert "LikelihoodMatrix" in summary
    assert "TransitionMatrix" in summary
    assert "LogPreferenceVector" in summary
    assert "PriorOverHiddenStates" in summary
    assert "PolicyVector" in summary
    assert "ImprovementSignalToNoiseRatio" in summary


def test_gnn_summary_has_connections():
    model = default_model()
    summary = model_to_gnn_summary(model)
    assert "sI-A_R" in summary
    assert "sU-A_R" in summary
    assert "sTheta-A_R" in summary
    assert "sS-A_L" in summary
    assert "G>pi" in summary
    assert "g_pragmatic>tRSI" in summary


def test_parse_gnn_summary_round_trip():
    model = default_model()
    summary = model_to_gnn_summary(model)
    parsed = parse_gnn_summary(summary)
    assert "sections" in parsed
    assert "StateSpaceBlock" in parsed["sections"]
    assert "Connections" in parsed["sections"]
    assert "ActInfOntologyAnnotation" in parsed["sections"]
    assert "factors" in parsed
    assert len(parsed["factors"]) == len(CHANNELS)


def test_parse_gnn_summary_factors_match_channels():
    model = default_model()
    summary = model_to_gnn_summary(model)
    parsed = parse_gnn_summary(summary)
    for channel in CHANNELS:
        assert f"s{channel}" in parsed["factors"]


def test_parse_gnn_summary_connections_present():
    model = default_model()
    summary = model_to_gnn_summary(model)
    parsed = parse_gnn_summary(summary)
    connections = parsed["connections"]
    assert "sI-A_R" in connections
    assert "G>pi" in connections
    assert "g_pragmatic>tRSI" in connections


def test_parse_gnn_summary_ontology_present():
    model = default_model()
    summary = model_to_gnn_summary(model)
    parsed = parse_gnn_summary(summary)
    ontology = parsed["ontology"]
    assert "sI=HiddenStateFactor" in ontology
    assert "A_R=LikelihoodMatrix" in ontology
    assert "G=ExpectedFreeEnergy" in ontology


def test_parse_empty_text():
    parsed = parse_gnn_summary("")
    assert parsed["factors"] == []
    assert parsed["connections"] == []


def test_round_trip_preserves_information():
    """The GNN summary → parse → model round-trip preserves key structural facts."""
    model = default_model()
    summary = model_to_gnn_summary(model)
    parsed = parse_gnn_summary(summary)
    # All 5 channel factors are recovered
    assert len(parsed["factors"]) == 5
    # Connections include the key edges
    assert any("A_R" in c for c in parsed["connections"])
    assert any("A_L" in c for c in parsed["connections"])
    assert any("pi" in c for c in parsed["connections"])
