"""Consumer-facing structural inspection of independently authored GNN text."""

from __future__ import annotations

from alphacogant.bridge.cogant_bridge import parse_gnn_summary


def test_inspector_extracts_authored_factors_edges_and_ontology():
    # The inspector is descriptive, not restricted to the finite numeric profile.
    source = """
    A separately authored model

    ## StateSpaceBlock
      sCapital[4,1,type=float]
      oReturn[3,1,type=float]
      sRisk[2,1,type=float]

    ## Connections
      sCapital-A_Return
      sRisk-A_Return
      A_Return-oReturn

    ## ActInfOntologyAnnotation
      sCapital=HiddenStateFactor
      sRisk=HiddenStateFactor
      A_Return=LikelihoodMatrix
    """
    inspected = parse_gnn_summary(source)
    assert inspected["factors"] == ["sCapital", "sRisk"]
    assert inspected["connections"] == ["sCapital-A_Return", "sRisk-A_Return", "A_Return-oReturn"]
    assert inspected["ontology"] == [
        "sCapital=HiddenStateFactor",
        "sRisk=HiddenStateFactor",
        "A_Return=LikelihoodMatrix",
    ]


def test_inspector_reports_missing_structural_data():
    inspected = parse_gnn_summary("An unstructured description, without declarations.")
    assert inspected["factors"] == []
    assert inspected["connections"] == []
    assert inspected["ontology"] == []
