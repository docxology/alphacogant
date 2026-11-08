"""Coefficient-level finite-profile and independently enumerated joint checks."""

from __future__ import annotations

import re
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from alphacogant.bridge.cogant_bridge import model_to_gnn_summary
from alphacogant.bridge.gnn import (
    MAX_GNN_BYTES,
    joint_matrices,
    model_from_gnn,
    model_parameters,
    model_to_cogant_gnn,
)
from alphacogant.efe.free_energy import efe_vector
from alphacogant.model.channels import ACTIONS, CHANNELS
from alphacogant.model.generative_model import default_model

ROOT = Path(__file__).resolve().parents[1]


def test_authored_gnn_matches_every_engine_coefficient(model):
    parsed = model_from_gnn((ROOT / "models/alphafund_ewm.md").read_text())
    for name, array in model_parameters(model).items():
        np.testing.assert_allclose(model_parameters(parsed)[name], array, atol=1e-15, rtol=0)


def test_numeric_round_trip_preserves_custom_coefficients_and_behavior(model, prior):
    custom = default_model(
        C_R=[-4.125, 0.725, 7.123456789012345],
        D={c: [0.1234567890123456, 0.8765432109876544] for c in CHANNELS},
    )
    restored = model_from_gnn(model_to_gnn_summary(custom))
    for name, array in model_parameters(custom).items():
        np.testing.assert_array_equal(model_parameters(restored)[name], array)
    for expected, actual in zip(
        efe_vector(custom, prior), efe_vector(restored, prior), strict=True
    ):
        np.testing.assert_allclose(
            [actual.pragmatic, actual.epistemic, actual.total],
            [expected.pragmatic, expected.epistemic, expected.total],
            atol=1e-15,
            rtol=0,
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("## Time", "## TimeMissing"),
        lambda s: s.replace("GNN v1", "GNN v999"),
        lambda s: s.replace("G>pi", "missing>pi"),
        lambda s: s.replace("G>pi", "G>>pi"),
        lambda s: s.replace("G>pi\n", ""),
        lambda s: s.replace("G>pi", "pi>G"),
        lambda s: s.replace("G>pi", "G>pi\nG>pi"),
        lambda s: s.replace("G>pi", "G>pi\no_R>sI"),
        lambda s: s.replace("G>pi", "G>pi\nsZ>g_epistemic"),
        lambda s: s + "\n## Time\nDynamic\n",
        lambda s: s.replace("A_R[3,2,2,2", "A_R[3,2,2,3"),
        lambda s: s.replace("sI[2,1", "sI[3,1"),
        lambda s: s.replace("o_R[3,1", "o_R[2,1"),
        lambda s: s.replace("o_L[3,1", "o_L[3,2"),
        lambda s: s.replace("u[1,type=float]", "u[6,type=float]"),
        lambda s: s.replace("pi[6", "pi[5"),
        lambda s: s.replace("G[6", "G[1"),
        lambda s: s.replace("g_epistemic[6", "g_epistemic[1"),
        lambda s: s.replace("g_pragmatic[6", "g_pragmatic[1"),
        lambda s: s.replace("tRSI[1", "tRSI[6"),
        lambda s: s.replace("D_I[2,type=float]", "D_I[2,type=int]"),
        lambda s: s.replace("u[1,type=float]", "u[1,type=bool]"),
        lambda s: s.replace("u[1,type=float]", "u[1,type=float] trailing"),
        lambda s: s.replace("u[1,type=float]", ""),
        lambda s: s.replace("u[1,type=float]", "u[1,type=float]\nextra[1,type=float]"),
        lambda s: s.replace("num_actions: 6", "num_actions: 7"),
        lambda s: s.replace("num_channel_factors: 5", "num_channel_factors: 4"),
        lambda s: s.replace("policy_horizon: 1", "policy_horizon: 2"),
        lambda s: s.replace("policy_horizon: 1", ""),
        lambda s: s.replace("policy_horizon: 1", "policy_horizon: 1\npolicy_horizon: 1"),
        lambda s: s.replace("D_I={(0.6, 0.4)}", "D_I={(0.2, 0.2)}"),
        lambda s: s.replace("D_I={(0.6, 0.4)}", "D_I={(True, False)}"),
        lambda s: s.replace("D_I={(0.6, 0.4)}", "D_I={__import__('os').getcwd()}"),
        lambda s: s.replace("D_I={(0.6, 0.4)}", "D_I={(0.6, 0.4)}\nD_I={(0.6, 0.4)}"),
        lambda s: s.replace("D_I={(0.6, 0.4)}", "D_I={(1e999, 0.4)}"),
        lambda s: s.replace("D_I={(0.6, 0.4)}", "unknown={(0.6, 0.4)}"),
        lambda s: s.replace("D_I={(0.6, 0.4)}", "D_I={(0.6, 0.4, 0.0)}"),
        lambda s: s.replace("D_I={(0.6, 0.4)}", ""),
        lambda s: s.replace("D_I={(0.6, 0.4)}", "nonsense"),
        lambda s: s.replace("D_I[2,type=float]", "D_I[2,type=float]\nD_I[2,type=float]"),
    ],
)
def test_profile_rejects_incomplete_ambiguous_or_executable_input(model, change):
    original = model_to_gnn_summary(model)
    changed = change(original)
    assert changed != original
    with pytest.raises(ValueError):
        model_from_gnn(changed)


def test_profile_bounds_input_before_parsing():
    with pytest.raises(ValueError, match="exceeds"):
        model_from_gnn("x" * (MAX_GNN_BYTES + 1))


def test_joint_compilation_matches_independent_enumeration(model):
    matrices = joint_matrices(model)
    states = list(product((0, 1), repeat=5))
    for current, state in enumerate(states):
        i, s, u, theta, _ = state
        assert matrices["D"][current] == pytest.approx(
            np.prod([model.D[channel][state[k]] for k, channel in enumerate(CHANNELS)])
        )
        for reward, loss in product(range(3), repeat=2):
            assert matrices["A"][3 * reward + loss, current] == (
                model.A_R[reward, i, u, theta] * model.A_L[loss, s, theta]
            )
            assert matrices["C"][3 * reward + loss] == model.C_R[reward] + model.C_L[loss]
        for nxt, successor in enumerate(states):
            for action in range(len(ACTIONS)):
                expected = 1.0
                for k, channel in enumerate(CHANNELS):
                    expected *= model.B[channel][successor[k], state[k], action]
                assert matrices["B"][nxt, current, action] == expected
    np.testing.assert_allclose(matrices["A"].sum(axis=0), 1, atol=1e-15)
    np.testing.assert_allclose(matrices["B"].sum(axis=0), 1, atol=1e-15)
    np.testing.assert_allclose(matrices["D"].sum(), 1, atol=1e-15)


def test_cogant_fence_contains_every_value_without_rounding(model):
    source = model_to_cogant_gnn(model)
    block = source.split("```gnn-matrices\n")[1].split("```")[0]
    matrices = joint_matrices(model)
    # Independent test decoder for COGANT's row/action-slice wire format.
    groups = re.split(r"[ABCD]\[\[.*?\]\]\n", block)[1:]
    decoded = []
    for group in groups:
        rows = [
            [float(v) for v in line.split()]
            for line in group.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        decoded.append(np.array(rows))
    np.testing.assert_array_equal(decoded[0], matrices["A"])
    np.testing.assert_array_equal(decoded[1].reshape(6, 32, 32).transpose(1, 2, 0), matrices["B"])
    np.testing.assert_array_equal(decoded[2].ravel(), matrices["C"])
    np.testing.assert_array_equal(decoded[3].ravel(), matrices["D"])


@pytest.mark.parametrize("mutation", ["coefficient", "rows", "slices", "duplicate", "open", "text"])
def test_joint_fence_cannot_contradict_factor_model(model, mutation):
    source = model_to_cogant_gnn(model)
    prefix, block = source.split("```gnn-matrices\n")
    lines = block.splitlines()
    if mutation == "coefficient":
        tokens = lines[1].split()
        tokens[0] = "0.123456"
        lines[1] = " ".join(tokens)
    elif mutation == "rows":
        del lines[1]
    elif mutation == "slices":
        lines[0] = "A[[rows=9][cols=31]]"
    elif mutation == "duplicate":
        lines += ["```gnn-matrices", "```"]
    elif mutation == "open":
        lines = lines[:-1]
    else:
        lines[1] = "invalid numeric value"
    with pytest.raises(ValueError):
        model_from_gnn(prefix + "```gnn-matrices\n" + "\n".join(lines))


def test_joint_fence_agrees_with_factor_model(model):
    restored = model_from_gnn(model_to_cogant_gnn(model))
    for name, expected in model_parameters(model).items():
        np.testing.assert_array_equal(model_parameters(restored)[name], expected)


@pytest.mark.parametrize("content", ["X" * 100_000, "D_I={(" + "9" * 401 + ", 0)}"])
def test_malformed_assignment_and_huge_integer_are_value_errors(model, content):
    source = model_to_gnn_summary(model)
    source = source.replace(
        "## InitialParameterization\n", "## InitialParameterization\n" + content + "\n"
    )
    with pytest.raises(ValueError):
        model_from_gnn(source)


def test_assignment_scanner_preserves_multiline_literals_and_rejects_leftovers(model):
    source = model_to_gnn_summary(model).replace(
        "D_I={(0.6, 0.4)}", "D_I={\n(0.6, # weak\n0.4)\n}"
    )
    restored = model_from_gnn(source)
    for name, expected in model_parameters(model).items():
        np.testing.assert_array_equal(model_parameters(restored)[name], expected)
    for content in ("trailing", "D_I={", "D_I={{0.6, 0.4}}"):
        with pytest.raises(ValueError):
            model_from_gnn(source.replace("\n## Equations", "\n" + content + "\n## Equations"))


def test_discrete_declaration_notation_and_descriptive_sections_preserve_coefficients(model):
    source = model_to_gnn_summary(model)
    for name, dims in (("o_R", "3,1"), ("o_L", "3,1"), ("u", "1")):
        source = source.replace(f"{name}[{dims},type=float]", f"{name}[{dims},type=int]")
    source = source.replace("G = -g_pragmatic - g_epistemic", "Descriptive equation, not executed")
    source = source.replace("ModelTimeHorizon=1", "ModelTimeHorizon=Finite")
    source = source.replace("sI=HiddenStateFactor", "sI=DescriptiveConcept")
    restored = model_from_gnn(source)
    for name, expected in model_parameters(model).items():
        np.testing.assert_array_equal(model_parameters(restored)[name], expected)


def test_joint_export_rejects_unrepresentable_combined_preferences():
    model = default_model(C_R=[1e308] * 3, C_L=[1e308] * 3)
    with np.errstate(over="raise", invalid="raise"), pytest.raises(ValueError):
        model_to_cogant_gnn(model)
