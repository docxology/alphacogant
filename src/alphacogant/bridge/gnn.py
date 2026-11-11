"""Lossless finite GNN profile and COGANT's complete joint-matrix extension.

This is a deliberately bounded parser for this project's A/B/C/D profile, not
an implementation of the full GNN language. Literals are data, never code.
"""

from __future__ import annotations

import ast
import re
from itertools import product

import numpy as np

from alphacogant.model.channels import ACTIONS, CHANNELS
from alphacogant.model.generative_model import EconomicWorldModel, _validate_model

MAX_GNN_BYTES = 1_000_000
PARAMETER_SHAPES = {
    "A_R": (3, 2, 2, 2),
    "A_L": (3, 2, 2),
    **{f"B_{c}": (2, 2, len(ACTIONS)) for c in CHANNELS},
    "C_R": (3,),
    "C_L": (3,),
    **{f"D_{c}": (2,) for c in CHANNELS},
}
DECLARATION_SHAPES = {
    **{f"s{c}": (2, 1) for c in CHANNELS},
    **PARAMETER_SHAPES,
    "o_R": (3, 1),
    "o_L": (3, 1),
    "pi": (6,),
    "u": (1,),
    "G": (6,),
    "g_epistemic": (6,),
    "g_pragmatic": (6,),
    "tRSI": (1,),
}
FIXED_CONNECTIONS = (
    *(f"D_{c}>s{c}" for c in CHANNELS),
    "sI-A_R",
    "sU-A_R",
    "sTheta-A_R",
    "sS-A_L",
    "sTheta-A_L",
    "A_R-o_R",
    "A_L-o_L",
    "C_R>G",
    "C_L>G",
    "G>pi",
    "pi>u",
    "g_epistemic>G",
    "g_pragmatic>G",
    "g_pragmatic>tRSI",
    "sTheta>tRSI",
    *(edge for c in CHANNELS for edge in (f"B_{c}>u", f"u>s{c}")),
)
MODEL_COUNTS = {
    "num_channel_factors": 5,
    "num_levels_per_factor": 2,
    "num_reward_levels": 3,
    "num_loss_levels": 3,
    "num_actions": 6,
    "policy_horizon": 1,
}
REQUIRED_SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
    "Signature",
)


def model_parameters(model: EconomicWorldModel) -> dict[str, np.ndarray]:
    """Return named matrices in stable serialization order after validation."""
    _validate_model(model)
    return {
        "A_R": model.A_R,
        "A_L": model.A_L,
        **{f"B_{c}": model.B[c] for c in CHANNELS},
        "C_R": model.C_R,
        "C_L": model.C_L,
        **{f"D_{c}": model.D[c] for c in CHANNELS},
    }


def _tuple(values: object) -> object:
    if isinstance(values, list):
        return tuple(_tuple(v) for v in values)
    return values


def model_to_gnn(model: EconomicWorldModel) -> str:
    """Emit complete factorized coefficients with all standard GNN sections."""
    params = model_parameters(model)
    sections: dict[str, list[str]] = {
        "GNNSection": ["ActInfPOMDP_AlphaFund"],
        "GNNVersionAndFlags": ["GNN v1"],
        "ModelName": ["AlphaFund Economic World Model"],
        "ModelAnnotation": [
            "Authored reduced finite model; not inferred from a codebase or market data.",
            "B axes: next state, current state, action. C contains log preferences.",
        ],
        "StateSpaceBlock": [f"s{c}[2,1,type=float]" for c in CHANNELS],
        "Connections": list(FIXED_CONNECTIONS),
        "InitialParameterization": [],
        "Equations": ["G = -g_pragmatic - g_epistemic"],
        "Time": ["Dynamic", "Discrete", "ModelTimeHorizon=1"],
        "ActInfOntologyAnnotation": [f"s{c}=HiddenStateFactor" for c in CHANNELS],
        "ModelParameters": ["num_channel_factors: 5", "num_actions: 6", "policy_horizon: 1"],
        "Footer": ["AlphaCOGANT finite profile; ExpectedFreeEnergy is minimized."],
        "Signature": ["AlphaCOGANT; illustrative model, not financial advice."],
    }
    for name, array in params.items():
        dims = ",".join(map(str, array.shape))
        sections["StateSpaceBlock"].append(f"{name}[{dims},type=float]")
        literal = repr(_tuple(array.tolist()))
        if array.ndim > 1:
            literal = literal[1:-1]
        sections["InitialParameterization"].append(f"{name}={{{literal}}}")
        concept = {
            "A": "LikelihoodMatrix",
            "B": "TransitionMatrix",
            "C": "LogPreferenceVector",
            "D": "PriorOverHiddenStates",
        }[name[0]]
        sections["ActInfOntologyAnnotation"].append(f"{name}={concept}")
    extras = {
        "o_R": ("3,1", "Observation"),
        "o_L": ("3,1", "Observation"),
        "pi": ("6", "PolicyVector"),
        "u": ("1", "Action"),
        "G": ("6", "ExpectedFreeEnergy"),
        "g_epistemic": ("6", "EpistemicValue"),
        "g_pragmatic": ("6", "PragmaticValue"),
        "tRSI": ("1", "ImprovementSignalToNoiseRatio"),
    }
    for name, (dims, concept) in extras.items():
        sections["StateSpaceBlock"].append(f"{name}[{dims},type=float]")
        sections["ActInfOntologyAnnotation"].append(f"{name}={concept}")
    return "\n\n".join(f"## {name}\n" + "\n".join(lines) for name, lines in sections.items()) + "\n"


def _content_lines(block: str):
    for raw_line in block.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            yield line


def model_from_gnn(text: str) -> EconomicWorldModel:
    """Read the finite profile without repairing coefficients or executing text.

    Missing, duplicate, nonliteral, misdimensioned and nonstochastic parameters
    fail explicitly. The input budget is checked before parsing or allocation.
    """
    if len(text.encode("utf-8")) > MAX_GNN_BYTES:
        raise ValueError(f"GNN source exceeds {MAX_GNN_BYTES} bytes.")
    sections: dict[str, str] = {}
    headers = list(re.finditer(r"^## ([^\n]+)\s*$", text, re.MULTILINE))
    for idx, header in enumerate(headers):
        name = header[1].strip()
        if name in sections:
            raise ValueError(f"Duplicate GNN section {name}.")
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        sections[name] = text[header.end() : end]
    missing = set(REQUIRED_SECTIONS) - sections.keys()
    if missing:
        raise ValueError(f"Missing GNN sections: {sorted(missing)}.")
    if not re.fullmatch(r"GNN v1(?:\.[01])?", sections["GNNVersionAndFlags"].strip()):
        raise ValueError("Unsupported GNNVersionAndFlags; expected GNN v1, v1.0 or v1.1.")
    block = re.sub(r"#[^\n]*", "", sections["InitialParameterization"])
    assignment = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*\{")
    arrays: dict[str, np.ndarray] = {}
    position = 0
    block = block.strip()
    while position < len(block):
        match = assignment.match(block, position)
        if match is None:
            raise ValueError("Unsupported InitialParameterization content.")
        name = match[1]
        if name not in PARAMETER_SHAPES or name in arrays:
            raise ValueError(f"Unexpected or duplicate parameter {name}.")
        end = block.find("}", match.end())
        if end == -1 or "{" in block[match.end() : end]:
            raise ValueError(f"Invalid numeric literal for {name}.")
        try:
            tree = ast.parse("(" + block[match.end() : end].strip() + ")", mode="eval")
            if any(
                isinstance(node, ast.Constant)
                and (isinstance(node.value, bool) or not isinstance(node.value, (int, float)))
                for node in ast.walk(tree)
            ):
                raise ValueError("only real numeric literals are allowed")
            array = np.asarray(ast.literal_eval(tree), dtype=float)
        except (
            SyntaxError, TypeError, ValueError, OverflowError, RecursionError, MemoryError
        ) as exc:
            raise ValueError(f"Invalid numeric literal for {name}.") from exc
        if array.shape != PARAMETER_SHAPES[name]:
            raise ValueError(f"{name} must have shape {PARAMETER_SHAPES[name]}, got {array.shape}.")
        arrays[name] = array
        position = end + 1
    if arrays.keys() != PARAMETER_SHAPES.keys():
        raise ValueError(f"Missing parameters: {sorted(PARAMETER_SHAPES.keys() - arrays.keys())}.")
    declarations = {}
    for line in _content_lines(sections["StateSpaceBlock"]):
        match = re.fullmatch(r"([A-Za-z_]\w*)\[([\d,]+),type=(float|int)\]", line)
        if match is None:
            raise ValueError(f"Unsupported StateSpaceBlock declaration: {line}.")
        name, dimensions, dtype = match.groups()
        if name not in DECLARATION_SHAPES or name in declarations:
            raise ValueError(f"Unexpected or duplicate declaration {name}.")
        shape = tuple(map(int, dimensions.split(",")))
        if shape != DECLARATION_SHAPES[name]:
            raise ValueError(
                f"StateSpaceBlock shape for {name} must be {DECLARATION_SHAPES[name]}."
            )
        if dtype != "float" and name not in {"o_R", "o_L", "u"}:
            raise ValueError(f"StateSpaceBlock type for {name} must be float.")
        declarations[name] = shape
    if declarations.keys() != DECLARATION_SHAPES.keys():
        raise ValueError("StateSpaceBlock must declare every fixed finite-profile variable.")
    connections = set()
    for line in _content_lines(sections["Connections"]):
        if line not in FIXED_CONNECTIONS or line in connections:
            raise ValueError(f"Unsupported or duplicate connection: {line}.")
        connections.add(line)
    if connections != set(FIXED_CONNECTIONS):
        raise ValueError("Connections must contain every fixed finite-profile edge.")
    counts = set()
    for line in _content_lines(sections["ModelParameters"]):
        name, separator, value = line.partition(":")
        name = name.strip()
        if separator and name in MODEL_COUNTS:
            if name in counts or value.strip() != str(MODEL_COUNTS[name]):
                raise ValueError(f"ModelParameters {name} must be {MODEL_COUNTS[name]}.")
            counts.add(name)
    if not {"num_channel_factors", "num_actions", "policy_horizon"}.issubset(counts):
        raise ValueError("ModelParameters must declare factor/action counts and policy_horizon.")
    model = EconomicWorldModel(
        A_R=arrays["A_R"],
        A_L=arrays["A_L"],
        B={c: arrays[f"B_{c}"] for c in CHANNELS},
        C_R=arrays["C_R"],
        C_L=arrays["C_L"],
        D={c: arrays[f"D_{c}"] for c in CHANNELS},
    )
    if "```gnn-matrices" in text:
        _validate_joint_fence(text, model)
    return model


def joint_matrices(model: EconomicWorldModel) -> dict[str, np.ndarray]:
    """Compile the declared factorization into 32 states, 9 observations, 6 actions.

    Lexicographic axes are (I,S,U,Theta,Z) and (reward,loss), last index
    fastest. Dense compilation preserves the declared conditional independence.
    COGANT's controller uses risk plus ambiguity; AlphaCOGANT scores information
    about Theta. Equal kernels do not imply equal policy scores or trajectories.
    """
    _validate_model(model)
    states = list(product(range(2), repeat=len(CHANNELS)))
    observations = list(product(range(3), repeat=2))
    try:
        with np.errstate(over="raise", invalid="raise"):
            C = np.array([model.C_R[r] + model.C_L[loss] for r, loss in observations])
    except FloatingPointError as exc:
        raise ValueError("Joint preference scores must be finite.") from exc
    A = np.empty((9, 32))
    B = np.empty((32, 32, len(ACTIONS)))
    D = np.empty(32)
    for cur, (i, s, u, t, _z) in enumerate(states):
        for obs, (r, loss) in enumerate(observations):
            A[obs, cur] = model.A_R[r, i, u, t] * model.A_L[loss, s, t]
        D[cur] = np.prod([model.D[c][v] for c, v in zip(CHANNELS, states[cur], strict=True)])
        for nxt, next_state in enumerate(states):
            B[nxt, cur] = np.prod(
                [
                    model.B[c][n, v, :]
                    for c, n, v in zip(CHANNELS, next_state, states[cur], strict=True)
                ],
                axis=0,
            )
    return {"A": A, "B": B, "C": C, "D": D}


def model_to_cogant_gnn(model: EconomicWorldModel) -> str:
    """Add COGANT's authoritative, complete ``gnn-matrices`` fence."""
    matrices = joint_matrices(model)
    lines = [
        model_to_gnn(model),
        "## JointMatrixAxes",
        "States: lexicographic (I,S,U,Theta,Z), last factor fastest.",
        "Observations: lexicographic (reward,loss), loss fastest.",
        "Actions: " + ", ".join(ACTIONS),
        "",
        "```gnn-matrices",
        "A[[rows=9][cols=32]]",
    ]
    lines += [" ".join(map(repr, row.tolist())) for row in matrices["A"]]
    lines.append("B[[rows=32][cols=32][depth=6]]")
    for action in range(len(ACTIONS)):
        lines.append(f"# action={action}")
        lines += [" ".join(map(repr, row.tolist())) for row in matrices["B"][:, :, action]]
    for name in ("C", "D"):
        lines.append(f"{name}[[rows={matrices[name].size}]]")
        lines += [repr(float(value)) for value in matrices[name]]
    return "\n".join(lines + ["```", ""])


def _validate_joint_fence(text: str, model: EconomicWorldModel) -> None:
    """Reject a dense extension that contradicts the factorized parameters."""
    fences = re.findall(r"```gnn-matrices[^\n]*\n(.*?)```", text, re.DOTALL)
    if len(fences) != 1 or text.count("```gnn-matrices") != 1:
        raise ValueError("Expected one complete gnn-matrices fence.")
    expected = model_to_cogant_gnn(model).split("```gnn-matrices\n")[1].split("```")[0]
    actual_lines = [line.strip() for line in fences[0].splitlines() if line.strip()]
    expected_lines = [line.strip() for line in expected.splitlines() if line.strip()]
    if len(actual_lines) != len(expected_lines):
        raise ValueError("Joint matrix fence has incomplete or extra rows.")
    for actual, reference in zip(actual_lines, expected_lines, strict=True):
        if reference.startswith(("A[", "B[", "C[", "D[", "#")):
            if actual != reference:
                raise ValueError("Joint matrix fence has incompatible axes or action slices.")
        else:
            try:
                values = [float(token) for token in actual.split()]
            except ValueError as exc:
                raise ValueError("Joint matrix fence requires numeric rows.") from exc
            if values != [float(token) for token in reference.split()]:
                raise ValueError("Joint matrices contradict factorized parameters.")
