"""Smoke tests for the demo and variable-generation scripts.

Runs each as a subprocess (no mocks) and verifies the expected output artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run(script_name: str) -> subprocess.CompletedProcess:
    env = {
        "PYTHONPATH": str(SRC),
        "MPLBACKEND": "Agg",
        "PATH": "",
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return result


def test_run_demo_produces_json_and_png() -> None:
    result = _run("run_alphacogant_demo.py")
    assert result.returncode == 0, (
        f"Demo failed (exit {result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    json_path = PROJECT_ROOT / "output" / "data" / "demo_summary.json"
    assert json_path.exists(), f"Expected {json_path} not found."
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Verify the demo JSON has the expected structure.
    assert "marginal_return_vector" in data
    assert "policy_posterior" in data
    assert "efe" in data
    assert "t_rsi" in data
    assert "certificate_delta_2.0" in data
    assert len(data["marginal_return_vector"]) == 6

    png_path = PROJECT_ROOT / "output" / "figures" / "value_decomposition.png"
    assert png_path.exists(), f"Expected {png_path} not found."
    assert png_path.stat().st_size > 100


def test_generate_manuscript_variables_check_mode() -> None:
    result = _run("z_generate_manuscript_variables.py")
    # The script writes variables and injected manuscript in default mode.
    assert result.returncode == 0, (
        f"Variable generation failed (exit {result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    vars_path = PROJECT_ROOT / "output" / "manuscript_variables.json"
    assert vars_path.exists(), f"Expected {vars_path} not found."
    data = json.loads(vars_path.read_text(encoding="utf-8"))

    required_keys = {
        "TOKEN", "NUM_CHANNELS", "NUM_ACTIONS", "PLANNING_HORIZON",
        "HEADLINE_T_RSI", "COASTING_T_RSI", "CREATE_RATE_MEAN", "DECAY_RATE_MEAN",
        "FUNDED_CHANNEL", "FUNDED_EPISTEMIC", "FUNDED_PRAGMATIC",
        "B_THETA_LEAK_PROB", "NUM_DEFINITIONS", "NUM_FIGURES",
        "EPISTEMIC_CHANNELS", "PRAGMATIC_CHANNELS",
    }
    assert required_keys.issubset(data.keys()), f"Missing keys: {required_keys - data.keys()}"
    assert data["TOKEN"] == "AlphaCOGANT"
    assert data["NUM_CHANNELS"] == "5"
    assert data["NUM_ACTIONS"] == "6"
    assert all(isinstance(v, str) and v for v in data.values())


def test_generate_manuscript_variables_no_orphans() -> None:
    """The script's --check mode exits 0 when all prose tokens are generated."""
    env = {
        "PYTHONPATH": str(SRC),
        "MPLBACKEND": "Agg",
        "PATH": "",
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "z_generate_manuscript_variables.py"), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Token check failed (exit {result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "orphans: 0" in result.stdout


def test_manuscript_injection_produces_token_free_output() -> None:
    """After injection, no {{TOKEN}} placeholders remain in output/manuscript/."""
    _run("z_generate_manuscript_variables.py")

    injected_dir = PROJECT_ROOT / "output" / "manuscript"
    assert injected_dir.exists(), "output/manuscript/ not created."
    for md in injected_dir.glob("*.md"):
        content = md.read_text(encoding="utf-8")
        assert "{{" not in content, f"Unresolved {{{{TOKEN}}}} in {md.name}"
