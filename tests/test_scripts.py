"""Smoke tests for the demo and variable-generation scripts.

Runs each as a subprocess (no mocks) and verifies the expected output artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
SCRIPT_TIMEOUT_SECONDS = 600


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
        timeout=SCRIPT_TIMEOUT_SECONDS,
    )
    return result


def _load_variable_script_module():
    spec = importlib.util.spec_from_file_location(
        "z_generate_manuscript_variables",
        SCRIPTS_DIR / "z_generate_manuscript_variables.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_demo_produces_json_and_png() -> None:
    result = _run("run_alphacogant_demo.py")
    assert result.returncode == 0, (
        f"Demo failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
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
        f"Variable generation failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    vars_path = PROJECT_ROOT / "output" / "manuscript_variables.json"
    assert vars_path.exists(), f"Expected {vars_path} not found."
    data = json.loads(vars_path.read_text(encoding="utf-8"))

    required_keys = {
        "NUM_CHANNELS",
        "NUM_ACTIONS",
        "PLANNING_HORIZON",
        "BOOTSTRAP_N",
        "HEADLINE_T_RSI",
        "COASTING_T_RSI",
        "CREATE_RATE_MEAN",
        "DECAY_RATE_MEAN",
        "FUNDED_CHANNEL",
        "FUNDED_EPISTEMIC",
        "FUNDED_PRAGMATIC",
        "B_THETA_LEAK_PROB",
        "NUM_DEFINITIONS",
        "NUM_FIGURES",
        "EPISTEMIC_CHANNELS",
        "PRAGMATIC_CHANNELS",
        "BREAK_EVEN_PROB",
        "COASTING_BREAK_EVEN_PROB",
        "BREAK_EVEN_MARGIN_MEAN",
    }
    assert required_keys.issubset(data.keys()), f"Missing keys: {required_keys - data.keys()}"
    assert data["NUM_CHANNELS"] == "5"
    assert data["NUM_ACTIONS"] == "6"
    assert int(data["NUM_DEFINITIONS"]) >= 19
    assert int(data["NUM_FIGURES"]) >= 15
    assert all(isinstance(v, str) and v for v in data.values())

    registry_path = PROJECT_ROOT / "output" / "figures" / "figure_registry.json"
    assert registry_path.exists(), f"Expected {registry_path} not found."
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    labels = {entry["label"] for entry in registry["figures"]}
    assert "fig:trsi" in labels
    assert "fig:sensitivity" in labels
    assert len(labels) == int(data["NUM_FIGURES"])

    manifest_path = PROJECT_ROOT / "output" / "reports" / "artifact_manifest.json"
    assert manifest_path.exists(), f"Expected {manifest_path} not found."
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["entries"]}
    assert "output/figures/figure_registry.json" in manifest_paths
    assert manifest["issues"] == []


def test_artifact_manifest_reports_broken_registered_figures(tmp_path: Path) -> None:
    module = _load_variable_script_module()
    output_dir = tmp_path / "output"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)

    ok_png = figures_dir / "ok.png"
    ok_png.write_bytes(PNG_MAGIC + (b"0" * 128))
    tiny_png = figures_dir / "tiny.png"
    tiny_png.write_bytes(PNG_MAGIC)
    text_png = figures_dir / "text.png"
    text_png.write_text("not a png" * 20, encoding="utf-8")

    scripts_dir = tmp_path / "scripts" / "figures"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "fig_ok.py").write_text("print('ok')\n", encoding="utf-8")

    issues = module.artifact_manifest_issues(
        output_dir,
        [
            {"filename": "ok.png", "generated_by": "scripts/figures/fig_ok.py", "label": "fig:ok"},
            {
                "filename": "missing.png",
                "generated_by": "scripts/figures/fig_missing.py",
                "label": "fig:missing",
            },
            {
                "filename": "tiny.png",
                "generated_by": "scripts/figures/fig_ok.py",
                "label": "fig:tiny",
            },
            {
                "filename": "text.png",
                "generated_by": "scripts/figures/fig_ok.py",
                "label": "fig:text",
            },
            {
                "filename": "ok.png",
                "generated_by": "scripts/figures/fig_ok.py",
                "label": "fig:ok",
            },
        ],
        project_root=tmp_path,
    )

    assert any("registered figure missing: fig:missing" in issue for issue in issues)
    assert any("registered figure producer missing: fig:missing" in issue for issue in issues)
    assert any("registered figure is too small: fig:tiny" in issue for issue in issues)
    assert any("registered figure is not a PNG: fig:text" in issue for issue in issues)
    assert any("duplicate figure label: fig:ok" in issue for issue in issues)
    assert any("duplicate registered figure filename: ok.png" in issue for issue in issues)


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
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Token check failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
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
