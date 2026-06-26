"""Smoke tests for figure scripts: each script runs and produces a PNG.

Uses subprocess (no mocks) to exercise the real figure scripts and verifies
the expected output file exists and is a valid PNG after each run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
FIGURES_DIR = PROJECT_ROOT / "scripts" / "figures"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"

FIGURE_SCRIPTS = sorted(FIGURES_DIR.glob("fig_*.py"))


def _run_script(script_path: Path) -> None:
    env = {
        "PYTHONPATH": str(SRC),
        "MPLBACKEND": "Agg",
        "PATH": "",
    }
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script_path.name} failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("script", FIGURE_SCRIPTS, ids=lambda s: s.stem)
def test_figure_script_produces_png(script: Path) -> None:
    """Each figure script runs and produces a non-empty PNG file."""
    _run_script(script)
    # The script prints the output path on success.
    png_name = script.stem.replace("fig_", "") + ".png"
    png_path = OUTPUT_DIR / png_name
    assert png_path.exists(), f"Expected output {png_path} not found."
    assert png_path.stat().st_size > 100, f"PNG {png_path} is suspiciously small."
    # Verify PNG magic bytes.
    with open(png_path, "rb") as f:
        magic = f.read(8)
    assert magic == b"\x89PNG\r\n\x1a\n", f"{png_path} is not a valid PNG."


def test_all_figure_scripts_discovered() -> None:
    """Ensure we are actually testing a non-empty set of scripts."""
    assert len(FIGURE_SCRIPTS) >= 10, f"Expected >=10 figure scripts, found {len(FIGURE_SCRIPTS)}"
