#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = sorted((PROJECT_ROOT / "scripts" / "figures").glob("fig_*.py"))


def main() -> int:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    for script in FIGURE_SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            print(f"{script.name} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
        # Each figure script self-bootstraps sys.path, so no PYTHONPATH injection
        # is needed here; each must still actually write its PNG.
        expected_png = (
            PROJECT_ROOT / "output" / "figures" / f"{script.stem.removeprefix('fig_')}.png"
        )
        if not expected_png.exists():
            print(
                f"{script.name} exited 0 but did not write {expected_png}",
                file=sys.stderr,
            )
            return 1

    print(f"generated {len(FIGURE_SCRIPTS)} figures into output/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
