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
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

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

    print(f"generated {len(FIGURE_SCRIPTS)} figure scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
