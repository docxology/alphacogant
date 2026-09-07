#!/usr/bin/env python3
"""Thin orchestrator: generate manuscript {{TOKEN}} values and inject them.

All business logic lives in ``src/alphacogant/``: token generation in
``alphacogant.tokens.manuscript_variables`` and the token/injection/registry/manifest
pipeline in ``alphacogant.tokens.manuscript_artifacts`` (single delegated call:
``run``). This script only parses ``--check`` and delegates; it produces
``output/manuscript_variables.json``, ``output/figures/figure_registry.json``,
``output/reports/artifact_manifest.json``, and (unless ``--check``) token-substituted
copies under ``output/manuscript/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.tokens.manuscript_artifacts import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate token coverage; do not write injected manuscript.",
    )
    args = parser.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
