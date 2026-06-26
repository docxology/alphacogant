#!/usr/bin/env python3
"""Thin orchestrator: generate manuscript {{TOKEN}} values and inject them.

Business logic lives in ``src/alphacogant/manuscript_variables.py``. This script
only handles I/O: it writes ``output/manuscript_variables.json``, checks that every
``{{TOKEN}}`` used in ``manuscript/*.md`` is produced by the generator (no orphans),
and (unless ``--check``) writes token-substituted copies under
``output/manuscript/``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.manuscript_variables import generate_variables  # noqa: E402

TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def used_tokens(manuscript_dir: Path) -> set[str]:
    """Every {{TOKEN}} referenced across the manuscript markdown."""
    tokens: set[str] = set()
    for md in sorted(manuscript_dir.glob("*.md")):
        tokens.update(TOKEN_RE.findall(md.read_text(encoding="utf-8")))
    return tokens


def inject(text: str, variables: dict[str, str]) -> str:
    return TOKEN_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate token coverage; do not write injected manuscript.",
    )
    args = parser.parse_args()

    manuscript_dir = PROJECT_ROOT / "manuscript"
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = generate_variables()

    used = used_tokens(manuscript_dir)
    missing = sorted(used - set(variables))
    if missing:
        print(f"ORPHAN TOKENS (used in prose, not generated): {missing}", file=sys.stderr)
        return 1

    vars_path = output_dir / "manuscript_variables.json"
    vars_path.write_text(json.dumps(variables, indent=2, sort_keys=True), encoding="utf-8")
    print(str(vars_path))
    print(f"tokens generated: {len(variables)} | tokens used in prose: {len(used)} | orphans: 0")

    if args.check:
        return 0

    injected_dir = output_dir / "manuscript"
    injected_dir.mkdir(parents=True, exist_ok=True)
    for md in sorted(manuscript_dir.glob("*.md")):
        out = injected_dir / md.name
        out.write_text(inject(md.read_text(encoding="utf-8"), variables), encoding="utf-8")
        print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
