#!/usr/bin/env python3
"""Refuse a tag build whose version, annotation or main ancestry is wrong."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant import __version__  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    args = parser.parse_args()
    receipt = {"version": __version__, "ref": args.ref, "tag_checked": False}
    if args.ref.startswith("refs/tags/"):
        expected = f"refs/tags/v{__version__}"
        if args.ref != expected:
            raise ValueError(f"Release tag must be {expected}, not {args.ref}.")
        environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

        def git(*arguments):
            result = subprocess.run(
                ["git", *arguments], cwd=PROJECT_ROOT, env=environment,
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20,
            )
            if result.returncode:
                raise ValueError(f"Release Git identity check failed: {result.stderr.strip()}")
            return result.stdout.strip()

        if git("cat-file", "-t", expected) != "tag":
            raise ValueError("Release ref must be an annotated tag, not a lightweight ref.")
        head = git("rev-parse", "HEAD")
        if git("rev-parse", expected + "^{commit}") != head:
            raise ValueError("Release tag and checked-out source commit differ.")
        git("merge-base", "--is-ancestor", head, "refs/remotes/origin/main")
        receipt.update(tag_checked=True, commit=head, tag_object=git("rev-parse", expected))
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
