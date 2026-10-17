"""Bounded UTF-8/JSON reads and complete-before-publish text output."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

MAX_SOURCE_BYTES = 1_000_000
MAX_JSON_BYTES = 32_000_000


def read_text(path: Path, *, limit: int = MAX_SOURCE_BYTES) -> str:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Input exceeds {limit} bytes.")
    return data.decode("utf-8")


def parse_json(text: str):
    """Reject duplicate keys and nonstandard NaN/Infinity literals."""

    depth = 0
    quoted = False
    escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("JSON nesting exceeds 64 levels.")
        elif char in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}.")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON literal: {value}.")

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("JSON number exceeds finite floating-point range.")
        return value

    try:
        return json.loads(
            text, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep.") from exc


def write_text(text: str, output: Path, *, overwrite: bool = False) -> None:
    """Publish a complete same-directory temporary with exclusive or replace semantics."""
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never follow an output symlink during overwrite. Replacing it could surprise callers.
    if output.is_symlink():
        raise ValueError("Output must not be a symlink.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=".alphacogant-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text if text.endswith("\n") else text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            if output.exists():
                temporary.chmod(output.stat().st_mode & 0o777)
            os.replace(temporary, output)
        else:
            os.link(temporary, output)  # atomic no-clobber publication
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
