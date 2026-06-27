#!/usr/bin/env python3
"""Thin orchestrator: generate manuscript {{TOKEN}} values and inject them.

Business logic lives in ``src/alphacogant/tokens/manuscript_variables.py``. This
script only handles I/O: it writes ``output/manuscript_variables.json``, checks
that every ``{{TOKEN}}`` used in ``manuscript/*.md`` is produced by the generator
(no orphans), records figure/artifact contracts, and (unless ``--check``) writes
token-substituted copies under ``output/manuscript/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.tokens.manuscript_variables import generate_variables  # noqa: E402

TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
FIGURE_REF_RE = re.compile(
    r"!\[(?P<caption>.*?)\]\((?P<path>[^)]+)\)\{#(?P<label>fig:[A-Za-z0-9_]+)\}",
    re.DOTALL,
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MIN_FIGURE_BYTES = 100


def used_tokens(manuscript_dir: Path) -> set[str]:
    """Every {{TOKEN}} referenced across the manuscript markdown."""
    tokens: set[str] = set()
    for md in sorted(manuscript_dir.glob("*.md")):
        tokens.update(TOKEN_RE.findall(md.read_text(encoding="utf-8")))
    return tokens


def inject(text: str, variables: dict[str, str]) -> str:
    return TOKEN_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), text)


def figure_registry_entries(
    manuscript_dir: Path, variables: dict[str, str]
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for md in sorted(manuscript_dir.glob("*.md")):
        text = inject(md.read_text(encoding="utf-8"), variables)
        for match in FIGURE_REF_RE.finditer(text):
            filename = Path(match.group("path")).name
            # The producer is always the per-figure script; recording it
            # unconditionally (no always-present fallback) lets the manifest's
            # producer-missing check actually fire when a figure has no generator.
            script = PROJECT_ROOT / "scripts" / "figures" / f"fig_{Path(filename).stem}.py"
            generated_by = script.relative_to(PROJECT_ROOT).as_posix()
            entries.append(
                {
                    "caption": " ".join(match.group("caption").split()),
                    "filename": filename,
                    "generated_by": generated_by,
                    "label": match.group("label"),
                    "source": f"manuscript/{md.name}",
                }
            )
    return entries


def write_figure_registry(manuscript_dir: Path, variables: dict[str, str]) -> Path:
    registry_dir = PROJECT_ROOT / "output" / "figures"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "figure_registry.json"
    payload = {
        "figures": figure_registry_entries(manuscript_dir, variables),
        "schema_version": "alphacogant.figure_registry.v1",
    }
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return registry_path


def artifact_manifest_issues(
    output_dir: Path,
    figure_entries: list[dict[str, str]],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    issues: list[str] = []
    seen_labels: set[str] = set()
    seen_filenames: set[str] = set()
    figures_dir = output_dir / "figures"

    for entry in figure_entries:
        label = entry["label"]
        filename = entry["filename"]
        if label in seen_labels:
            issues.append(f"duplicate figure label: {label}")
        seen_labels.add(label)
        if filename in seen_filenames:
            issues.append(f"duplicate registered figure filename: {filename}")
        seen_filenames.add(filename)

        figure_path = figures_dir / filename
        relative_figure = figure_path.relative_to(project_root).as_posix()
        if not figure_path.exists():
            issues.append(f"registered figure missing: {label} -> {relative_figure}")
        elif figure_path.stat().st_size <= MIN_FIGURE_BYTES:
            issues.append(f"registered figure is too small: {label} -> {relative_figure}")
        elif figure_path.suffix.lower() == ".png":
            with figure_path.open("rb") as handle:
                if handle.read(len(PNG_MAGIC)) != PNG_MAGIC:
                    issues.append(f"registered figure is not a PNG: {label} -> {relative_figure}")

        producer = entry.get("generated_by", "")
        if producer and not (project_root / producer).exists():
            issues.append(f"registered figure producer missing: {label} -> {producer}")

    return issues


def read_figure_registry(output_dir: Path) -> list[dict[str, str]]:
    registry_path = output_dir / "figures" / "figure_registry.json"
    if not registry_path.exists():
        return []
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return list(payload.get("figures", []))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_artifact_paths(output_dir: Path) -> list[Path]:
    patterns = [
        "data/*",
        "figures/*",
        "manuscript_variables.json",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in output_dir.glob(pattern) if path.is_file())
    return sorted(paths)


def write_artifact_manifest(output_dir: Path) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "artifact_manifest.json"
    # Honor SOURCE_DATE_EPOCH for byte-reproducible manifests; else wall-clock UTC.
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    timestamp = (
        datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc).isoformat()
        if source_date_epoch
        else datetime.now(timezone.utc).isoformat()
    )
    entries = [
        {
            "contract_match": path.stat().st_size > 0,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "stage_name": "Project Analysis",
            "stage_num": 4,
            "timestamp": timestamp,
        }
        for path in stable_artifact_paths(output_dir)
    ]
    issues = artifact_manifest_issues(output_dir, read_figure_registry(output_dir))
    manifest_path.write_text(
        json.dumps({"entries": entries, "issues": issues}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


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
    print(
        f"tokens generated: {len(variables)} | tokens used in prose: {len(used)} "
        f"| orphans: {len(missing)}"
    )
    print(str(write_figure_registry(manuscript_dir, variables)))
    print(str(write_artifact_manifest(output_dir)))

    # Artifact-integrity contract (manuscript §4): in verification mode the registered
    # figures, their producers, and the PNG surface must all be present and consistent.
    figure_entries = figure_registry_entries(manuscript_dir, variables)
    artifact_issues = artifact_manifest_issues(output_dir, figure_entries)
    if artifact_issues:
        print("ARTIFACT INTEGRITY ISSUES:", file=sys.stderr)
        for issue in artifact_issues:
            print(f"  - {issue}", file=sys.stderr)
        if args.check:
            return 1

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
