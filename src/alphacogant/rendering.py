"""Content checks for trusted local manuscript producers; no process execution."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import tomllib
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import yaml

from alphacogant import __version__
from alphacogant.io import read_text
from alphacogant.profiling import source_fingerprint
from alphacogant.tokens.manuscript_artifacts import (
    figure_registry_entries,
    saved_artifact_issues,
    sha256,
    stable_artifact_paths,
    used_tokens,
)
from alphacogant.tokens.manuscript_variables import generate_variables


def _symlink_below(root: Path, path: Path) -> bool:
    return any(p.is_symlink() for p in (path, *path.parents)
               if p != root and p.is_relative_to(root))


def _regular(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    if ".." in relative.parts or _symlink_below(root, path):
        raise ValueError(f"Render inputs must not traverse symlinks: {relative}")
    if not path.is_file():
        raise ValueError(f"Required render input is not a regular file: {relative}")


def render_destination(root: Path, destination: Path) -> Path:
    """Refuse occupied bundles, lexical escapes and symlink traversal below root."""
    path = destination if destination.is_absolute() else root / destination
    try:
        relative = path.relative_to(root / "output")
    except ValueError as exc:
        raise ValueError(
            "Render destination must be a fresh directory below project output/."
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Render destination must be a fresh directory below project output/.")
    if _symlink_below(root, path):
        raise ValueError("Render destination must not traverse symlinks.")
    if path.exists():
        raise FileExistsError(f"Render destination is already occupied: {path.relative_to(root)}")
    return path


def render_inputs(root: Path) -> list[dict]:
    """Bind consumed sources/artifacts, not unrelated old injected chapters."""
    manuscript = root / "docs/manuscript"
    paths = {
        root / "pyproject.toml", root / "uv.lock", root / ".python-version",
        root / "models/alphafund_ewm.md", root / "src/alphacogant/SPEC.md",
        manuscript / "config.yaml", manuscript / "references.bib",
        root / "output/reports/artifact_manifest.json", root / "scripts/render_manuscript.py",
        *manuscript.glob("*.md"), *(root / "src/alphacogant").rglob("*.py"),
        *stable_artifact_paths(root / "output"),
        *(root / "scripts/figures").glob("*.py"),
    }
    paths.update(root / "output/manuscript" / source.name for source in manuscript.glob("*.md"))
    entries = []
    for path in sorted(paths):
        _regular(root, path)
        entries.append({"path": path.relative_to(root).as_posix(),
                        "sha256": sha256(path), "size_bytes": path.stat().st_size})
    return entries


def prepare_render(root: Path) -> dict:
    """Check version/source identity and real saved consistency before rendering.

    A matching source checkout is required. Disk hashes do not reload previously
    imported modules; use the producer in a fresh process. Consistency does not
    certify empirical calibration or the execution of historical producers.
    """
    inputs = render_inputs(root)
    authority = tomllib.loads(read_text(root / "pyproject.toml"))["project"]
    if authority["name"] != "alphacogant" or authority["version"] != __version__:
        raise ValueError("Render checkout and running package versions must match.")
    engine = source_fingerprint()
    if source_fingerprint(root=root / "src/alphacogant") != engine:
        raise ValueError("Render checkout and running package Python sources must match.")
    manuscript = root / "docs/manuscript"
    variables = generate_variables(manuscript_dir=manuscript)
    issues = saved_artifact_issues(root, manuscript, variables)
    if used_tokens(manuscript) - variables.keys() or issues:
        raise ValueError(f"Saved manuscript artifacts are stale or unavailable: {issues}")
    try:
        config = yaml.safe_load(read_text(manuscript / "config.yaml"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid manuscript configuration: {exc}") from exc
    if config["paper"]["version"] != __version__:
        raise ValueError("Manuscript configuration and package versions must match.")
    # Select current authored chapters, never orphan output left by deleted sources.
    chapters = [root / "output/manuscript" / source.name
                for source in sorted(manuscript.glob("[0-9][0-9]_*.md"))]
    if not chapters:
        raise ValueError("No numbered manuscript chapters are available.")
    return {"inputs": inputs, "engine_source_sha256": engine, "config": config,
            "chapters": chapters, "figures": figure_registry_entries(manuscript, variables)}


class _HTMLSurface(HTMLParser):
    def __init__(self, labels: set[str]):
        super().__init__(convert_charrefs=True)
        self.ids = Counter()
        self.anchors = []
        self.images = []
        self.mathml = 0
        self.labels = labels
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids[attributes["id"]] += 1
        label = attributes.get("id")
        if label not in self.labels:
            label = self.stack[-1][1] if self.stack else None
        href = attributes.get("href") or ""
        if tag == "a" and href.startswith("#") and len(href) > 1:
            self.anchors.append(unquote(href[1:]))
        if tag == "img":
            self.images.append((label, attributes))
        if tag == "math":
            self.mathml += 1
        if tag not in {
            "area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr",
        }:
            self.stack.append((tag, label))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break


def html_metrics(path: Path, figures: list[dict], root: Path) -> dict:
    """Validate image bytes/accessibility and internal navigation in real HTML."""
    expected = {f["label"]: sha256(root / "output/figures" / f["filename"]) for f in figures}
    if len(expected) != len(figures):
        raise ValueError("Expected figure labels must be unique.")
    surface = _HTMLSurface(set(expected))
    surface.feed(path.read_text(encoding="utf-8"))
    if any(count != 1 for count in surface.ids.values()):
        raise ValueError("Rendered HTML contains duplicate anchor IDs.")
    missing = set(surface.anchors) - surface.ids.keys()
    if missing or any(figure["label"] not in surface.ids for figure in figures):
        raise ValueError(f"Rendered HTML has missing anchors/figure labels: {sorted(missing)}")
    observed = {}
    for label, image in surface.images:
        if label is None or label in observed:
            raise ValueError("Each rendered image must belong to one unique registered figure.")
        source = image.get("src") or ""
        if not (image.get("alt") or "").strip() or not source.startswith("data:image/png;base64,"):
            raise ValueError("Rendered figures require nonempty alt text and embedded PNG data.")
        try:
            data = base64.b64decode(source.split(",", 1)[1], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Rendered HTML contains malformed embedded image data.") from exc
        observed[label] = hashlib.sha256(data).hexdigest()
    if observed != expected:
        raise ValueError("Rendered HTML figure labels do not match their registered PNG bytes.")
    return {"registered_images": len(observed), "mathml_elements": surface.mathml,
            "missing_anchors": [], "duplicate_ids": []}


def check_render_diagnostics(text: str) -> None:
    """Reject emitted unresolved-reference/citation and missing-glyph diagnostics."""
    failure = re.compile(
        r"\b(?:citation|reference)s?\b[^\n]*(?:not found|undefined)"
        r"|undefined (?:citation|reference)s?\b|missing character:",
        re.IGNORECASE,
    )
    issues = [line for line in text.splitlines() if failure.search(line)]
    if issues:
        raise ValueError("Unresolved manuscript diagnostics:\n" + "\n".join(issues))
