"""Unit tests for manuscript build integrity (src/alphacogant/tokens/manuscript_artifacts.py).

The pipeline used to live inline in ``scripts/z_generate_manuscript_variables.py``;
after the 2026-09 thin-orchestrator refactor these functions are importable and tested
directly (real files, real hashing — no mocks).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.image import imsave

from alphacogant.tokens.manuscript_artifacts import (
    PNG_MAGIC,
    _saved_artifact_issues,
    artifact_manifest_issues,
    figure_registry_entries,
    inject,
    read_figure_registry,
    sha256,
    stable_artifact_paths,
    used_tokens,
    write_artifact_manifest,
    write_figure_registry,
)


def _write(directory: Path, name: str, text: str) -> None:
    (directory / name).write_text(text, encoding="utf-8")


def test_used_tokens_scans_all_manuscript_markdown(tmp_path: Path) -> None:
    _write(tmp_path, "00_abstract.md", "Firm with {{NUM_CHANNELS}} channels.")
    _write(tmp_path, "05_value.md", "{{NUM_CHANNELS}} channels, {{NUM_ACTIONS}} actions.")
    _write(tmp_path, "notes.txt", "{{IGNORED}} in a non-markdown file.")
    assert used_tokens(tmp_path) == {"NUM_CHANNELS", "NUM_ACTIONS"}


def test_inject_substitutes_known_tokens_and_keeps_orphans() -> None:
    text = "channels={{NUM_CHANNELS}}, mystery={{NOT_A_TOKEN}}"
    assert inject(text, {"NUM_CHANNELS": "5"}) == "channels=5, mystery={{NOT_A_TOKEN}}"


def test_figure_registry_entries_parse_captions_and_producers(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "04_generative_model_inference.md",
        "![The $B_\\Theta$ decay law](figures/theta_decay.png){#fig:thetadecay}\n",
    )
    entries = figure_registry_entries(tmp_path, {})
    assert entries == [
        {
            "caption": "The $B_\\Theta$ decay law",
            "alt_text": "The $B_\\Theta$ decay law",
            "filename": "theta_decay.png",
            "generated_by": "scripts/figures/fig_theta_decay.py",
            "label": "fig:thetadecay",
            "source": "docs/manuscript/04_generative_model_inference.md",
        }
    ]


def test_figure_registry_write_read_roundtrip(tmp_path: Path) -> None:
    manuscript_dir = tmp_path / "manuscript"
    manuscript_dir.mkdir()
    _write(manuscript_dir, "07.md", "![Cover](figures/cover_art.png){#fig:cover}\n")
    registry_path = write_figure_registry(manuscript_dir, {}, project_root=tmp_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "alphacogant.figure_registry.v1"
    assert read_figure_registry(tmp_path / "output") == payload["figures"]


def test_read_figure_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_figure_registry(tmp_path) == []


def test_sha256_and_stable_artifact_paths(tmp_path: Path) -> None:
    data = tmp_path / "data"
    figures = tmp_path / "figures"
    data.mkdir()
    figures.mkdir()
    (data / "demo_summary.json").write_text("{}", encoding="utf-8")
    cover = figures / "cover.png"
    cover.write_bytes(PNG_MAGIC)
    (tmp_path / "scratch.txt").write_text("not a stable artifact", encoding="utf-8")

    assert stable_artifact_paths(tmp_path) == [data / "demo_summary.json", figures / "cover.png"]
    assert sha256(cover) == hashlib.sha256(PNG_MAGIC).hexdigest()


def test_write_artifact_manifest_hashes_outputs_relative_to_project_root(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    figures = tmp_path / "figures"
    data.mkdir()
    figures.mkdir()
    (data / "manuscript_variables.json").write_text("{}", encoding="utf-8")
    (figures / "cover.png").write_bytes(PNG_MAGIC + b"0" * 128)

    manifest_path = write_artifact_manifest(tmp_path, project_root=tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert {entry["path"] for entry in manifest["entries"]} == {
        "data/manuscript_variables.json",
        "figures/cover.png",
    }
    assert all(entry["contract_match"] for entry in manifest["entries"])
    assert manifest["issues"] == []


@pytest.mark.parametrize("tamper", ["variables", "injected", "figure", "duplicate_manifest"])
def test_saved_artifact_validation_rejects_tampering_without_rewriting(tmp_path, tamper):
    manuscript = tmp_path / "docs/manuscript"
    manuscript.mkdir(parents=True)
    source = manuscript / "00_example.md"
    source.write_text("Mean {{MEAN}}.\n![Data](figures/data.png){#fig:data}\n")
    output = tmp_path / "output"
    figures = output / "figures"
    figures.mkdir(parents=True)
    values = np.linspace(0.0, 1.0, 1024).reshape(32, 32)
    variables = {"MEAN": f"{values.mean():.1f}"}
    imsave(figures / "data.png", values, cmap="viridis")
    producer = tmp_path / "scripts/figures/fig_data.py"
    producer.parent.mkdir(parents=True)
    producer.write_text("# Inventory checks producer presence, not execution provenance.\n")
    tokens = output / "manuscript_variables.json"
    tokens.write_text(json.dumps(variables))
    injected = output / "manuscript/00_example.md"
    injected.parent.mkdir()
    injected.write_text(inject(source.read_text(), variables))
    write_figure_registry(manuscript, variables, project_root=tmp_path)
    manifest_path = write_artifact_manifest(output, project_root=tmp_path)
    assert _saved_artifact_issues(tmp_path, manuscript, variables) == []

    if tamper == "variables":
        tokens.write_text('{"MEAN":"999"}')
    elif tamper == "injected":
        injected.write_text("Mean 999.\n")
    elif tamper == "figure":
        image = figures / "data.png"
        image.write_bytes(image.read_bytes()[:120])
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["entries"].append(manifest["entries"][0])
        manifest_path.write_text(json.dumps(manifest))
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.rglob("*") if path.is_file()
    }
    assert _saved_artifact_issues(tmp_path, manuscript, variables)
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
    } == before


def test_registered_signature_only_png_cannot_pass_decode(tmp_path):
    figures = tmp_path / "output/figures"
    figures.mkdir(parents=True)
    (figures / "forged.png").write_bytes(PNG_MAGIC + b"0" * 128)
    issues = artifact_manifest_issues(
        tmp_path / "output",
        [{"filename": "forged.png", "label": "fig:forged", "generated_by": ""}],
        project_root=tmp_path,
    )
    assert any("cannot be decoded" in issue for issue in issues)
