"""Cross-format release identity and installed distribution checks."""

from __future__ import annotations

import json
import tomllib
from importlib.metadata import version
from pathlib import Path

import yaml

import alphacogant

ROOT = Path(__file__).resolve().parents[1]


def test_release_identity_is_consistent():
    expected = alphacogant.__version__
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["version"] == expected
    assert version("alphacogant") == expected
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["version"] == expected
    assert json.loads((ROOT / "codemeta.json").read_text())["version"] == expected
    assert json.loads((ROOT / ".zenodo.json").read_text())["version"] == expected
    manuscript = yaml.safe_load((ROOT / "docs/manuscript/config.yaml").read_text())
    assert manuscript["paper"]["version"] == expected
    assert manuscript["paper"]["date"] == citation["date-released"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    package = next(row for row in lock["package"] if row["name"] == "alphacogant")
    assert package["version"] == expected
    assert f"## {expected}" in (ROOT / "CHANGELOG.md").read_text()
    assert citation["doi"] == "10.5281/zenodo.20976824"
