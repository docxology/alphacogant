"""Cross-format release identity and installed distribution checks."""

from __future__ import annotations

import json
import re
from importlib.metadata import version
from pathlib import Path

import yaml

import alphacogant

ROOT = Path(__file__).resolve().parents[1]


def test_release_identity_is_consistent():
    expected = alphacogant.__version__
    project = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', project, re.MULTILINE)[1] == expected
    assert version("alphacogant") == expected
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["version"] == expected
    assert json.loads((ROOT / "codemeta.json").read_text())["version"] == expected
    lock = (ROOT / "uv.lock").read_text()
    assert f'name = "alphacogant"\nversion = "{expected}"' in lock
    assert f"## {expected}" in (ROOT / "CHANGELOG.md").read_text()
    assert citation["doi"] == "10.5281/zenodo.20976824"
    assert not any(
        item.get("description", "").startswith("Version 1.0.1") for item in citation["identifiers"]
    )
