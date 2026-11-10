"""Real-model render preflight: stale content and mismatched sources cannot pass."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from matplotlib.image import imsave

import alphacogant
from alphacogant.rendering import prepare_render, render_inputs
from alphacogant.tokens.manuscript_artifacts import (
    inject,
    write_artifact_manifest,
    write_figure_registry,
)
from alphacogant.tokens.manuscript_variables import generate_variables


@pytest.fixture
def manuscript_fixture(tmp_path):
    package = Path(alphacogant.__file__).resolve().parent
    checkout = package.parents[1]
    root = tmp_path / "render-project"
    root.mkdir()
    shutil.copytree(package, root / "src/alphacogant",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in ("pyproject.toml", "uv.lock", ".python-version",
                     "models/alphafund_ewm.md", "scripts/render_manuscript.py",
                     "docs/manuscript/config.yaml"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / relative, target)
    manuscript = root / "docs/manuscript"
    (manuscript / "references.bib").write_text("")
    text = (
        "A fixture with {{NUM_ACTIONS}} actions.\n"
        "![Synthetic array](../output/figures/evidence.png){#fig:evidence}\n"
    )
    (manuscript / "00_fixture.md").write_text(text)
    output = root / "output"
    figures = output / "figures"
    figures.mkdir(parents=True)
    imsave(figures / "evidence.png", np.arange(16).reshape(4, 4), cmap="viridis")
    producer = root / "scripts/figures/fig_evidence.py"
    producer.parent.mkdir(parents=True)
    producer.write_text(
        "import numpy as np\nfrom matplotlib.image import imsave\n"
        "imsave('output/figures/evidence.png', np.arange(16).reshape(4, 4), cmap='viridis')\n"
    )
    variables = generate_variables(manuscript_dir=manuscript)
    (output / "manuscript_variables.json").write_text(json.dumps(variables))
    injected = output / "manuscript"
    injected.mkdir()
    (injected / "00_fixture.md").write_text(inject(text, variables))
    write_figure_registry(manuscript, variables, project_root=root)
    write_artifact_manifest(output, project_root=root)
    return root


def test_render_does_not_publish_obsolete_injected_chapters(manuscript_fixture):
    root = manuscript_fixture
    stale = root / "output/manuscript/98_deleted_source.md"
    stale.write_text("This deleted-source chapter must never be published.\n")
    context = prepare_render(root)
    assert context["chapters"] == [root / "output/manuscript/00_fixture.md"]
    assert stale.relative_to(root).as_posix() not in {row["path"] for row in context["inputs"]}
    assert stale.read_text() == "This deleted-source chapter must never be published.\n"


@pytest.mark.parametrize(
    "damage", ["package_version", "same_version_code", "saved_numerics", "paper_version"]
)
def test_render_preflight_refuses_identity_or_model_content_drift(manuscript_fixture, damage):
    root = manuscript_fixture
    if damage == "package_version":
        (root / "pyproject.toml").write_text('[project]\nname="alphacogant"\nversion="0.0.0"\n')
    elif damage == "same_version_code":
        (root / "src/alphacogant/unreviewed.py").write_text(
            "def alternate_coefficient():\n    return 99\n"
        )
    elif damage == "saved_numerics":
        path = root / "output/manuscript_variables.json"
        values = json.loads(path.read_text())
        values["NUM_ACTIONS"] = "999"
        path.write_text(json.dumps(values))
    else:
        path = root / "docs/manuscript/config.yaml"
        import yaml

        config = yaml.safe_load(path.read_text())
        config["paper"]["version"] = "0.0.0"
        path.write_text(yaml.safe_dump(config))
    before = {
        p.relative_to(root): p.read_bytes()
        for p in (root / "output").rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError):
        prepare_render(root)
    after = {
        p.relative_to(root): p.read_bytes()
        for p in (root / "output").rglob("*") if p.is_file()
    }
    assert after == before
    assert not list((root / "output").rglob("render.json"))


def test_render_preflight_reports_missing_source_without_generating_replacements(
    manuscript_fixture,
):
    root = manuscript_fixture
    source = root / "models/alphafund_ewm.md"
    source.unlink()
    with pytest.raises(ValueError):
        render_inputs(root)
    assert not source.exists()
