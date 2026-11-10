"""Consumer-visible render containment, embedded-figure and navigation boundaries."""

import base64
from pathlib import Path

import numpy as np
import pytest
from matplotlib.image import imsave

from alphacogant.rendering import _regular, html_metrics, render_destination


@pytest.fixture
def rendered_fixture(tmp_path):
    root = tmp_path / "project"
    figures = root / "output/figures"
    figures.mkdir(parents=True)
    png = figures / "evidence.png"
    imsave(png, np.arange(16).reshape(4, 4), cmap="viridis")
    uri = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    registry = [{"filename": png.name, "label": "fig:evidence"}]
    html = root / "output/fixture.html"
    return root, html, registry, uri


def _document(
    uri, *, label="fig:evidence", href="#fig%3Aevidence", alt="Computed values", extra=""
):
    return (
        '<!doctype html><html><body>'
        f'<a href="{href}">Evidence</a><figure id="{label}">'
        f'<img src="{uri}" alt="{alt}"></figure>{extra}'
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>'
        '</body></html>'
    )


def test_rendered_html_binds_real_png_bytes_and_decoded_fragment_links(rendered_fixture):
    root, html, registry, uri = rendered_fixture
    html.write_text(_document(uri))
    metrics = html_metrics(html, registry, root)
    assert metrics == {
        "registered_images": 1, "mathml_elements": 1,
        "missing_anchors": [], "duplicate_ids": [],
    }
    # A different, still-valid PNG must not be silently substituted in the HTML.
    other = root / "output/other.png"
    imsave(other, np.arange(16, 32).reshape(4, 4), cmap="magma")
    altered = "data:image/png;base64," + base64.b64encode(other.read_bytes()).decode()
    html.write_text(_document(altered))
    with pytest.raises(ValueError):
        html_metrics(html, registry, root)


def test_rendered_html_cannot_swap_registered_figures_while_preserving_all_hashes(rendered_fixture):
    root, html, registry, first_uri = rendered_fixture
    second = root / "output/figures/other.png"
    imsave(second, np.arange(16).reshape(4, 4), cmap="magma")
    second_uri = "data:image/png;base64," + base64.b64encode(second.read_bytes()).decode()
    registry.append({"filename": second.name, "label": "fig:other"})

    def document(left, right):
        return (
            '<html><body><figure id="fig:evidence">'
            f'<div><img src="{left}" alt="First data"></div></figure>'
            f'<figure id="fig:other"><img src="{right}" alt="Second data"></figure>'
            '<a href="#fig:evidence">First</a><a href="#fig:other">Second</a>'
            '</body></html>'
        )

    html.write_text(document(first_uri, second_uri))
    assert html_metrics(html, registry, root)["registered_images"] == 2
    # All image bytes, IDs, captions and links remain; only associations change.
    html.write_text(document(second_uri, first_uri))
    with pytest.raises(ValueError):
        html_metrics(html, registry, root)


@pytest.mark.parametrize("damage", [
    "missing_alt", "external_image", "invalid_base64", "missing_anchor",
    "missing_figure", "duplicate_anchor", "duplicate_image",
])
def test_rendered_html_rejects_accessibility_identity_and_navigation_damage(
    rendered_fixture, damage
):
    root, html, registry, uri = rendered_fixture
    options = {}
    if damage == "missing_alt":
        options["alt"] = " "
    elif damage == "external_image":
        uri = "https://example.invalid/evidence.png"
    elif damage == "invalid_base64":
        uri = "data:image/png;base64,not!base64"
    elif damage == "missing_anchor":
        options["href"] = "#absent"
    elif damage == "missing_figure":
        options.update(label="other", href="#other")
    elif damage == "duplicate_anchor":
        options["extra"] = '<div id="fig:evidence">duplicate</div>'
    else:
        options["extra"] = f'<img src="{uri}" alt="Unexpected duplicate">'
    html.write_text(_document(uri, **options))
    observed = html.read_bytes()
    with pytest.raises(ValueError):
        html_metrics(html, registry, root)
    assert html.read_bytes() == observed


@pytest.mark.parametrize("escape", ["../protected", "output/../protected", "output"])
def test_render_destination_cannot_escape_output_or_occupy_its_root(tmp_path, escape):
    root = tmp_path / "project"
    root.mkdir()
    protected = tmp_path / "protected"
    protected.write_text("Preserve original bytes.")
    with pytest.raises(ValueError):
        render_destination(root, Path(escape))
    assert protected.read_text() == "Preserve original bytes."
    assert not (root / "output").exists()


def test_render_refuses_occupied_bundle_without_replacing_prior_evidence(tmp_path):
    root = tmp_path / "project"
    destination = root / "output/release"
    destination.mkdir(parents=True)
    receipt = destination / "render.json"
    receipt.write_text('{"prior":"retained"}')
    before = receipt.stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        render_destination(root, destination)
    assert receipt.read_text() == '{"prior":"retained"}'
    assert receipt.stat().st_mtime_ns == before


@pytest.mark.parametrize("ancestor", [False, True])
def test_render_rejects_symlink_destination_components_without_external_writes(tmp_path, ancestor):
    root = tmp_path / "project"
    output = root / "output"
    output.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "preserved"
    sentinel.write_text("Do not replace.")
    (output / "linked").symlink_to(outside, target_is_directory=True)
    destination = output / "linked" / "new" if ancestor else output / "linked"
    with pytest.raises(ValueError):
        render_destination(root, destination)
    assert sorted(p.name for p in outside.iterdir()) == [sentinel.name]
    assert sentinel.read_text() == "Do not replace."


def test_render_input_rejects_symlink_even_when_target_is_regular(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    target = tmp_path / "private.txt"
    target.write_text("Not a project input.")
    link = root / "input.md"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        _regular(root, link)
    assert target.read_text() == "Not a project input."
