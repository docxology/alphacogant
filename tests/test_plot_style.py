"""Smoke test for the plot_style module.

The module is imported by figure scripts (which run as subprocesses, so
coverage is not tracked). This test verifies the module imports cleanly
and exposes the expected constants and functions.
"""

from __future__ import annotations


def test_plot_style_imports():
    from alphacogant import plot_style

    assert hasattr(plot_style, "apply_style")
    assert hasattr(plot_style, "CHANNEL_COLORS")
    assert hasattr(plot_style, "ACTION_COLORS")
    assert hasattr(plot_style, "CREATE_COLOR")
    assert hasattr(plot_style, "DECAY_COLOR")


def test_plot_style_colors():
    from alphacogant.viz.plot_style import ACTION_COLORS, CHANNEL_COLORS

    assert len(CHANNEL_COLORS) == 5
    assert len(ACTION_COLORS) == 6
    for color in CHANNEL_COLORS.values():
        assert isinstance(color, str)
        assert color.startswith("#")


def test_apply_style_does_not_raise():
    from alphacogant.viz.plot_style import apply_style

    apply_style()


def test_styled_figure_produces_real_png(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from alphacogant.viz.plot_style import styled_figure

    fig, ax = styled_figure(figsize=(4, 3))
    ax.plot([0, 1, 2], [0, 1, 0])
    out = tmp_path / "styled.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_add_provenance_footer_renders_text(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from alphacogant.viz.plot_style import add_provenance_footer, styled_figure

    fig, ax = styled_figure(figsize=(4, 3))
    add_provenance_footer(fig, "test provenance", reserve_bottom=0.2)
    assert fig.subplotpars.bottom >= 0.2
    texts = [t.get_text() for t in fig.texts]
    assert "test provenance" in texts
    out = tmp_path / "footer.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_styled_figure_defaults_and_kwargs(tmp_path):
    """Exercise the default-figsize and **kwargs branches of styled_figure."""
    import matplotlib

    matplotlib.use("Agg")
    from alphacogant.viz.plot_style import DEFAULT_FIGSIZE, styled_figure

    fig, ax = styled_figure()  # figsize=None -> DEFAULT_FIGSIZE branch
    assert fig.get_size_inches()[0] == DEFAULT_FIGSIZE[0]
    fig2, ax2 = styled_figure(figsize=(5, 2), dpi=72)
    assert fig2.get_size_inches()[1] == 2
    out = tmp_path / "kwargs.png"
    fig2.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_add_provenance_footer_defaults(tmp_path):
    """Exercise the default y/reserve_bottom/color branch of add_provenance_footer."""
    import matplotlib

    matplotlib.use("Agg")
    from alphacogant.viz.plot_style import (
        NEUTRAL_COLOR,
        add_provenance_footer,
        styled_figure,
    )

    fig, ax = styled_figure(figsize=(4, 3))
    add_provenance_footer(fig, "default footer")
    footer = [t for t in fig.texts if t.get_text() == "default footer"][0]
    assert footer.get_color() == NEUTRAL_COLOR
    assert fig.subplotpars.bottom >= 0.08
