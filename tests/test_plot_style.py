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
