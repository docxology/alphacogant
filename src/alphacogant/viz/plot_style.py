"""Shared visualization style for AlphaCOGANT figures.

Centralizes color palettes, typography, and layout constants so every figure
script produces a visually consistent manuscript.  Importing this module and
calling ``apply_style()`` sets matplotlib rcParams for publication-quality
output at 200 DPI.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Color palettes ───────────────────────────────────────────────────────────

# Channel colors: pragmatic = warm, epistemic = cool, both = purple
CHANNEL_COLORS: dict[str, str] = {
    "I": "#b91c1c",  # red — pragmatic (Investments)
    "S": "#2563eb",  # blue — epistemic (Sensors)
    "U": "#ea580c",  # orange — pragmatic (Actuators)
    "Theta": "#7c3aed",  # purple — both (Parameters)
    "Z": "#0891b2",  # cyan — epistemic (R&D)
}

# Action colors (same ordering as ACTIONS tuple)
ACTION_COLORS: list[str] = [
    "#b91c1c",  # fund_I
    "#2563eb",  # fund_S
    "#ea580c",  # fund_U
    "#7c3aed",  # fund_Theta
    "#0891b2",  # fund_Z
    "#64748b",  # hold
]

# Semantic colors
CREATE_COLOR: str = "#1d4ed8"  # blue for create-rate
DECAY_COLOR: str = "#dc2626"  # red for decay-rate
EPISTEMIC_COLOR: str = "#7c3aed"  # purple for epistemic value
PRAGMATIC_COLOR: str = "#f59e0b"  # amber for pragmatic value
POSITIVE_COLOR: str = "#16a34a"  # green for positive / admit
NEGATIVE_COLOR: str = "#dc2626"  # red for negative / reject
NEUTRAL_COLOR: str = "#64748b"  # gray for neutral / hold

# Regime colors
IMPROVING_COLOR: str = "#dc2626"  # red — stale, needs improvement
COASTING_COLOR: str = "#16a34a"  # green — fresh, coasting

# Figure quality constants
DPI: int = 200
FIG_DPI: int = 200

# Typography
TITLE_FONTSIZE: int = 12
SUPTITLE_FONTSIZE: int = 13
LABEL_FONTSIZE: int = 10
TICK_FONTSIZE: int = 9
LEGEND_FONTSIZE: int = 8
ANNOTATION_FONTSIZE: int = 8
SMALL_FONTSIZE: int = 7

# Layout
GRID_ALPHA: float = 0.25
LEGEND_FRAMEALPHA: float = 0.95
DEFAULT_FIGSIZE: tuple[float, float] = (8.0, 5.0)
WIDE_FIGSIZE: tuple[float, float] = (11.0, 4.8)
TALL_FIGSIZE: tuple[float, float] = (7.0, 8.0)

ANALYTIC_FOOTER: str = (
    "Analytical/schematic plot from the AlphaCOGANT reduced NumPy engine; "
    "not a simulation and not full PyMDP."
)
REDUCED_SIM_FOOTER: str = (
    "Reduced NumPy Active Inference simulation over AlphaCOGANT beliefs; not full PyMDP."
)
BOOTSTRAP_FOOTER: str = (
    "Dirichlet-bootstrap diagnostic on the reduced NumPy engine; "
    "not empirical data and not full PyMDP."
)
COVER_ART_FOOTER: str = "Cover art schematic; not data, not a simulation, and not full PyMDP."


def apply_style() -> None:
    """Apply publication-quality matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": TICK_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.titlesize": SUPTITLE_FONTSIZE,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )


def styled_figure(figsize: tuple[float, float] | None = None, **kwargs):
    """Create a figure with the shared style applied."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGSIZE, **kwargs)
    return fig, ax


def add_provenance_footer(
    fig: plt.Figure,
    text: str,
    *,
    y: float = 0.01,
    reserve_bottom: float = 0.08,
    color: str = NEUTRAL_COLOR,
) -> None:
    current_bottom = getattr(fig.subplotpars, "bottom", 0.0)
    fig.subplots_adjust(bottom=max(current_bottom, reserve_bottom))
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=SMALL_FONTSIZE,
        color=color,
    )


__all__ = [
    "ACTION_COLORS",
    "ANALYTIC_FOOTER",
    "ANNOTATION_FONTSIZE",
    "BOOTSTRAP_FOOTER",
    "CHANNEL_COLORS",
    "COASTING_COLOR",
    "COVER_ART_FOOTER",
    "CREATE_COLOR",
    "DECAY_COLOR",
    "DPI",
    "DEFAULT_FIGSIZE",
    "EPISTEMIC_COLOR",
    "FIG_DPI",
    "GRID_ALPHA",
    "IMPROVING_COLOR",
    "LABEL_FONTSIZE",
    "LEGEND_FONTSIZE",
    "LEGEND_FRAMEALPHA",
    "NEUTRAL_COLOR",
    "NEGATIVE_COLOR",
    "POSITIVE_COLOR",
    "PRAGMATIC_COLOR",
    "REDUCED_SIM_FOOTER",
    "SMALL_FONTSIZE",
    "SUPTITLE_FONTSIZE",
    "TALL_FIGSIZE",
    "TICK_FONTSIZE",
    "TITLE_FONTSIZE",
    "WIDE_FIGSIZE",
    "add_provenance_footer",
    "apply_style",
    "styled_figure",
]
