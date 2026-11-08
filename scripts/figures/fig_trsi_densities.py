#!/usr/bin/env python3
"""Figure: t-RSI create/decay sample densities at the IMPROVING operating point.

The canonical paired sampler supplies the create/decay arrays. Histograms and
KDE overlays show marginal draw distributions; t-RSI standardizes the mean
PAIRED create-minus-decay margin by its Monte Carlo standard error.
These synthetic perturbations are not calibrated corporate observations.

Run (from the repository root):
    PYTHONPATH=src MPLBACKEND=Agg python scripts/figures/fig_trsi_densities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic raster backend

import matplotlib.pyplot as plt
import numpy as np

# Engine import (NumPy-only business logic lives under src/alphacogant).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from alphacogant.model.generative_model import default_model  # noqa: E402
from alphacogant.model.operating_points import (  # noqa: E402
    BOOTSTRAP_CONCENTRATION as CONCENTRATION,
)
from alphacogant.model.operating_points import (
    BOOTSTRAP_N as N_SAMPLES,
)
from alphacogant.model.operating_points import (
    BOOTSTRAP_SEED as SEED,
)
from alphacogant.model.operating_points import (
    IMPROVING,
)
from alphacogant.trsi.t_rsi import (  # noqa: E402
    DEFAULT_HORIZON,
    _sample_statistics,
    _stable_mean,
    paired_bootstrap_samples,
    t_rsi,
)
from alphacogant.viz.plot_style import (  # noqa: E402
    BOOTSTRAP_FOOTER,
    CREATE_COLOR,
    DECAY_COLOR,
    DPI,
    add_provenance_footer,
    apply_style,
)

# Match the canonical manuscript bootstrap recipe without resampling it twice.
OUTPUT_PATH = _PROJECT_ROOT / "output" / "figures" / "trsi_densities.png"


def _gaussian_kde(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Deterministic Silverman-bandwidth Gaussian KDE evaluated on ``grid``."""
    n = samples.size
    std = float(np.std(samples, ddof=1))
    if std == 0.0:
        # Degenerate (all-equal) sample: return a narrow spike on the grid.
        density = np.zeros_like(grid)
        density[np.argmin(np.abs(grid - samples[0]))] = 1.0
        return density
    bandwidth = 1.06 * std * n ** (-1.0 / 5.0)
    diffs = (grid[:, None] - samples[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs**2) / np.sqrt(2.0 * np.pi)
    return kernel.sum(axis=1) / (n * bandwidth)


def main() -> Path:
    """Generate the t-RSI create/decay density figure and return its path."""
    apply_style()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = default_model()
    rng = np.random.default_rng(SEED)

    create_samples, decay_samples = paired_bootstrap_samples(
        model,
        IMPROVING,
        rng,
        n=N_SAMPLES,
        horizon=DEFAULT_HORIZON,
        concentration=CONCENTRATION,
    )

    create_mean = _stable_mean(create_samples)
    decay_mean = _stable_mean(decay_samples)
    margin_mean, _, paired_se = _sample_statistics(create_samples - decay_samples)
    value = t_rsi(create_samples, decay_samples)

    # Shared grid spanning both sample distributions.
    lo = float(min(create_samples.min(), decay_samples.min()))
    hi = float(max(create_samples.max(), decay_samples.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    grid = np.linspace(lo - pad, hi + pad, 512)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    bins = np.linspace(lo - pad, hi + pad, 40)
    ax.hist(
        create_samples,
        bins=bins,
        density=True,
        color=CREATE_COLOR,
        alpha=0.32,
        label="create-rate samples",
    )
    ax.hist(
        decay_samples,
        bins=bins,
        density=True,
        color=DECAY_COLOR,
        alpha=0.32,
        label="decay-rate samples",
    )

    ax.plot(grid, _gaussian_kde(create_samples, grid), color=CREATE_COLOR, lw=2.0)
    ax.plot(grid, _gaussian_kde(decay_samples, grid), color=DECAY_COLOR, lw=2.0)

    ax.axvline(
        create_mean,
        color=CREATE_COLOR,
        ls="--",
        lw=1.6,
        label=f"create mean = {create_mean:.4f}",
    )
    ax.axvline(
        decay_mean,
        color=DECAY_COLOR,
        ls="--",
        lw=1.6,
        label=f"decay mean = {decay_mean:.4f}",
    )

    annotation = (
        f"t-RSI = mean(create - decay) / paired SE = {value:.3f}\n"
        f"mean paired margin = {margin_mean:+.4f}\n"
        f"paired margin SE = {paired_se:.4f}\n"
        f"n={N_SAMPLES}; seed={SEED}; horizon={DEFAULT_HORIZON}\n"
        f"concentration={CONCENTRATION:g}"
    )
    ax.text(
        0.36,
        0.78,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.9, "pad": 6.0},
    )

    ax.set_xlabel("path-integral rate (nats / cycle)")
    ax.set_ylabel("density")
    ax.set_title(
        f"t-RSI create/decay densities at the IMPROVING operating point (t-RSI = {value:.3f})"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    add_provenance_footer(fig, BOOTSTRAP_FOOTER)  # Dirichlet bootstrap samples

    fig.savefig(OUTPUT_PATH, dpi=DPI)
    plt.close(fig)

    print(str(OUTPUT_PATH.resolve()))
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
