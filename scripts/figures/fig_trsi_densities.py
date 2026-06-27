#!/usr/bin/env python3
"""Figure: t-RSI create/decay sample densities at the IMPROVING operating point.

Mirrors the AlphaFund whitepaper Figures 2/9 in the AlphaCOGANT engine. At the
self-improving (IMPROVING) operating belief, this thin orchestrator draws the
create-rate and decay-rate sample distributions from a deterministic Dirichlet
bootstrap (``Dirichlet(belief * concentration + 1)`` perturbations, fixed rng) by
calling the engine's ``create_rate`` / ``decay_rate`` over each perturbed belief,
then plots the two sample distributions as histograms with KDE overlays
(create = blue, decay = red), draws vertical lines at the two means, and annotates
the standardized t-RSI distance and pooled standard error.

All numeric machinery lives in ``src/alphacogant`` (the generative model, the
free-energy / marginal-return engine, and the t-RSI rates + bootstrap). This script
only orchestrates the sampling and renders the figure — no business logic, no mocks,
fully deterministic via a fixed ``numpy.random.default_rng`` seed.

The headline t-RSI here is *modestly negative* (the honest-negative result the
manuscript's Section 5 preserves): once belief uncertainty is propagated through the
bootstrap, the residual decay rate sits above the create rate at this coarse
two-level operating point. The figure reports that honestly rather than tuning the
number favorable.

Run:
    cd projects/working/alphacogant && \
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

from alphacogant.model.channels import CHANNELS  # noqa: E402
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
    bootstrap_t_rsi,
    create_rate,
    decay_rate,
    t_rsi,
)

# Deterministic figure parameters — IDENTICAL to the canonical bootstrap that
# manuscript_variables.generate_variables() uses for the HEADLINE_T_RSI token, so the
# number displayed here is byte-identical to the manuscript's headline (asserted in
# main()). Do not diverge these from manuscript_variables._BOOTSTRAP_{SEED,N}.
OUTPUT_PATH = _PROJECT_ROOT / "output" / "figures" / "trsi_densities.png"

CREATE_COLOR = "#1f77b4"  # blue
DECAY_COLOR = "#d62728"  # red


def _bootstrap_samples(
    model,
    belief: dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    horizon: int,
    concentration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw create/decay rate samples over Dirichlet belief perturbations.

    Mirrors ``t_rsi.bootstrap_t_rsi`` perturbation scheme so the returned per-sample
    arrays (needed for the density plot) are consistent with the engine's bootstrap
    summary, while the standardized distance itself is computed by ``t_rsi.t_rsi``.
    """
    create_samples = np.empty(n, dtype=float)
    decay_samples = np.empty(n, dtype=float)
    for index in range(n):
        perturbed: dict[str, np.ndarray] = {}
        for channel in CHANNELS:
            alpha = np.clip(belief[channel], 1e-9, None) * concentration + 1.0
            perturbed[channel] = rng.dirichlet(alpha)
        create_samples[index] = create_rate(model, perturbed, horizon)
        decay_samples[index] = decay_rate(model, perturbed, horizon)
    return create_samples, decay_samples


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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = default_model()
    rng = np.random.default_rng(SEED)

    create_samples, decay_samples = _bootstrap_samples(
        model,
        IMPROVING,
        rng,
        n=N_SAMPLES,
        horizon=DEFAULT_HORIZON,
        concentration=CONCENTRATION,
    )

    create_mean = float(np.mean(create_samples))
    decay_mean = float(np.mean(decay_samples))
    create_se = float(np.std(create_samples, ddof=1) / np.sqrt(create_samples.size))
    decay_se = float(np.std(decay_samples, ddof=1) / np.sqrt(decay_samples.size))
    pooled_se = float(np.sqrt(create_se**2 + decay_se**2))
    value = t_rsi(create_samples, decay_samples)

    # Self-check: this figure's draw MUST equal the engine's canonical bootstrap
    # (the same call that produces the HEADLINE_T_RSI manuscript token). Fail loudly
    # on any drift so the figure can never silently contradict the manuscript.
    canonical = bootstrap_t_rsi(
        model,
        IMPROVING,
        np.random.default_rng(SEED),
        n=N_SAMPLES,
        horizon=DEFAULT_HORIZON,
        concentration=CONCENTRATION,
    )
    assert abs(value - canonical["t_rsi"]) < 1e-9, (value, canonical["t_rsi"])
    assert abs(create_mean - canonical["create_mean"]) < 1e-9
    assert abs(decay_mean - canonical["decay_mean"]) < 1e-9

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
        f"standardized t-RSI distance = {value:.3f}\n"
        f"pooled SE = {pooled_se:.4f}\n"
        f"create SE = {create_se:.4f}   decay SE = {decay_se:.4f}\n"
        f"n = {N_SAMPLES}   horizon = {DEFAULT_HORIZON}   "
        f"concentration = {CONCENTRATION:g}"
    )
    ax.text(
        0.02,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.9, "pad": 6.0},
    )

    ax.set_xlabel("path-integral rate (horizon-mean pragmatic value, nats)")
    ax.set_ylabel("density")
    ax.set_title(
        f"t-RSI create/decay densities at the IMPROVING operating point (t-RSI = {value:.3f})"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print(str(OUTPUT_PATH.resolve()))
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
