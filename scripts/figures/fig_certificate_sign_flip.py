#!/usr/bin/env python3
"""Deterministic figure: the comparator is not green-by-construction.

Thin-orchestrator figure script. All create/decay math comes from
``alphacogant.t_rsi`` (the engine); this script only orchestrates the engine calls
and renders the grouped bar chart.

This plots deterministic POINT-ESTIMATE create/decay rates at three beliefs.
The labels report only the sign of create minus decay. No bootstrap t-RSI,
threshold predicate, candidate comparison or deployment gate is evaluated.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import belief_prior, default_model
from alphacogant.model.operating_points import COASTING_RAW, IMPROVING_RAW
from alphacogant.trsi.t_rsi import DEFAULT_HORIZON, create_rate, decay_rate


def _as_belief(raw: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    return {channel: np.array(raw[channel], dtype=float) for channel in CHANNELS}


def main() -> None:
    from alphacogant.viz.plot_style import (
        ANALYTIC_FOOTER,
        CREATE_COLOR,
        DECAY_COLOR,
        DPI,
        NEGATIVE_COLOR,
        POSITIVE_COLOR,
        add_provenance_footer,
        apply_style,
    )

    apply_style()
    output_dir = PROJECT_ROOT / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "certificate_sign_flip.png"

    model = default_model()
    operating_points = [
        ("prior", belief_prior(model)),
        ("IMPROVING", _as_belief(IMPROVING_RAW)),
        ("COASTING", _as_belief(COASTING_RAW)),
    ]

    labels = [name for name, _ in operating_points]
    creates = [create_rate(model, belief, DEFAULT_HORIZON) for _, belief in operating_points]
    decays = [decay_rate(model, belief, DEFAULT_HORIZON) for _, belief in operating_points]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.bar(
        x - w / 2,
        creates,
        w,
        label="create-rate (greedy advantage over hold)",
        color=CREATE_COLOR,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        decays,
        w,
        label=r"decay-rate (residual $\Theta$ staleness)",
        color=DECAY_COLOR,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )

    # Value labels on bars
    for i in range(len(labels)):
        ax.text(
            x[i] - w / 2,
            creates[i] + 0.01,
            f"{creates[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=CREATE_COLOR,
        )
        ax.text(
            x[i] + w / 2,
            decays[i] + 0.01,
            f"{decays[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=DECAY_COLOR,
        )

    # Raw margins are not standardized certificate or deployment decisions.
    ymax = max(creates + decays) or 1.0
    for i, (create, decay) in enumerate(zip(creates, decays, strict=True)):
        gap = create - decay
        sign = "positive" if gap > 0 else "negative" if gap < 0 else "zero"
        color = POSITIVE_COLOR if gap > 0 else NEGATIVE_COLOR if gap < 0 else "#64748b"
        ax.text(
            x[i],
            ymax * 1.10,
            f"raw {sign} margin\n(create - decay = {gap:+.3f})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.9,
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("pragmatic rate (nats / cycle)")
    ax.set_title(
        "Point-estimate create vs decay (not a certificate decision)\n"
        f"raw margins at three beliefs; horizon={DEFAULT_HORIZON}; no sampling"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=8)
    ax.margins(y=0.30)
    ax.axhline(0.0, color="black", linewidth=1.0)

    fig.tight_layout()
    add_provenance_footer(fig, ANALYTIC_FOOTER)  # point estimates from the engine
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(str(output_path))


if __name__ == "__main__":
    main()
