#!/usr/bin/env python3
"""Deterministic figure: the comparator is not green-by-construction.

Thin-orchestrator figure script. All create/decay math comes from
``alphacogant.t_rsi`` (the engine); this script only orchestrates the engine calls
and renders the grouped bar chart.

The integrity property of t-RSI is that its create-rate and decay-rate are
posteriors over **genuinely different processes**, so the comparator can order
either way across operating points — the certificate can honestly fire OR fail. A
green-by-construction comparator (``decay <= create`` always) could never produce a
regime where ``create < decay``. This figure plots the POINT-ESTIMATE
``create_rate`` and ``decay_rate`` at three operating points and shows the ordering
**flips**:

- ``prior``     — the model's default ``belief_prior`` operating point.
- ``IMPROVING`` — weak channels, stale parameters: the greedy policy funds cheap
  gains, so ``create > decay`` (the certificate would admit at this point estimate).
- ``COASTING``  — strong channels, fresh parameters: the greedy policy coasts on
  ``hold``, so ``create < decay`` (the certificate rejects).

This is exactly what ``tests/test_t_rsi.py::test_comparator_is_not_green_by_construction``
asserts. Consistent with section 5, the honest result is preserved, not softened:
under belief uncertainty the bootstrap t-RSI is negative at both regimes (the reduced
two-level encoding does not robustly certify net improvement); the sign-discrimination
shown here is the point-estimate evidence that the gate is genuinely conditional.
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

from alphacogant.channels import CHANNELS
from alphacogant.generative_model import belief_prior, default_model
from alphacogant.operating_points import IMPROVING_RAW, COASTING_RAW
from alphacogant.t_rsi import create_rate, decay_rate


def _as_belief(raw: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    return {channel: np.array(raw[channel], dtype=float) for channel in CHANNELS}


def main() -> None:
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
    creates = [create_rate(model, belief) for _, belief in operating_points]
    decays = [decay_rate(model, belief) for _, belief in operating_points]
    # The certificate's point-estimate verdict: admit only where create out-paces decay.
    admits = [c > d for c, d in zip(creates, decays)]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(x - w / 2, creates, w, label="create-rate (active policy)", color="#1f77b4", edgecolor="black")
    ax.bar(x + w / 2, decays, w, label="decay-rate (residual $\\Theta$ staleness)", color="#d62728", edgecolor="black")

    ymax = max(creates + decays) or 1.0
    for i, ok in enumerate(admits):
        ax.text(
            x[i], ymax * 1.04,
            "create > decay\nADMIT" if ok else "create < decay\nREJECT",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#2ca02c" if ok else "#d62728",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("pragmatic rate (nats / cycle)")
    ax.set_title("Comparator is not green-by-construction: create vs decay orders oppositely")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.margins(y=0.22)
    ax.axhline(0.0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(str(output_path))


if __name__ == "__main__":
    main()
