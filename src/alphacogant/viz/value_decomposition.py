"""Value-decomposition figure renderer for the demo pipeline.

Plot logic promoted out of ``scripts/run_alphacogant_demo.py`` to honor the
thin-orchestrator contract: the script only orchestrates engine calls and
delegates rendering here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from alphacogant.model.channels import ACTIONS
from alphacogant.viz.plot_style import (
    ANALYTIC_FOOTER,
    DPI,
    EPISTEMIC_COLOR,
    PRAGMATIC_COLOR,
    add_provenance_footer,
    apply_style,
)

__all__ = ["render_value_decomposition"]


def render_value_decomposition(
    efe_by_action: Mapping[int, Any], output_path: Path | str
) -> Path:
    """Render the epistemic vs pragmatic value bar chart for each action.

    ``efe_by_action`` maps action index -> result object exposing ``pragmatic``
    and ``epistemic`` floats (as returned by
    :func:`alphacogant.efe.free_energy.expected_free_energy`). The figure is
    written to ``output_path`` (parent directories are created) and its path
    is returned.
    """
    apply_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(ACTIONS)
    pragmatic = [float(efe_by_action[a].pragmatic) for a in range(len(ACTIONS))]
    epistemic = [float(efe_by_action[a].epistemic) for a in range(len(ACTIONS))]
    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        x - width / 2,
        pragmatic,
        width,
        color=PRAGMATIC_COLOR,
        label="pragmatic (expected log-equity)",
    )
    ax.bar(
        x + width / 2,
        epistemic,
        width,
        color=EPISTEMIC_COLOR,
        label="epistemic (information gain)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value (nats)")
    ax.set_title("AlphaCOGANT: epistemic vs pragmatic value per capital channel")
    ax.legend()
    add_provenance_footer(fig, ANALYTIC_FOOTER)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    return output_path
