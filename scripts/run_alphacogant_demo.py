#!/usr/bin/env python3
"""Thin orchestrator: run the AlphaCOGANT Active Inference demo.

Builds the AlphaFund Economic World Model, infers channel state from a synthetic
high-reward/low-loss observation, computes the Expected-Free-Energy decomposition
and the marginal-return vector across the six allocation actions, and evaluates the
t-RSI certificate. All computation comes from ``src/alphacogant``; this script only
orchestrates and renders outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.efe.free_energy import (  # noqa: E402
    expected_free_energy,
    marginal_return_vector,
    policy_posterior,
)
from alphacogant.model.channels import ACTIONS  # noqa: E402
from alphacogant.model.generative_model import (  # noqa: E402
    belief_prior,
    default_model,
    infer_states,
)
from alphacogant.trsi.t_rsi import bootstrap_t_rsi, certificate  # noqa: E402


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_data = PROJECT_ROOT / "output" / "data"
    out_fig = PROJECT_ROOT / "output" / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    model = default_model()
    prior = belief_prior(model)
    # Synthetic observation: high reward bucket (2), low predictive loss bucket (2).
    posterior = infer_states(model, obs_R=2, obs_L=2, prior=prior)

    g = marginal_return_vector(model, posterior)
    pi = policy_posterior(model, posterior)
    efe = {a: expected_free_energy(model, posterior, a) for a in range(len(ACTIONS))}

    rng = np.random.default_rng(1618033)
    rsi = bootstrap_t_rsi(model, posterior, rng=rng, n=500)
    cert = certificate(rsi["t_rsi"], delta=2.0)

    summary = {
        "marginal_return_vector": {ACTIONS[a]: round(float(v), 4) for a, v in g.items()},
        "policy_posterior": {ACTIONS[a]: round(float(pi[a]), 4) for a in range(len(ACTIONS))},
        "efe": {
            ACTIONS[a]: {
                "pragmatic": round(float(r.pragmatic), 4),
                "epistemic": round(float(r.epistemic), 4),
                "total": round(float(r.total), 4),
            }
            for a, r in efe.items()
        },
        "t_rsi": {k: round(float(v), 4) for k, v in rsi.items()},
        "certificate_delta_2.0": bool(cert),
    }
    data_path = out_data / "demo_summary.json"
    data_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(str(data_path))

    # Figure: epistemic vs pragmatic value per allocation action.
    labels = list(ACTIONS)
    prag = [float(efe[a].pragmatic) for a in range(len(ACTIONS))]
    epis = [float(efe[a].epistemic) for a in range(len(ACTIONS))]
    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, prag, w, label="pragmatic (expected log-equity)")
    ax.bar(x + w / 2, epis, w, label="epistemic (information gain)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value (nats)")
    ax.set_title("AlphaCOGANT: epistemic vs pragmatic value per capital channel")
    ax.legend()
    fig.tight_layout()
    fig_path = out_fig / "value_decomposition.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(str(fig_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
