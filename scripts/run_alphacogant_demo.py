#!/usr/bin/env python3
"""Thin orchestrator: run the AlphaCOGANT Active Inference demo.

Builds the AlphaFund Economic World Model, infers channel state from a synthetic
high-reward/low-loss observation, computes the Expected-Free-Energy decomposition
and the marginal-return vector across the six allocation actions, and evaluates the
t-RSI certificate. All computation comes from ``src/alphacogant``; this script only
orchestrates and delegates output rendering.
"""

from __future__ import annotations

import argparse
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AlphaCOGANT demo pipeline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory for the demo summary JSON and value-decomposition figure.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")

    from alphacogant.viz import render_value_decomposition  # noqa: E402

    out_data = args.output_dir / "data"
    out_fig = args.output_dir / "figures"
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

    # Figure: epistemic vs pragmatic value per allocation action (rendered by viz).
    fig_path = render_value_decomposition(efe, out_fig / "value_decomposition.png")
    print(str(fig_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
