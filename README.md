# AlphaCOGANT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20976824.svg)](https://doi.org/10.5281/zenodo.20976824)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Recursive corporate self-improvement as Active Inference, rendered in GNN via COGANT.**

**Version 1.2.1 (2026-10-16).** Standalone NumPy research package with lossless
finite GNN serialization, complete COGANT joint-matrix export, and replayable
observation-conditioned inference, batched action scoring and a versioned agent API. See [CHANGELOG.md](CHANGELOG.md) and the
[pinned upstream contract](docs/upstream_compatibility.md). CI runs lint, the full
test suite with ≥90% coverage, model parity, and installed-wheel checks on
Python 3.10, 3.12, 3.14, and 3.15. The development interpreter defaults to 3.15. The research scope is recorded in [`ISA.md`](ISA.md).

The [AlphaFund whitepaper](https://www.alphafund.com/whitepaper) reframes recursive
self-improvement (RSI) as a portfolio-optimization problem: a corporation recursively
improves when realized economic gains finance the next cycle of better prediction and
deployment, and the firm's standing is summarized by **t-RSI**, the standardized gap
between alpha-creation and alpha-decay rates.

AlphaCOGANT observes that this is, term for term, an **Active Inference** agent — and
makes the correspondence executable:

| AlphaFund | Active Inference |
| --- | --- |
| Economic World Model (EWM) | generative model $P(o,s'\mid s,a)$ |
| Five channels $I,S,U,\Theta,Z$ | hidden-state factors |
| Capital allocation $a_t$ | control / policy |
| Marginal-return objective $J_t$ | **Expected Free Energy** (minimized) |
| Sensors / R&D returns | **epistemic value** (information gain) |
| Investments / Actuators returns | **pragmatic value** (expected log-equity) |
| t-RSI certificate | thresholded EFE-improvement gate |

Its design draws on three related projects:

- **`template_code_project`** — the reproducible code-research scaffold (thin
  orchestrator, `{{TOKEN}}` manuscript, ≥90% coverage, no mocks).
- **GNN** (Generalized Notation Notation) — the text language for Active Inference
  generative models. `models/alphafund_ewm.md` encodes the five-channel EWM.
- **[COGANT](https://github.com/ActiveInferenceInstitute/COGANT)** — codebase-to-GNN
  translation and finite-model execution. AlphaCOGANT uses an authored model,
  maps firm structure → priors, and exports all coefficients in both factorized
  and joint form. See [the bridge contract](docs/cogant_bridge.md).

## Layout

```
models/alphafund_ewm.md       # AlphaFund EWM as a GNN Active Inference model
src/alphacogant/              # NumPy engine (SPEC.md is the contract)
  model/                      # channels, A/B/C/D matrices, operating points
  efe/                        # EFE = epistemic + pragmatic; marginal-return vector
  trsi/                       # create/decay rates, t-RSI, certificate
  bridge/                     # firm structure -> priors -> GNN summary (round-trip)
  stats/                      # trajectories, sensitivity, paired bootstrap profiles
  tokens/                     # every prose {{TOKEN}} is generated here
  viz/                        # shared plot style and color palettes
docs/manuscript/                   # the concept, fully written out (00–09 + refs)
scripts/                      # thin orchestrators (demo, variable generation, figures)
  y_generate_figures.py       # runs every manuscript figure script
  figures/                    # 15 manuscript figures + cover art
    fig_aif_dictionary.py     # AlphaFund ↔ Active Inference dictionary
    fig_certificate_sign_flip.py  # not-green-by-construction comparator
    fig_gnn_factor_graph.py    # EWM-as-GNN factor graph
    fig_self_forecasting_loop.py  # whitepaper Figure 1 schematic
    fig_theta_decay.py         # alpha-decay vs refresh of Theta
    fig_trsi_densities.py      # bootstrap posteriors of create/decay rates
    fig_value_by_regime.py     # EFE decomposition per channel × regime
    fig_belief_trajectory.py   # multi-cycle belief trajectory
    fig_marginal_return_heatmap.py  # value landscape over actions × cycles
    fig_policy_posterior.py    # explore→exploit transition
    fig_trsi_sensitivity.py    # t-RSI vs belief precision and Theta freshness
    fig_regime_comparison.py   # bootstrap CIs + EFE decomposition per regime
    fig_efe_waterfall.py       # EFE decomposition waterfall per action
    fig_create_vs_decay_scatter.py  # bootstrap scatter with break-even line
    fig_break_even_probability.py  # paired break-even probability over Theta freshness
    fig_cover_art.py          # PDF title-page cover art
```

## Quick start

```bash
# From a fresh checkout; Python (the bound in `pyproject.toml`) or newer and uv are required.
uv sync --locked --extra dev
uv run --locked pytest tests/ --cov=alphacogant --cov-fail-under=90 -q
uv run --locked alphacogant validate models/alphafund_ewm.md
uv run --locked alphacogant export --cogant -o output/model.gnn.md
uv run --locked alphacogant infer output/model.gnn.md --steps 8 -o output/trace.json
uv run --locked alphacogant verify-trace output/model.gnn.md output/trace.json
uv run --locked python scripts/run_alphacogant_demo.py
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
```

Exports preserve existing files; use `--overwrite` to replace a generated file.
`infer --observations observations.json` accepts a JSON array of `[reward, loss]`
integer bucket pairs. Without observations it records a synthetic model rollout.
The trace binds the exact source bytes, coefficients, filtering, EFE scores, and
selected actions. The existing `simulate_trajectory` API records open-loop
predictions; t-RSI is a separately evaluated certificate, not an enforced action
gate in either simulator. See [CONTRIBUTING.md](CONTRIBUTING.md) for CI parity and
wheel installation checks.

## Backend and agent workflows

```bash
uv run --locked alphacogant doctor --json
uv run --locked alphacogant capabilities
uv run --locked alphacogant simulate --policy stochastic --seed 7 --horizon 12
uv run --locked alphacogant bootstrap --seed 7 --n 32 --horizon 12
uv run --locked alphacogant profile --samples 5 -o output/profile.json
printf '%s\n' '{"id":"demo","method":"simulation.run","params":{"horizon":3}}' | uv run --locked alphacogant request
```

The [agent API](docs/agent_api.md) shares method admission between Python and
JSONL. `--json` gives CLI computations machine-readable result/error envelopes;
normal argument-usage errors remain on stderr. Outputs publish atomically and
require `--overwrite` to replace an existing file. See
[profiling and performance](docs/performance.md) for reproducible Python 3.15
measurements, profiler limits and numerical compatibility.

## Read the concept

The manuscript writes the idea out in full: the AlphaFund↔Active Inference dictionary
(§2), the GNN-via-COGANT realization (§3), filtration-safe inference (§4), the
epistemic/pragmatic value split and t-RSI as the EFE certificate (§5), and the
functionality + integrity case (§6), numbered formalisms (§8), limitations and
sensitivity analysis (§9).

Implementation and artifact regeneration details live in
[`docs/methods_and_artifacts.md`](docs/methods_and_artifacts.md).

> **Not financial advice.** AlphaCOGANT is a modeling and integrity instrument. The
> GNN model is an illustrative reduced encoding; every numeric in the manuscript is a
> property of that model, generated and gated by the engine — not a claim about
> AlphaFund's proprietary books.
