# AlphaCOGANT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20976824.svg)](https://doi.org/10.5281/zenodo.20976824)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An authored reduced Active Inference analogy for corporate self-improvement.**

**Version 1.3.1 (2026-11-10).** Standalone NumPy research package with lossless
finite GNN serialization, complete COGANT joint-matrix export, replayable
observation-conditioned inference, numerically guarded paired diagnostics and a
versioned agent API. See [CHANGELOG.md](CHANGELOG.md) and the
[pinned upstream contract](docs/upstream_compatibility.md). CI exercises Python
3.11–3.15, ≥90% coverage, installed-wheel console/package-data checks and a separate
contained manuscript render/reproducibility job. The development pin is
`.python-version`; historical commissioning scope is recorded in [`ISA.md`](ISA.md).

The historical [AlphaFund whitepaper](https://www.alphafund.com/whitepaper) frames
recursive self-improvement (RSI) as a portfolio-optimization problem: realized
economic gains finance better prediction and deployment. Its corporate assertions
are attributed, not independently reproduced financial evidence; see the
[historical concept inventory](docs/whitepaper_concepts.md).

AlphaCOGANT implements an illustrative, reduced Active Inference analogy —
not an equivalence proof of the firm's economic objective:

| AlphaFund | Active Inference |
| --- | --- |
| Economic World Model (EWM) | generative model $P(o,s'\mid s,a)$ |
| Five channels $I,S,U,\Theta,Z$ | hidden-state factors |
| Capital allocation $a_t$ | control / policy |
| Marginal-return analogy | discrete negative-EFE action scores |
| Sensors / R&D roles | conceptual epistemic roles; implemented information gain targets Theta |
| Investments / Actuators roles | expected reward/loss preference contributions |
| t-RSI diagnostic | separately evaluated paired synthetic mean margin divided by its SE |

Its design draws on three related projects:

- **`template_code_project`** — historical inspiration for source-owned research
  artifacts and thin orchestration, not a runtime or rendering dependency.
- **GNN** (Generalized Notation Notation) — the text language for Active Inference
  generative models. `models/alphafund_ewm.md` encodes the five-channel EWM.
- **[COGANT](https://github.com/ActiveInferenceInstitute/COGANT)** — codebase-to-GNN
  translation and finite-model execution. AlphaCOGANT uses an authored model,
  maps firm structure → priors, and exports all coefficients in both factorized
  and joint form. See [the bridge contract](docs/cogant_bridge.md).

## Find engine, manuscript and figure sources

```
models/alphafund_ewm.md       # AlphaFund EWM as a GNN Active Inference model
src/alphacogant/              # NumPy engine (SPEC.md is the contract)
  model/                      # channels, A/B/C/D matrices, operating points
  efe/                        # minimizes G = -(pragmatic + epistemic); value = -G
  trsi/                       # paired synthetic rates, mean/SE, scalar predicate
  bridge/                     # firm structure -> priors -> GNN summary (round-trip)
  stats/                      # trajectories, sensitivity, paired bootstrap profiles
  tokens/                     # every prose {{TOKEN}} is generated here
  viz/                        # shared plot style and color palettes
  rendering.py                # source/artifact identity and rendered HTML content checks
docs/manuscript/                   # the concept, fully written out (00–09 + refs)
scripts/                      # thin demo, figure, token, render and release-check orchestrators
  y_generate_figures.py       # runs every manuscript figure script
  render_manuscript.py        # fresh contained PDF/HTML bundle; external tools required
  figures/                    # registered manuscript figures, including graphical abstract
    fig_aif_dictionary.py     # AlphaFund ↔ Active Inference dictionary
    fig_certificate_sign_flip.py  # not-green-by-construction comparator
    fig_gnn_factor_graph.py    # EWM-as-GNN factor graph
    fig_self_forecasting_loop.py  # conceptual corporate loop; external handoffs explicit
    fig_theta_decay.py         # alpha-decay vs refresh of Theta
    fig_trsi_densities.py      # paired synthetic rate densities
    fig_value_by_regime.py     # one-step pragmatic and Theta-information scores
    fig_belief_trajectory.py   # prediction-only multi-cycle belief trajectory
    fig_marginal_return_heatmap.py  # value landscape over actions × cycles
    fig_policy_posterior.py    # softmax evaluated along a greedy trajectory, not a switch
    fig_trsi_sensitivity.py    # t-RSI vs Dirichlet concentration and Theta freshness
    fig_regime_comparison.py   # bootstrap CIs + EFE decomposition per regime
    fig_efe_waterfall.py       # EFE decomposition waterfall per action
    fig_create_vs_decay_scatter.py  # bootstrap scatter with break-even line
    fig_break_even_probability.py  # paired break-even probability over Theta freshness
    fig_cover_art.py          # registered landscape graphical abstract: cover_art.png
```

## Install and exercise the standalone engine

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
predictions; t-RSI is a separate paired synthetic diagnostic whose scalar threshold
does not gate actions or updates. Chronological input integrity remains caller-owned.
See [CONTRIBUTING.md](CONTRIBUTING.md) for CI parity and
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

## Choose a reading or regeneration route

- [Manuscript reading order](docs/manuscript/README.md): the conceptual argument,
  authored GNN, factorized filtering, one-step EFE and separate diagnostics.
- [Documentation task hub](docs/README.md): one detailed owner per technical contract.
- [Source-to-output map](docs/methods_and_artifacts.md): trace claims and figures to code.
- [Regeneration and manuscript-release rendering](docs/rendering_and_validation.md):
  serialized generation, source-bound PDF/HTML bundles and actual page/figure inspection.

The abstract ends with a registered landscape graphical abstract, generated as
`cover_art.png` by `fig_cover_art.py`. It distinguishes the observation-conditioned
controller from the prediction-only greedy/hold diagnostic and carries provenance;
it is not decorative title-page art or an extra chapter. The dictionary,
self-forecasting loop and GNN factor graph remain separate explanatory figures.

> **Not financial advice.** AlphaCOGANT is a modeling and integrity instrument. The
> GNN model is illustrative; generated numerics describe that model, not
> AlphaFund's proprietary books. Token coverage is not empirical calibration.
