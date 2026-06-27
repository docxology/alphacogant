# AlphaCOGANT

**Recursive corporate self-improvement as Active Inference, rendered in GNN via COGANT.**

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

It is built from three pieces of the template monorepo:

- **`template_code_project`** — the reproducible code-research scaffold (thin
  orchestrator, `{{TOKEN}}` manuscript, ≥90% coverage, no mocks).
- **GNN** (Generalized Notation Notation) — the text language for Active Inference
  generative models. `models/alphafund_ewm.md` encodes the five-channel EWM.
- **COGANT** — the codebase-to-GNN translation pattern; `src/alphacogant/bridge/cogant_bridge.py`
  maps firm structure → generative-model priors and re-emits a GNN summary (round-trip).

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
manuscript/                   # the concept, fully written out (00–09 + refs)
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
# from the template repo root (after `uv sync`)
cd projects/working/alphacogant
uv run --no-project pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q
uv run --no-project python scripts/run_alphacogant_demo.py
uv run --no-project python scripts/y_generate_figures.py
uv run --no-project python scripts/z_generate_manuscript_variables.py
```

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
