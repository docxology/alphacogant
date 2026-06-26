# AlphaCOGANT — Project Agent Documentation

## Overview

AlphaCOGANT renders the AlphaFund self-improving corporation as an Active
Inference agent. The firm's five capital channels (Investments, Sensors,
Actuators, Parameters, R&D) are hidden-state factors of a partially-observed
generative model; capital allocation is the control vector; Expected Free
Energy is the portfolio optimizer's marginal-return objective; t-RSI is the
thresholded EFE-improvement certificate.

## Architecture

```
models/alphafund_ewm.md       # GNN model file (the contract)
src/alphacogant/              # NumPy engine (SPEC.md is the contract)
  channels.py                 # 5 channels + 6 allocation actions
  generative_model.py         # A/B/C/D matrices, state inference
  free_energy.py              # EFE = epistemic + pragmatic; marginal-return vector
  t_rsi.py                    # create/decay rates, t-RSI, certificate
  cogant_bridge.py            # firm structure -> priors -> GNN summary (round-trip)
  operating_points.py         # canonical IMPROVING/COASTING beliefs + bootstrap constants
  manuscript_variables.py     # every prose {{TOKEN}} is generated here
  simulation.py              # multi-cycle trajectory recorder
  sensitivity.py             # t-RSI sweep over belief precision and Theta freshness
manuscript/                   # the concept, fully written out (00–09 + refs)
scripts/                      # thin orchestrators (demo, variables, figures)
tests/                        # 158 tests, 99.75% coverage, no mocks
```

## Key Invariants

- **Active Inference sign convention**: EFE `G` is minimized; value `= -G`.
  `EFEResult.total == -(pragmatic + epistemic)`.
- **Epistemic value is never negative** (KL ≥ 0).
- **All randomness via explicit `numpy.random.Generator`**; no global seed;
  same seed → identical output (determinism enforced by tests).
- **No mocks**: all tests use real data and real computation.
- **Thin orchestrator pattern**: business logic in `src/alphacogant/`, scripts
  only orchestrate I/O and rendering.

## Quick Commands

```bash
# Tests
uv run --no-project pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q

# Demo
uv run --no-project python scripts/run_alphacogant_demo.py

# Manuscript variables
uv run --no-project python scripts/z_generate_manuscript_variables.py

# Individual figures
uv run --no-project python scripts/figures/fig_belief_trajectory.py
uv run --no-project python scripts/figures/fig_marginal_return_heatmap.py
uv run --no-project python scripts/figures/fig_policy_posterior.py
uv run --no-project python scripts/figures/fig_trsi_sensitivity.py
```

## Token Discipline

Every numeric cited in prose is a `{{TOKEN}}` produced by
`src/alphacogant/manuscript_variables.py::generate_variables` and written to
`output/manuscript_variables.json` by `scripts/z_generate_manuscript_variables.py`.
A token used in prose but absent from the generator is a build failure (no orphans).

Validate tokens:
```bash
uv run --no-project python scripts/z_generate_manuscript_variables.py --check
```

## GNN Model

The model file `models/alphafund_ewm.md` conforms to GNN section structure:
StateSpaceBlock, Connections, InitialParameterization, Equations,
ActInfOntologyAnnotation. The engine's `default_model()` loads the same numeric
values as the GNN file, and `cogant_bridge.model_to_gnn_summary()` re-emits a
GNN-style block from the live arrays (round-trip).

## Private Repo

This project lives in the private GitHub repo `docxology/alphacogant` and is
symlinked into the template at `projects/working/alphacogant`. It is
local-only and must never be committed to the public template repo.

## Not Financial Advice

AlphaCOGANT is a modeling and integrity instrument. The GNN model is an
illustrative reduced encoding; every numeric in the manuscript is a property
of that model, generated and gated by the engine — not a claim about
AlphaFund's proprietary books.
