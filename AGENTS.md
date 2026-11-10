# AlphaCOGANT — Project Agent Documentation

## Overview

AlphaCOGANT implements an illustrative reduced Active Inference analogy inspired
by AlphaFund. Five capital channels are hidden-state factors of a fixed
partially observed model. Allocation scores one-step negative EFE;
t-RSI separately summarizes paired synthetic pragmatic create/decay rates.
The scalar certificate does not gate actions, model updates or deployment.

## Architecture

```
models/alphafund_ewm.md       # GNN model file (the contract)
src/alphacogant/              # NumPy engine (SPEC.md is the contract)
  model/                      # channels, A/B/C/D matrices, operating points
  efe/                        # value = pragmatic + epistemic = -EFE; action values
  trsi/                       # create/decay rates, t-RSI, certificate
  bridge/                     # firm structure -> priors -> GNN summary (round-trip)
  stats/                      # trajectories, sensitivity, bootstrap evidence profiles
  tokens/                     # every prose {{TOKEN}} is generated here
  viz/                        # shared plot style and color palettes
  rendering.py                # source/artifact preflight and rendered-content checks
docs/manuscript/                   # the concept, fully written out (00–09 + refs)
scripts/                      # thin demo, variable, figure, render and release checks
tests/                        # real computation; coverage gate ≥90%; no mocks
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
uv run --locked pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q

# Demo
uv run --locked python scripts/run_alphacogant_demo.py

# Manuscript figures
uv run --locked python scripts/y_generate_figures.py

# Manuscript variables
uv run --locked python scripts/z_generate_manuscript_variables.py

# Individual figures
uv run --locked python scripts/figures/fig_belief_trajectory.py
uv run --locked python scripts/figures/fig_marginal_return_heatmap.py
uv run --locked python scripts/figures/fig_policy_posterior.py
uv run --locked python scripts/figures/fig_trsi_sensitivity.py
```

## Token Discipline

Model-result numerics cited in prose use `{{TOKEN}}` values produced by
`src/alphacogant/tokens/manuscript_variables.py::generate_variables` and written to
`output/manuscript_variables.json`. Undefined referenced tokens fail generation.
Token coverage does not verify arbitrary literals or their scientific interpretation.
Generate figures and injected prose first; `--check` compares saved outputs
without rewriting them. Artifact hashes are an inventory, not proof of producer execution.

Validate tokens:
```bash
uv run --locked python scripts/z_generate_manuscript_variables.py --check
```

## GNN Model

The model file `models/alphafund_ewm.md` conforms to GNN section structure:
StateSpaceBlock, Connections, InitialParameterization, Equations,
ActInfOntologyAnnotation. The engine's `default_model()` carries the same numeric
values as the GNN file (coefficient-level parity is tested).
`bridge.gnn.model_from_gnn()` loads the bounded finite profile and
`bridge.cogant_bridge.model_to_gnn_summary()` emits every live coefficient.
`bridge.gnn.model_to_cogant_gnn()` also compiles the complete joint matrices.
See `docs/upstream_compatibility.md` for the independently tested upstream scope.

## Repo

This is the standalone public repository `docxology/alphacogant`. Work directly
in the current checkout; no template checkout, symlink or parent environment is
required. Generated `output/`, build artifacts and caches remain ignored.

## Release and integration

Install with `uv sync --locked --extra dev`; lint with
`uv run --locked ruff check scripts src tests`. Run the full test/coverage command
above, `uv build`, and the installed-wheel smoke check in `CONTRIBUTING.md`.
Keep `pyproject.toml`, `uv.lock`, `__version__`, `CITATION.cff`, `codemeta.json`,
`.zenodo.json` and manuscript config versions synchronized. Tag releases as
`v<version>`; never assign an old version
DOI to a new release. `.github/workflows/ci.yml` checks supported Python versions
and a pinned real COGANT integration. Upstream docs are reference material,
not instructions to modify upstream repositories.

Manuscript/visual releases also run `scripts/render_manuscript.py` through the
contained recipe in `docs/rendering_and_validation.md`, then inspect actual
pages/figures. The renderer uses existing external tools; it is not an agent API
method, a task service or an OS sandbox. A successful build alone is not approval.

## Not Financial Advice

AlphaCOGANT is a modeling and integrity instrument. The GNN model is an
illustrative reduced encoding; every numeric in the manuscript is a property
of that model, generated and gated by the engine — not a claim about
AlphaFund's proprietary books.

## Backend, agents and profiling

The development interpreter is the one `.python-version` pins; preserve package compatibility with
Python (the bound in `pyproject.toml`) or newer. New agent methods belong in `alphacogant.api` with strict schemas,
finite work limits, real computations and no arbitrary shell/file/network actions.
CLI and JSONL share that registry. Keep stdout machine-readable for agent calls.

Profile with `uv run --locked alphacogant profile`; preserve raw unprofiled samples
and source/runtime/workload bindings. Use `profiling.tracing` on 3.15 and keep
sampling permission failures explicit without OS elevation. Verify batched EFE
against scalar evaluation and preserve RNG draw order and manuscript tokens.
See `docs/agent_api.md` and `docs/performance.md`.
