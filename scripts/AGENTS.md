# AGENTS: `scripts/` — Thin Orchestrator Scripts

Technical specification for AlphaCOGANT's analysis scripts.

## Thin-Orchestrator Contract

- Scripts do orchestration only: path bootstrap, argparse, logging, and a single
  delegated call into a `src/alphacogant/` entrypoint.
- Business, data, plot, and analysis logic lives in `src/alphacogant/` and must be
  importable and covered by `tests/` (no logic may exist only inside a script).
- Scripts must not import each other; shared logic is promoted into `src/alphacogant/`.
- Hygiene: no bytecode caches, build artifacts, or scratch files under `scripts/`
  (`__pycache__/` and `*.pyc` are gitignored; tests import `src` modules, never script
  files, so no bytecode is generated here).

## Script Inventory

| Script | Pattern | Delegates to | Output |
|--------|---------|--------------|--------|
| `run_alphacogant_demo.py` | Thin orchestrator | `alphacogant.model.generative_model`, `alphacogant.efe.free_energy`, `alphacogant.trsi.t_rsi` | `output/data/demo_summary.json`, `output/figures/value_decomposition.png` |
| `y_generate_figures.py` | Thin orchestrator (batch runner) | every `scripts/figures/fig_*.py` | 16 PNGs in `output/figures/` |
| `z_generate_manuscript_variables.py` | Thin orchestrator | `alphacogant.tokens.manuscript_variables`, `alphacogant.tokens.manuscript_artifacts::run` | `output/manuscript_variables.json`, `output/figures/figure_registry.json`, `output/reports/artifact_manifest.json`, injected `output/manuscript/*.md` |

## Figure Scripts (`scripts/figures/`)

| Script | Delegates to | Output |
|--------|--------------|--------|
| `fig_aif_dictionary.py` | `efe.free_energy`, `model.generative_model` | `output/figures/aif_dictionary.png` |
| `fig_belief_trajectory.py` | `stats.simulation` | `output/figures/belief_trajectory.png` |
| `fig_break_even_probability.py` | `stats.statistics`, `viz.plot_style` | `output/figures/break_even_probability.png` |
| `fig_certificate_sign_flip.py` | `trsi.t_rsi`, `model.generative_model` | `output/figures/certificate_sign_flip.png` |
| `fig_cover_art.py` | `viz.plot_style` | `output/figures/cover_art.png` |
| `fig_create_vs_decay_scatter.py` | `trsi.t_rsi`, `viz.plot_style` | `output/figures/create_vs_decay_scatter.png` |
| `fig_efe_waterfall.py` | `efe.free_energy`, `viz.plot_style` | `output/figures/efe_waterfall.png` |
| `fig_gnn_factor_graph.py` | `efe.free_energy`, `model.generative_model` | `output/figures/gnn_factor_graph.png` |
| `fig_marginal_return_heatmap.py` | `stats.simulation` | `output/figures/marginal_return_heatmap.png` |
| `fig_policy_posterior.py` | `efe.free_energy`, `stats.simulation` | `output/figures/policy_posterior.png` |
| `fig_regime_comparison.py` | `stats.statistics`, `viz.plot_style` | `output/figures/regime_comparison.png` |
| `fig_self_forecasting_loop.py` | `efe.free_energy`, `model.generative_model` | `output/figures/self_forecasting_loop.png` |
| `fig_theta_decay.py` | `model.generative_model` | `output/figures/theta_decay.png` |
| `fig_trsi_densities.py` | `trsi.t_rsi`, `model.operating_points` | `output/figures/trsi_densities.png` |
| `fig_trsi_sensitivity.py` | `stats.sensitivity`, `viz.plot_style` | `output/figures/trsi_sensitivity.png` |
| `fig_value_by_regime.py` | `efe.free_energy`, `model.generative_model` | `output/figures/value_by_regime.png` |

## Testing

- `tests/test_scripts.py` — smoke tests: every script path runs as a real subprocess and
  its artifacts are validated; the manifest integrity check is exercised against
  `alphacogant.tokens.manuscript_artifacts.artifact_manifest_issues` directly.
- `tests/test_manuscript_artifacts.py` — unit tests for the moved pipeline logic
  (token scanning/injection, figure registry, artifact manifest hashing).
