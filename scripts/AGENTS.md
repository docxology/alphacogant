# AGENTS: `scripts/` — Thin Orchestrator Scripts

Technical specification for AlphaCOGANT's analysis scripts.

## Thin-Orchestrator Contract

- Scripts own I/O, path bootstrap, argument parsing and per-figure layout.
  Model, inference, statistics and other scientific computations belong in
  `src/alphacogant/`, where they are reusable and tested.
- Keep reusable visual helpers in `src/alphacogant/viz/`; do not duplicate
  scientific formulas in annotations or import scripts from other scripts.
- Keep bytecode caches, build artifacts and scratch files out of committed
  `scripts/` content. Put generated products under the ignored `output/` tree.

## Script Inventory

| Script | Pattern | Delegates to | Output |
|--------|---------|--------------|--------|
| `run_alphacogant_demo.py` | Thin orchestrator | `alphacogant.model.generative_model`, `alphacogant.efe.free_energy`, `alphacogant.trsi.t_rsi`, `alphacogant.viz.value_decomposition::render_value_decomposition` | `output/data/demo_summary.json`, `output/figures/value_decomposition.png` |
| `y_generate_figures.py` | Thin orchestrator (batch runner) | every `scripts/figures/fig_*.py` | 16 PNGs in `output/figures/` |
| `z_generate_manuscript_variables.py` | Thin orchestrator | `alphacogant.tokens.manuscript_variables`, `alphacogant.tokens.manuscript_artifacts::run` | `output/manuscript_variables.json`, `output/figures/figure_registry.json`, `output/reports/artifact_manifest.json`, injected `output/manuscript/*.md` |

## Figure Scripts (`scripts/figures/`)

Every figure uses `viz.plot_style.apply_style()` and the shared `plot_style.DPI`.
Use provenance appropriate to the plotted computation: ANALYTIC for schematics
and direct evaluations, REDUCED_SIM for prediction-only trajectories, BOOTSTRAP
for synthetic perturbation diagnostics. The graphical abstract, dictionary,
dependency graph and loop include readable in-diagram provenance; other charts
use `add_provenance_footer()`. The graphical abstract is registered in the
manuscript body, not an unregistered decorative cover.

| Script | Delegates to | Footer | Output |
|--------|--------------|--------|--------|
| `fig_aif_dictionary.py` | `model.channels`, `viz.plot_style` | ANALYTIC (in diagram) | `output/figures/aif_dictionary.png` |
| `fig_belief_trajectory.py` | `stats.simulation`, `viz.plot_style` | REDUCED_SIM | `output/figures/belief_trajectory.png` |
| `fig_break_even_probability.py` | `stats.statistics`, `viz.plot_style` | BOOTSTRAP | `output/figures/break_even_probability.png` |
| `fig_certificate_sign_flip.py` | `trsi.t_rsi`, `model.generative_model`, `viz.plot_style` | ANALYTIC | `output/figures/certificate_sign_flip.png` |
| `fig_cover_art.py` | `tokens.manuscript_variables`, `model.operating_points`, `viz.plot_style` | ANALYTIC + BOOTSTRAP (in diagram) | `output/figures/cover_art.png` |
| `fig_create_vs_decay_scatter.py` | `trsi.t_rsi`, `viz.plot_style` | BOOTSTRAP | `output/figures/create_vs_decay_scatter.png` |
| `fig_efe_waterfall.py` | `efe.free_energy`, `viz.plot_style` | ANALYTIC | `output/figures/efe_waterfall.png` |
| `fig_gnn_factor_graph.py` | `model.channels`, `viz.plot_style` | ANALYTIC (in diagram) | `output/figures/gnn_factor_graph.png` |
| `fig_marginal_return_heatmap.py` | `stats.simulation`, `viz.plot_style` | REDUCED_SIM | `output/figures/marginal_return_heatmap.png` |
| `fig_policy_posterior.py` | `efe.free_energy`, `stats.simulation`, `viz.plot_style` | REDUCED_SIM | `output/figures/policy_posterior.png` |
| `fig_regime_comparison.py` | `stats.statistics`, `viz.plot_style` | BOOTSTRAP | `output/figures/regime_comparison.png` |
| `fig_self_forecasting_loop.py` | `efe.free_energy`, `model.generative_model`, `model.operating_points`, `viz.plot_style` | ANALYTIC (in diagram) | `output/figures/self_forecasting_loop.png` |
| `fig_theta_decay.py` | `model.generative_model`, `viz.plot_style` | ANALYTIC | `output/figures/theta_decay.png` |
| `fig_trsi_densities.py` | `trsi.t_rsi`, `model.operating_points`, `viz.plot_style` | BOOTSTRAP | `output/figures/trsi_densities.png` |
| `fig_trsi_sensitivity.py` | `stats.sensitivity`, `viz.plot_style` | BOOTSTRAP | `output/figures/trsi_sensitivity.png` |
| `fig_value_by_regime.py` | `efe.free_energy`, `model.generative_model`, `viz.plot_style` | ANALYTIC | `output/figures/value_by_regime.png` |

## Testing

- `tests/test_scripts.py` — smoke tests: every script path runs as a real subprocess and
  its artifacts are validated; the manifest integrity check is exercised against
  `alphacogant.tokens.manuscript_artifacts.artifact_manifest_issues` directly.
- `tests/test_manuscript_artifacts.py` — unit tests for the moved pipeline logic
  (token scanning/injection, figure registry, artifact manifest hashing).
