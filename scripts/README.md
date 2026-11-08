# Generate demo, figure and manuscript artifacts

Top-level scripts coordinate package computations; figure scripts additionally
own layout and annotations. Scientific model computations live in
`src/alphacogant/`, not in manuscript literals. This map identifies producers;
[rendering and validation](../docs/rendering_and_validation.md) owns the ordered
regeneration recipe and [figures and artifacts](../docs/figures_and_artifacts.md)
owns registry/manifest semantics.

## Run the demo or generate manuscript products

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `run_alphacogant_demo.py` | Demo: model, filtering, EFE/action values and separate t-RSI predicate example; writes a non-manuscript comparison figure | `alphacogant.model.generative_model`, `alphacogant.efe.free_energy`, `alphacogant.trsi.t_rsi`, `alphacogant.viz.value_decomposition` | `uv run --locked python scripts/run_alphacogant_demo.py` |
| `y_generate_figures.py` | Batch runner: executes every `scripts/figures/fig_*.py` deterministically (Agg backend), verifies each PNG lands, stops on first failure | all `scripts/figures/fig_*.py` | `uv run --locked python scripts/y_generate_figures.py` |
| `z_generate_manuscript_variables.py` | Manuscript build gate: generates {{TOKEN}}s, fails on orphan tokens, writes figure registry + artifact manifest, and (default mode) writes token-substituted chapters | `alphacogant.tokens.manuscript_variables`, `alphacogant.tokens.manuscript_artifacts` | `uv run --locked python scripts/z_generate_manuscript_variables.py [--check]` |

## Locate registered manuscript figure producers

Each script renders one deterministic PNG into `output/figures/`, using engine
functions/constants for displayed scientific quantities. Provenance appears inside
each figure: analytical schematics/direct evaluations use ANALYTIC, prediction-only
trajectories REDUCED_SIM, and synthetic perturbation diagnostics BOOTSTRAP.
The graphical abstract, dictionary, factor graph and corporate loop use legible
in-diagram provenance rather than relying on a shared tiny footer.
`cover_art.png` is registered from the abstract chapter, not a no-footer cover exception.

| Script | Purpose | Delegates to | Provenance mode | Command |
|--------|---------|--------------|--------|---------|
| `fig_aif_dictionary.py` | Channel/state/dependency table and decision-term key | `model.channels`, `viz.plot_style` | ANALYTIC | `uv run --locked python scripts/figures/fig_aif_dictionary.py` |
| `fig_belief_trajectory.py` | Multi-cycle belief trajectory under the greedy policy | `stats.simulation`, `viz.plot_style` | REDUCED_SIM | `uv run --locked python scripts/figures/fig_belief_trajectory.py` |
| `fig_break_even_probability.py` | Break-even probability profile | `stats.statistics`, `viz.plot_style` | BOOTSTRAP | `uv run --locked python scripts/figures/fig_break_even_probability.py` |
| `fig_certificate_sign_flip.py` | Point-belief create-minus-decay comparator; not a deployment gate | `trsi.t_rsi`, `model.generative_model`, `viz.plot_style` | ANALYTIC | `uv run --locked python scripts/figures/fig_certificate_sign_flip.py` |
| `fig_cover_art.py` | Registered landscape graphical abstract: controller and separate paired synthetic diagnostic, with scope limits | `tokens.manuscript_variables`, `model.operating_points`, `viz.plot_style` | ANALYTIC + BOOTSTRAP | `uv run --locked python scripts/figures/fig_cover_art.py` |
| `fig_create_vs_decay_scatter.py` | Create-rate vs decay-rate scatter across bootstrap perturbations | `trsi.t_rsi`, `viz.plot_style` | BOOTSTRAP | `uv run --locked python scripts/figures/fig_create_vs_decay_scatter.py` |
| `fig_efe_waterfall.py` | EFE decomposition per discrete action at IMPROVING; minimize G | `efe.free_energy`, `viz.plot_style` | ANALYTIC | `uv run --locked python scripts/figures/fig_efe_waterfall.py` |
| `fig_gnn_factor_graph.py` | Fixed likelihood/transition dependency graph | `model.channels`, `viz.plot_style` | ANALYTIC | `uv run --locked python scripts/figures/fig_gnn_factor_graph.py` |
| `fig_marginal_return_heatmap.py` | Marginal-return vector (negative EFE) heatmap over actions × cycles | `stats.simulation`, `viz.plot_style` | REDUCED_SIM | `uv run --locked python scripts/figures/fig_marginal_return_heatmap.py` |
| `fig_policy_posterior.py` | Policy-posterior evolution across the greedy trajectory | `efe.free_energy`, `stats.simulation`, `viz.plot_style` | REDUCED_SIM | `uv run --locked python scripts/figures/fig_policy_posterior.py` |
| `fig_regime_comparison.py` | Regime comparison with bootstrap confidence intervals | `stats.statistics`, `viz.plot_style` | BOOTSTRAP | `uv run --locked python scripts/figures/fig_regime_comparison.py` |
| `fig_self_forecasting_loop.py` | Conceptual corporate loop with explicit external execution/learning handoffs | `efe.free_energy`, `model.generative_model`, `viz.plot_style` | ANALYTIC | `uv run --locked python scripts/figures/fig_self_forecasting_loop.py` |
| `fig_theta_decay.py` | Fixed-matrix Theta-capability predictions under `hold` / `fund_Theta` | `model.generative_model`, `viz.plot_style` | ANALYTIC | `uv run --locked python scripts/figures/fig_theta_decay.py` |
| `fig_trsi_densities.py` | t-RSI create/decay sample densities at the IMPROVING operating point | `trsi.t_rsi`, `model.operating_points`, `viz.plot_style` | BOOTSTRAP | `uv run --locked python scripts/figures/fig_trsi_densities.py` |
| `fig_trsi_sensitivity.py` | Synthetic t-RSI sensitivity to Dirichlet concentration and Theta freshness | `stats.sensitivity`, `viz.plot_style` | BOOTSTRAP | `uv run --locked python scripts/figures/fig_trsi_sensitivity.py` |
| `fig_value_by_regime.py` | One-step pragmatic and Theta-information values across operating beliefs | `efe.free_energy`, `model.generative_model`, `viz.plot_style` | ANALYTIC | `uv run --locked python scripts/figures/fig_value_by_regime.py` |


## Identify generated outputs, not hand-edited sources

- `output/data/demo_summary.json`, `output/figures/value_decomposition.png` — demo run
- `output/figures/*.png` — engine-generated manuscript figures
- `output/manuscript_variables.json`, `output/figures/figure_registry.json`,
  `output/reports/artifact_manifest.json`, `output/manuscript/*.md` — manuscript build
