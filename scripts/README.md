# AlphaCOGANT Scripts

Thin orchestrators only: every script bootstraps paths, parses CLI flags, and makes a
single delegated call into `src/alphacogant/`. All business, data, and plot logic lives
in the package and is covered by `tests/`.

## Inventory

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `run_alphacogant_demo.py` | Demo pipeline: EWM build, state inference, EFE decomposition + marginal-return vector, t-RSI certificate; renders one comparison figure | `alphacogant.model.generative_model`, `alphacogant.efe.free_energy`, `alphacogant.trsi.t_rsi` | `uv run --no-project python scripts/run_alphacogant_demo.py` |
| `y_generate_figures.py` | Batch runner: executes every `scripts/figures/fig_*.py` deterministically (Agg backend, `PYTHONPATH=src`), stops on first failure | all `scripts/figures/fig_*.py` | `uv run --no-project python scripts/y_generate_figures.py` |
| `z_generate_manuscript_variables.py` | Manuscript build gate: generates {{TOKEN}}s, fails on orphan tokens, writes figure registry + artifact manifest, and (default mode) writes token-substituted chapters | `alphacogant.tokens.manuscript_variables`, `alphacogant.tokens.manuscript_artifacts` | `uv run --no-project python scripts/z_generate_manuscript_variables.py [--check]` |

## Manuscript figures (`scripts/figures/`)

Each script renders one deterministic PNG into `output/figures/`; every number it draws
comes from `src/alphacogant/`.

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `fig_aif_dictionary.py` | Two-column visual dictionary: AlphaFund constructs → Active-Inference objects | `efe.free_energy`, `model.generative_model` | `uv run --no-project python scripts/figures/fig_aif_dictionary.py` |
| `fig_belief_trajectory.py` | Multi-cycle belief trajectory under the greedy policy | `stats.simulation` | `uv run --no-project python scripts/figures/fig_belief_trajectory.py` |
| `fig_break_even_probability.py` | Break-even probability profile | `stats.statistics`, `viz.plot_style` | `uv run --no-project python scripts/figures/fig_break_even_probability.py` |
| `fig_certificate_sign_flip.py` | t-RSI comparator certificate sign-flip (not green-by-construction) | `trsi.t_rsi`, `model.generative_model` | `uv run --no-project python scripts/figures/fig_certificate_sign_flip.py` |
| `fig_cover_art.py` | Cover art from live channel values | `viz.plot_style` | `uv run --no-project python scripts/figures/fig_cover_art.py` |
| `fig_create_vs_decay_scatter.py` | Create-rate vs decay-rate scatter across bootstrap perturbations | `trsi.t_rsi`, `viz.plot_style` | `uv run --no-project python scripts/figures/fig_create_vs_decay_scatter.py` |
| `fig_efe_waterfall.py` | EFE decomposition waterfall for the funded channel | `efe.free_energy`, `viz.plot_style` | `uv run --no-project python scripts/figures/fig_efe_waterfall.py` |
| `fig_gnn_factor_graph.py` | EWM-as-GNN factor graph of the five-channel firm | `efe.free_energy`, `model.generative_model` | `uv run --no-project python scripts/figures/fig_gnn_factor_graph.py` |
| `fig_marginal_return_heatmap.py` | Marginal-return vector (negative EFE) heatmap over actions × cycles | `stats.simulation` | `uv run --no-project python scripts/figures/fig_marginal_return_heatmap.py` |
| `fig_policy_posterior.py` | Policy-posterior evolution across the greedy trajectory | `efe.free_energy`, `stats.simulation` | `uv run --no-project python scripts/figures/fig_policy_posterior.py` |
| `fig_regime_comparison.py` | Regime comparison with bootstrap confidence intervals | `stats.statistics`, `viz.plot_style` | `uv run --no-project python scripts/figures/fig_regime_comparison.py` |
| `fig_self_forecasting_loop.py` | Self-Forecasting Loop schematic (perception–action cycle) | `efe.free_energy`, `model.generative_model` | `uv run --no-project python scripts/figures/fig_self_forecasting_loop.py` |
| `fig_theta_decay.py` | Theta-belief alpha-decay vs refresh under `hold` / `fund_Theta` | `model.generative_model` | `uv run --no-project python scripts/figures/fig_theta_decay.py` |
| `fig_trsi_densities.py` | t-RSI create/decay sample densities at the IMPROVING operating point | `trsi.t_rsi`, `model.operating_points` | `uv run --no-project python scripts/figures/fig_trsi_densities.py` |
| `fig_trsi_sensitivity.py` | t-RSI sensitivity to belief precision and Theta freshness | `stats.sensitivity`, `viz.plot_style` | `uv run --no-project python scripts/figures/fig_trsi_sensitivity.py` |
| `fig_value_by_regime.py` | Epistemic vs pragmatic value per channel across operating regimes | `efe.free_energy`, `model.generative_model` | `uv run --no-project python scripts/figures/fig_value_by_regime.py` |

## Outputs

- `output/data/demo_summary.json`, `output/figures/value_decomposition.png` — demo run
- `output/figures/*.png` — engine-generated manuscript figures
- `output/manuscript_variables.json`, `output/figures/figure_registry.json`,
  `output/reports/artifact_manifest.json`, `output/manuscript/*.md` — manuscript build
