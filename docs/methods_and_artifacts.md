# Methods and artifact contract

AlphaCOGANT is source-owned: manuscript claims, figures, and registries should be
regenerated from `src/alphacogant/` and `scripts/`, not edited in `output/`.
This design follows the Whitepaper's closed-loop methodology [1], the
Active Inference filtration contract [2, 3, 4], and the COGANT/GNN source-owned
artifact pattern [5, 6].

## Method surfaces

| Surface | Source owner | Output |
| --- | --- | --- |
| Generative model | `src/alphacogant/model/` | A/B/C/D matrices and state inference |
| Expected Free Energy | `src/alphacogant/efe/free_energy.py` | pragmatic, epistemic, total G, policy posterior |
| t-RSI certificate | `src/alphacogant/trsi/t_rsi.py` | create-rate, decay-rate, standardized gap |
| Regime statistics | `src/alphacogant/stats/statistics.py` | confidence intervals, effect sizes, break-even probability |
| Manuscript tokens | `src/alphacogant/tokens/manuscript_variables.py` | `output/manuscript_variables.json` |
| Figure scripts | `scripts/figures/` via `scripts/y_generate_figures.py` | `output/figures/*.png` and `figure_registry.json` |

## Break-even probability

`statistics.break_even_profile` estimates `P(create_rate > decay_rate)` with paired
bootstrap perturbations. Each draw perturbs the operating belief once and evaluates
both rates on that same perturbed belief, so the event is not distorted by
independent create and decay samples. The statistic complements t-RSI:

- t-RSI reports standardized distance.
- break-even probability reports event mass over the paired bootstrap cloud.
- the margin interval reports uncertainty in `create_rate - decay_rate`.

## Regeneration sequence

Run from the project root:

```bash
uv run --no-project pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q
uv run --no-project python scripts/run_alphacogant_demo.py
uv run --no-project python scripts/y_generate_figures.py
uv run --no-project python scripts/z_generate_manuscript_variables.py
```

Then from the template root:

```bash
uv run python scripts/03_render_pdf.py --project working/alphacogant
uv run python scripts/04_validate_output.py --project working/alphacogant
```

The validation gate expects `output/figures/figure_registry.json`; it is written by
`scripts/z_generate_manuscript_variables.py` from live manuscript references.

The same gate closes over numeric claims: `scripts/z_generate_manuscript_variables.py`
injects `{{TOKEN}}` values from `src/alphacogant/tokens/` and fails the build if
manuscript constants drift from engine outputs [5, 6].

## Modular references

- [architecture.md](architecture.md) explains package ownership and data flow.
- [figures_and_artifacts.md](figures_and_artifacts.md) details the registry and
  manifest checks.
- [rendering_and_validation.md](rendering_and_validation.md) lists the template
  render and validation commands.
- [claim_boundaries.md](claim_boundaries.md) records the manuscript honesty
  contract.
