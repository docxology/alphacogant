# Methods and artifact contract

AlphaCOGANT is source-owned: manuscript claims, figures, and registries should be
regenerated from `src/alphacogant/` and `scripts/`, not edited in `output/`.
The [manuscript bibliography](manuscript/references.bib) identifies methodological
sources; the [historical concept inventory](whitepaper_concepts.md) distinguishes
external attribution from locally reproduced synthetic results.

## Method surfaces

| Surface | Source owner | Output |
| --- | --- | --- |
| Generative model | `src/alphacogant/model/` | A/B/C/D matrices and state inference |
| Expected Free Energy | `src/alphacogant/efe/free_energy.py` | pragmatic, epistemic, total G, policy posterior |
| t-RSI diagnostic | `src/alphacogant/trsi/t_rsi.py` | post-action create/decay rates, paired standardized gap |
| Regime statistics | `src/alphacogant/stats/statistics.py` | synthetic percentile intervals, effect sizes, event mass |
| Manuscript tokens | `src/alphacogant/tokens/manuscript_variables.py` | `output/manuscript_variables.json` |
| Figure scripts | `scripts/figures/` via `scripts/y_generate_figures.py` | `output/figures/*.png` and `figure_registry.json` |

## Break-even probability

`statistics.break_even_profile` estimates `P(create_rate > decay_rate)` with paired
bootstrap perturbations. Each draw perturbs the operating belief once and evaluates
both rates on that same perturbed belief, so the event is not distorted by
independent create and decay samples. The statistic complements t-RSI:

- t-RSI reports standardized distance.
- break-even probability reports event mass over the paired bootstrap cloud.
- the margin interval reports percentiles of the synthetic paired-margin cloud.

## Regeneration sequence

Run from the project root:

```bash
uv run --locked pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q
uv run --locked python scripts/run_alphacogant_demo.py
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
```

Use the contained Pandoc recipe in
[rendering_and_validation.md](rendering_and_validation.md). External template
integration is optional and requires independent authorization; standalone work
does not run another checkout's pipeline.

The validation gate expects `output/figures/figure_registry.json`; it is written by
`scripts/z_generate_manuscript_variables.py` from live manuscript references.

The token gate checks referenced token coverage. `--check` compares saved tokens,
registry, manifest hashes and injected prose without rewriting them. Regenerate
before checking. Arbitrary literals, scientific interpretations and producer
execution are not certified by those structural comparisons.

## Modular references

- [architecture.md](architecture.md) explains package ownership and data flow.
- [figures_and_artifacts.md](figures_and_artifacts.md) details the registry and
  manifest checks.
- [rendering_and_validation.md](rendering_and_validation.md) lists the template
  render and validation commands.
- [claim_boundaries.md](claim_boundaries.md) records the manuscript honesty
  contract.
