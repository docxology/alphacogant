# Trace a manuscript claim from source to output

AlphaCOGANT is source-owned: repair manuscript prose, engine code or figure
producers, then regenerate; do not patch generated output. This page owns the
cross-surface map, not the detailed numerical or build contracts.

## Source owners and generated products

| Surface | Source owner | Product / detailed contract |
| --- | --- | --- |
| Generative model | `src/alphacogant/model/` | A/B/C/D and filtering; [model contract](generative_model_contract.md) |
| Discrete action scoring | `src/alphacogant/efe/free_energy.py` | pragmatic, Theta-information, total G and policy; [EFE contract](efe_and_policy.md) |
| Separate paired diagnostics | `src/alphacogant/trsi/t_rsi.py`, `stats/statistics.py` | post-action rates, paired mean/SE, intervals and event mass; [statistics contract](trsi_statistics.md) |
| Manuscript numerics | `src/alphacogant/tokens/manuscript_variables.py` | `output/manuscript_variables.json` |
| Manuscript image references | `docs/manuscript/` and `tokens/manuscript_artifacts.py` | substituted `output/manuscript/*.md`, `output/figures/figure_registry.json` |
| Figure content | `scripts/figures/` via `scripts/y_generate_figures.py` | `output/figures/*.png`; [figure/provenance contract](figures_and_artifacts.md) |
| Artifact inventory | `tokens/manuscript_artifacts.py` | `output/reports/artifact_manifest.json` |

The graphical abstract is a registered manuscript figure in `00_abstract.md`,
produced by `scripts/figures/fig_cover_art.py` as `cover_art.png`. It is not a
separate manuscript chapter or unregistered decorative cover.

## Follow the regeneration and review route

[Rendering and validation](rendering_and_validation.md) owns the ordered generation,
structural check and contained PDF/HTML recipe required for manuscript releases. [Testing contract](testing_contract.md)
owns software-test expectations. [Figures and artifacts](figures_and_artifacts.md)
explains exactly what registry and manifest checks establish.

The [manuscript bibliography](manuscript/references.bib) identifies methodological
sources. The [historical concept inventory](whitepaper_concepts.md) preserves
attribution without treating unavailable corporate assertions as reproduced local
results. [Claim boundaries](claim_boundaries.md) owns the high-level honesty contract.
