# Architecture

AlphaCOGANT follows the template's thin-orchestrator pattern. Computation lives
under `src/alphacogant/`; scripts coordinate output generation; manuscript and
docs consume generated artifacts.

## Package ownership

| Package | Owns | Does not own |
| --- | --- | --- |
| `model/` | Channels, action names, A/B/C/D matrices, belief validation, inference. | Figure layout or manuscript wording. |
| `efe/` | Expected Free Energy decomposition, action values, policy posterior. | t-RSI bootstrapping or regime comparison tables. |
| `trsi/` | Create-rate, decay-rate, standardized t-RSI, certificate predicate. | Confidence intervals or figure rendering. |
| `stats/` | Bootstrap CIs, effect sizes, paired break-even profiles, trajectories. | Low-level model normalization. |
| `bridge/` | Firm-structure priors and GNN round-trip summaries. | General COGANT indexing beyond this reduced project. |
| `tokens/` | Manuscript token generation from live model computations. | Direct rendering to PDF. |
| `viz/` | Shared figure style constants and utility helpers. | Per-figure scientific computation. |

## Data flow

1. `default_model()` builds the reduced Economic World Model.
2. Engine functions compute inference, EFE, t-RSI, trajectories, and statistics.
3. Figure scripts under `scripts/figures/` write `output/figures/*.png`.
4. `scripts/z_generate_manuscript_variables.py` writes:
   - `output/manuscript_variables.json`
   - injected Markdown under `output/manuscript/`
   - `output/figures/figure_registry.json`
   - `output/reports/artifact_manifest.json`
5. The template renderer builds `output/pdf/alphacogant_combined.pdf`.
6. `scripts/05_copy_outputs.py` copies the validated output tree to
   `template/output/working/alphacogant/`.

## Boundary

The project is a local private working project. It is edited in
`/Users/4d/Documents/GitHub/projects/working/alphacogant` and rendered through
the sibling template checkout as `working/alphacogant`.
