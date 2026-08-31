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
6. `scripts/pipeline/stage_05_copy.py` copies the validated output tree to
   `template/output/ongoing/ActiveInference/alphacogant/` (verify the live output path: `ls /Volumes/external_drive/Git/template/output/ongoing/ActiveInference/ 2>/dev/null`).

## Boundary

The project is a local private project. It is edited in this repo
(`/Volumes/external_drive/Git/projects/ongoing/docxology/alphacogant`), which is
symlinked into the template checkout as `projects/ongoing/ActiveInference/alphacogant`
and rendered through the template as `ongoing/ActiveInference/alphacogant`.
