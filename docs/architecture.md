# Package ownership and standalone computation flow

AlphaCOGANT is a standalone NumPy package. Computation lives under
`src/alphacogant/`; scripts coordinate artifact generation and own per-figure
layout; manuscript and docs consume generated artifacts. The thin orchestration
style is a design choice, not a dependency on an external template.

## Package ownership

| Package | Owns | Does not own |
| --- | --- | --- |
| `model/` | Channels, action names, A/B/C/D matrices, belief validation, inference. | Figure layout or manuscript wording. |
| `efe/` | Expected Free Energy decomposition, action values, policy posterior. | t-RSI bootstrapping or regime comparison tables. |
| `trsi/` | Post-action create/decay rates, paired t-RSI, scalar threshold predicate. | Action/model-update admission or financial calibration. |
| `stats/` | Bootstrap CIs, effect sizes, paired break-even profiles, trajectories. | Low-level model normalization. |
| `bridge/` | Firm priors, numeric GNN round trips, joint export and source-bound traces. | General COGANT indexing beyond this reduced project. |
| `tokens/` | Manuscript token generation from live model computations. | Direct rendering to PDF. |
| `viz/` | Shared figure style constants and utility helpers. | Per-figure scientific computation. |

## From fixed model to traces and manuscript artifacts

1. `default_model()` builds the reduced Economic World Model.
2. The observation-conditioned route filters caller-supplied current observations,
   scores one-step EFE, selects an action, then predicts the next state. Its
   [trace contract](cogant_bridge.md#source-bound-traces) does not authenticate chronology.
3. A separate prediction-only greedy/hold route computes
   [paired synthetic diagnostics](trsi_statistics.md); no t-RSI predicate gates allocation.
4. Figure scripts under `scripts/figures/` write `output/figures/*.png`.
5. `scripts/z_generate_manuscript_variables.py` writes:
   - `output/manuscript_variables.json`
   - injected Markdown under `output/manuscript/`
   - `output/figures/figure_registry.json`
   - `output/reports/artifact_manifest.json`
6. The standalone CLI validates/exports GNN and produces replayable traces.
7. Contained PDF/HTML rendering consumes the generated manuscript and is required
   for manuscript releases; external-template integration is optional. See the single
   [rendering recipe](rendering_and_validation.md).

## Standalone runtime; optional publication integration

The project is a standalone public repository (`docxology/alphacogant`).
Installation, model computation, CLI execution, tests and artifact generation
use the repository's own `pyproject.toml` and `uv.lock`. Template rendering is an
optional external integration. No absolute workstation path is part of the
runtime contract.
