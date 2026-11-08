# Generate registered figures and inspect artifact provenance

Figures are source-owned by scripts, not by binary edits under `output/`.

## Generate manuscript figures from source-owned producers

The ordered generation command is owned by
[rendering and validation](rendering_and_validation.md). Each script under
`scripts/figures/` writes one PNG to `output/figures/`; the aggregate runner sets
`MPLBACKEND=Agg` and runs them in deterministic order. The
[script inventory](../scripts/README.md) maps filenames to producers and footer modes.

`fig_cover_art.py` produces `cover_art.png`, the landscape graphical abstract
registered from the end of `manuscript/00_abstract.md`. It distinguishes supplied
observation filtering and EFE action selection from the separate prediction-only
greedy/hold diagnostic. It carries provenance like other manuscript figures;
it is not no-footer decorative title art. The dictionary, conceptual
self-forecasting loop and GNN factor graph retain their distinct explanatory roles.

## Discover labels, captions and expected producer paths

`scripts/z_generate_manuscript_variables.py` reads manuscript image references
and writes `output/figures/figure_registry.json`. Each registry entry records:

- figure label
- filename
- source manuscript file
- caption text after token injection
- producer script

## Inventory current bytes and detect structural image issues

`output/reports/artifact_manifest.json` records hashes and sizes for stable
artifacts. It also checks registered manuscript figures:

- duplicate labels
- duplicate filenames
- missing files
- tiny files
- invalid PNG magic bytes
- missing producer scripts
- registered PNG decoding, not only the signature

The manifest `issues` array must be empty before rendering. `--check` compares
saved tokens, registry, file hashes and injected Markdown without writing them;
tampered/missing/malformed output fails. Generation publishes text atomically
and fails on artifact issues. A multi-file generation is not a transaction.

Hash equality is an inventory check, not proof that a producer ran on current
source. Regenerate, retain source/runtime/command receipts and inspect actual
figure semantics and rendered pages before publication.

The graphical abstract receives the same registry/image checks as other manuscript
figures. The demo-only `value_decomposition.png` is separate from the manuscript.
The raw LaTeX page breaks around the graphical abstract affect PDF only; HTML
ignores those raw blocks and keeps the image after the abstract text.

## Optional publication-copy parity

An independently configured template workflow may copy the project output tree
to its authorized publication directory. Those stages are not standalone gates;
if used, the copied PDF should hash-match the project-local PDF.
