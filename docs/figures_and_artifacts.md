# Figures and Artifacts

Figures are source-owned by scripts, not by binary edits under `output/`.

## Figure generation

Run all figure scripts from the project root:

```bash
uv run --locked python scripts/y_generate_figures.py
```

Each script writes one PNG under `output/figures/`. The aggregate script sets
`MPLBACKEND=Agg` and runs scripts in deterministic order.

## Registry

`scripts/z_generate_manuscript_variables.py` reads manuscript image references
and writes `output/figures/figure_registry.json`. Each registry entry records:

- figure label
- filename
- source manuscript file
- caption text after token injection
- producer script

## Manifest

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

## Root copy

When using the optional template integration, stage 05 copies the project
output tree to the template's configured publication directory.
The copied PDF should hash-match the project-local PDF.
