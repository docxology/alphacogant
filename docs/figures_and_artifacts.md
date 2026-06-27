# Figures and Artifacts

Figures are source-owned by scripts, not by binary edits under `output/`.

## Figure generation

Run all figure scripts from the project root:

```bash
uv run --no-project python scripts/y_generate_figures.py
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

The manifest `issues` array must be empty before render closeout.

## Root copy

After validation, template stage 05 copies the project output tree to
`/Users/4d/Documents/GitHub/template/output/working/alphacogant/`. The copied PDF
should hash-match the project-local PDF.
