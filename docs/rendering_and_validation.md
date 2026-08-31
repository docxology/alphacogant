# Rendering and Validation

AlphaCOGANT is a private project rendered through the sibling template checkout.

## Project-root commands

From this repo root (`/Volumes/external_drive/Git/projects/ongoing/docxology/alphacogant`):

```bash
uv run --no-project pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q
uv run --no-project python scripts/run_alphacogant_demo.py
uv run --no-project python scripts/y_generate_figures.py
uv run --no-project python scripts/z_generate_manuscript_variables.py
```

## Template-root commands

From the template checkout (`/Volumes/external_drive/Git/template`):

```bash
uv run python scripts/pipeline/stage_03_render.py --project ongoing/ActiveInference/alphacogant
uv run python scripts/pipeline/stage_04_validate.py --project ongoing/ActiveInference/alphacogant
uv run python scripts/pipeline/stage_05_copy.py --project ongoing/ActiveInference/alphacogant
```

## Expected validation surfaces

Stage 04 checks:

- PDF validity
- transmission bookends
- Markdown validity
- output directory structure
- figure registry references
- evidence registry
- design overlays
- artifact manifest

The evidence-registry step may print unsupported-number diagnostics for reference
metadata and public AlphaFund numbers. The load-bearing status is the stage
summary: it must report PASS for every check.
