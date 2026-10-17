# Rendering and Validation

AlphaCOGANT lives in the public GitHub repo `docxology/alphacogant` and runs standalone. Optional PDF rendering uses a separately configured
template checkout.

## Project-root commands

From this repo root, after `uv sync --locked --extra dev`:

```bash
uv run --locked pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q
uv run --locked python scripts/run_alphacogant_demo.py
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
```

## Optional template integration

From a compatible template checkout with this project linked at
`projects/ongoing/ActiveInference/alphacogant` (adjust the project identifier to
your template configuration):

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
