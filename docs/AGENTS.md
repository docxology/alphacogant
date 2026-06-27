# Docs Agent Notes

Documentation in this directory is authored source, not generated output.

## Scope

- Keep each file modular: one purpose, one technical surface.
- Prefer links to source-owned paths over copying large code snippets.
- Keep claim language aligned with `docs/claim_boundaries.md` and the manuscript
  limitations section.
- When a doc references a generated artifact, name the source script that creates it.

## Required checks after docs edits

Run these from the project root:

```bash
uv run --no-project pytest tests/test_docs_contract.py tests/test_manuscript_contract.py -q
uvx ruff check scripts src tests
```

For PDF-facing changes, also regenerate:

```bash
uv run --no-project python scripts/y_generate_figures.py
uv run --no-project python scripts/z_generate_manuscript_variables.py
```

Then render from the template root with:

```bash
uv run python scripts/03_render_pdf.py --project working/alphacogant
uv run python scripts/04_validate_output.py --project working/alphacogant
uv run python scripts/05_copy_outputs.py --project working/alphacogant
```
