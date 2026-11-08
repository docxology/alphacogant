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
uv run --locked pytest tests/test_docs_contract.py tests/test_manuscript_contract.py -q
uv run --locked ruff check scripts src tests
```

For PDF-facing changes, also regenerate:

```bash
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
```

Use the contained Pandoc recipe in [rendering_and_validation.md](rendering_and_validation.md).
Template stages are optional external integration, not a required standalone gate
or permission to write another checkout. Inspect regenerated figures and rendered
pages; token/manifest checks alone do not establish scientific correctness.
