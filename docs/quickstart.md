# Standalone setup and examples

## Install and exercise the standalone engine

```bash
# From a fresh checkout; Python (the bound in `pyproject.toml`) or newer and uv are required.
uv sync --locked --extra dev
uv run --locked alphacogant validate models/alphafund_ewm.md
uv run --locked alphacogant export --cogant -o output/model.gnn.md
uv run --locked alphacogant infer output/model.gnn.md --steps 8 -o output/trace.json
uv run --locked alphacogant verify-trace output/model.gnn.md output/trace.json
uv run --locked python scripts/run_alphacogant_demo.py
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
uv run --locked python scripts/z_generate_manuscript_variables.py --check
uv run --locked pytest tests/ --cov=alphacogant --cov-fail-under=90 -q
```

Exports preserve existing files; use `--overwrite` to replace a generated file.
`infer --observations observations.json` accepts a JSON array of `[reward, loss]`
integer bucket pairs. Without observations it records a synthetic model rollout.
The trace binds the exact source bytes, coefficients, filtering, EFE scores, and
selected actions. The existing `simulate_trajectory` API records open-loop
predictions; t-RSI is a separate paired synthetic diagnostic whose scalar threshold
does not gate actions or updates. Chronological input integrity remains caller-owned.
See [CONTRIBUTING.md](contributing.md) for CI parity and
wheel installation checks.

See the [documentation index](README.md) for the model contract, output
inventory, and manuscript rendering.
