# Contributing

AlphaCOGANT is a standalone Python package. Use a fresh checkout, Python (the bound in `pyproject.toml`) or newer
and [uv](https://docs.astral.sh/uv/). No template workspace is required.

```bash
uv sync --locked --extra dev
uv run --locked ruff check scripts src tests
uv run --locked pytest tests/ --cov=alphacogant --cov-fail-under=90 -q
uv run --locked alphacogant validate models/alphafund_ewm.md
uv run --locked python scripts/z_generate_manuscript_variables.py --check
uv build
```

For the fast numerical loop use `pytest tests/ -m 'not figures'`. The `figures`
marker selects figure subprocess tests and manuscript checks that require
those generated figures. Full validation includes those
tests and the demo/manuscript pipeline. Tests use real computations and explicit
NumPy generators; do not add mocks or global random seeds. Keep EFE minimized,
epistemic information gain nonnegative, and claims confined to the reduced model.

## Installed wheel

After building, run outside the source checkout to detect missing package data,
undeclared imports, and accidental reliance on the repository layout:

```bash
# Replace /absolute/path/alphacogant with your checkout.
cd /tmp
uv run --isolated --no-project \
  --with /absolute/path/alphacogant/dist/alphacogant-1.2.1-py3-none-any.whl \
  python -c 'import alphacogant; from importlib.metadata import version; assert alphacogant.__version__ == version("alphacogant"); from alphacogant.bridge.gnn import model_from_gnn, model_to_gnn; model_from_gnn(model_to_gnn(alphacogant.default_model()))'
```

CI checks Python 3.10, 3.12, 3.14, and 3.15. The development pin is 3.15;
use `uv sync --python 3.10 --locked --extra dev` to exercise the oldest supported runtime. Run the real pinned COGANT integration as
shown in [upstream compatibility](docs/upstream_compatibility.md) when changing
the bridge. It is intentionally separate from ordinary installation.

## Artifacts and releases

Business logic belongs in `src/alphacogant`; scripts orchestrate it. Generated
figures, variables, traces, exports and reports belong in ignored `output/`.
Regenerate manuscript artifacts with:

```bash
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
```

Keep every prose `{{TOKEN}}` generated, all citations defined, and the figure
registry and artifact manifest valid. External template PDF rendering is an
optional integration requiring that template's configured project location;
standalone validation does not require the old workstation paths.

For a release, update `pyproject.toml`, `uv.lock`, `alphacogant.__version__`,
`CITATION.cff`, `codemeta.json`, and `CHANGELOG.md` together. Preserve the concept
DOI; only add a new version DOI once that deposit exists. Complete the checks,
commit, and create an annotated `v<version>` tag on the released commit. Publish
with ordinary fast-forward pushes; never force-push a release.

## Agent and performance development

Use `alphacogant capabilities` for the current schema and `alphacogant doctor`
for installed-runtime validation. Profile with `uv run --locked alphacogant profile`;
keep timings separate from instrumentation and compare matched runtimes,
NumPy versions, workloads, seeds and thread settings. Read [the profiling guide](docs/performance.md)
and [agent API](docs/agent_api.md). Never gate CI on wall-time thresholds from a
shared runner; numerical parity, output provenance and workload bounds are gates.
