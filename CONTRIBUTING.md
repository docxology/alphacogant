# Develop, verify and release the standalone research package

AlphaCOGANT is a standalone Python package. Use a fresh checkout, Python (the bound in `pyproject.toml`) or newer
and [uv](https://docs.astral.sh/uv/). No template workspace is required.

```bash
uv sync --locked --extra dev
uv run --locked ruff check scripts src tests
uv run --locked pytest tests/ --cov=alphacogant --cov-fail-under=90 -q
uv run --locked alphacogant validate models/alphafund_ewm.md
uv run --locked python scripts/run_alphacogant_demo.py
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
uv run --locked python scripts/z_generate_manuscript_variables.py --check
uv build
```

For the fast numerical loop use `pytest tests/ -m 'not figures'`. The `figures`
marker selects figure subprocess tests and manuscript checks that require
those generated figures. Full validation includes those
tests and the demo/manuscript pipeline. Tests use real computations and explicit
NumPy generators; do not add mocks or global random seeds. Keep EFE minimized,
epistemic information gain nonnegative, and claims confined to the reduced model.

## Prove the installed wheel and console in an independent environment

After building, use an isolated wheel environment and a separate working
directory without source imports. Keep durable caches/environments local:

```bash
checkout="$PWD"
mkdir -p "$checkout/output"
export UV_CACHE_DIR="$checkout/output/.cache/uv"
work="$(mktemp -d "$checkout/output/installed-smoke.XXXXXX")"
uv venv --python "$checkout/.venv/bin/python" "$work/environment"
uv pip install --python "$work/environment/bin/python" \
  "$checkout/dist/alphacogant-1.3.1-py3-none-any.whl"
mkdir "$work/cwd"
cd "$work/cwd"
PATH="$work/environment/bin:$PATH" "$work/environment/bin/python" \
  "$checkout/scripts/smoke_installed.py" --source-root "$checkout"
```

The smoke requires the installed module/console to belong to that environment,
checks packaged `SPEC.md` bytes, model round-trip, real diagnostics and JSONL.
It refuses accidental checkout imports. CI uses its disposable runner temp cwd.

CI checks Python 3.11, 3.12, 3.13, 3.14, and 3.15. The development pin lives in `.python-version`;
use `uv sync --python 3.11 --locked --extra dev` to exercise the oldest supported runtime. Run the real pinned COGANT integration as
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
registry/artifact manifest valid. Manuscript/visual releases must also run the
[contained source-owned renderer and actual inspection](docs/rendering_and_validation.md).
External-template rendering is optional and never a substitute for that local
source/artifact contract.

For a release, update `pyproject.toml`, `uv.lock`, `alphacogant.__version__`,
`CITATION.cff`, `codemeta.json`, `.zenodo.json`, `docs/manuscript/config.yaml`,
and `CHANGELOG.md` together. Preserve the concept DOI; remove stale version-specific
deposit fields and add a new version DOI only once that deposit exists. Complete
native/installed-wheel checks, the real pinned upstream check, rendered-product
inspection and required CI before merging. The CI tag gate requires an annotated
`v<version>` whose commit belongs to main history; create it on the validated main
commit. Publish ordinary fast-forward changes and retain release asset hashes;
never force-push or replace an existing tag. Source-owned scripts, manuscript
sources and archival metadata are included in the source distribution.

## Agent and performance development

Use `alphacogant capabilities` for the current schema and `alphacogant doctor`
for installed-runtime validation. Profile with `uv run --locked alphacogant profile`;
keep timings separate from instrumentation and compare matched runtimes,
NumPy versions, workloads, seeds and thread settings. Read [the profiling guide](docs/performance.md)
and [agent API](docs/agent_api.md). Never gate CI on wall-time thresholds from a
shared runner; numerical parity, output provenance and workload bounds are gates.
