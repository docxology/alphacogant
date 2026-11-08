# TODO — AlphaCOGANT backlog

Canonical backlog. Single source of truth for open work. Add entries as one line + file path(s).
Candidate package version: **1.2.2**. Integration/release status must be read from
fresh evidence; local version metadata is not a remote publication receipt.

Historical 1.2.1 validation (2026-10-16): 352 passed, one optional upstream
integration skipped in the base environment, 94.12% coverage. Historical alternate
runtime and token comparisons are not current-floor or current-release proof.
See [CHANGELOG.md](CHANGELOG.md), [agent API](docs/agent_api.md), and
[profiling](docs/performance.md) for scope and repeatable commands.

## Completed for 1.2.1

- [x] Native wheel dependency refresh and explicit no-build gates for Python 3.15 Linux (`uv.lock`, `.github/workflows/ci.yml`).

## Completed for 1.2.0

- [x] Batched EFE, bootstrap internal validation reductions and lazy plotting imports (`src/alphacogant/efe/free_energy.py`, `src/alphacogant/trsi/t_rsi.py`, `src/alphacogant/__init__.py`).
- [x] Versioned client/JSONL methods, capability schemas and runtime diagnostics (`src/alphacogant/api.py`).
- [x] Seeded CLI simulation/bootstrap, JSON envelopes, bounded inputs and atomic outputs (`src/alphacogant/__main__.py`, `src/alphacogant/io.py`).
- [x] Python 3.15 tracing and source-bound performance reports (`src/alphacogant/profiling.py`, `.github/workflows/ci.yml`).

## Completed for 1.1.0

- [x] Numeric GNN import/export, complete joint matrices, source-bound inference and replay (`src/alphacogant/bridge/`).
- [x] Real COGANT handoff and coefficient-level authored-model parity (`tests/test_cogant_integration.py`, `tests/test_gnn_numeric.py`).
- [x] Numerical boundary fixes and regression coverage (`tests/test_numerical_admission.py`).
- [x] Standalone packaging, refreshed lockfile, release identity and CI (`pyproject.toml`, `.github/workflows/ci.yml`).
- [x] Accurate installation, upstream scope and release documentation (`CONTRIBUTING.md`, `docs/upstream_compatibility.md`).

Historical round records below retain their as-of dates.

## Minor

- [x] AGENTS.md "198 tests" claim stale → verified 199 collected (2026-08-31, `pytest tests/ --collect-only -q`). Fixed in pass 1.
- [x] AGENTS.md "98%+ coverage" claim unverified → measured 97.70% (2026-08-31, full pytest --cov run). Fixed in pass 1; superseded by 98.70% after Round 2.
- [x] MANUSCRIPT_STATUS.md migration note garbled: said "Migrated from legacy `docs/manuscript/`" while this dir IS docs/manuscript; the legacy location was `manuscript/`. Fixed in pass 1.
- [x] README.md / AGENTS.md / docs/architecture.md / docs/rendering_and_validation.md referenced the stale `projects/working/alphacogant` mirror path and a nonexistent `/Users/4d/Documents/GitHub` checkout; actual location is this repo, symlinked into the template as `projects/ongoing/ActiveInference/alphacogant`. Fixed in pass 1.
- [x] `src/alphacogant/viz/plot_style.py` coverage outlier (86.05%, lines 116-118, 129-131). Fixed 2026-08-31 Round 2: two real-plot tests added (default-figsize/kwargs branch, default footer color/position branch); module now 100.00% (verified `pytest tests/test_plot_style.py --cov=alphacogant.viz.plot_style`).

## Medium

- [x] 5 pre-existing test failures from the manuscript migration (`manuscript/` → `docs/manuscript/`); tests read the legacy path. Resolved 2026-08-31 Round 2: the migration commit (eaf19cd) landed the prepared path-constant fixes (test_manuscript_contract.py, test_scripts.py deps); all five tests pass (verified full-suite run + individual rerun).

## Major

- [x] Land the pending manuscript-migration commit. Done 2026-08-31 Round 2: commit eaf19cd "build: complete manuscript/ -> docs/manuscript/ migration" moved manuscript/* (renames, 100%) into docs/manuscript/ together with the code/test/doc path fixes; suite green afterward (200 passed, 98.70% coverage).

## Open (re-scoped Round 2)

- [x] (Minor, owner-intent 2026-08-31) Generated per-directory AGENTS.md/README.md skeletons DELETED (described this standalone repo as template-monorepo local-only — wrong docs are worse than missing docs; acceptance met: 0 untracked AGENTS/README pairs). Recreate on demand with accurate standalone content if wanted.
- [x] (Minor, hygiene) output/ confirmed gitignored — acceptance met (`git status --porcelain | grep output/` empty, verified 2026-08-31). clean_output demo step left as optional nice-to-have, not backlog-worthy.
