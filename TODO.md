# TODO — AlphaCOGANT backlog

Canonical backlog. Single source of truth for open work. Add entries as one line + file path(s).
Status as of 2026-08-31 (verified by `uv run --no-project pytest tests/ --collect-only -q` → 199 tests; full-suite run recorded in REVIEW_LOG_2026-08-31.md).

## Minor

- [x] AGENTS.md "198 tests" claim stale → verified 199 collected (2026-08-31, `pytest tests/ --collect-only -q`). Fixed in this pass.
- [x] AGENTS.md "98%+ coverage" claim unverified → measured 97.70% (2026-08-31, full pytest --cov run). Fixed in this pass.
- [x] MANUSCRIPT_STATUS.md migration note garbled: said "Migrated from legacy `docs/manuscript/`" while this dir IS docs/manuscript; the legacy location was `manuscript/`. Fixed in this pass.
- [x] README.md / AGENTS.md / docs/architecture.md / docs/rendering_and_validation.md referenced the stale `projects/working/alphacogant` mirror path and a nonexistent `/Users/4d/Documents/GitHub` checkout; actual location is this repo, symlinked into the template as `projects/ongoing/ActiveInference/alphacogant`. Fixed in this pass.
- [ ] `src/alphacogant/viz/plot_style.py` is the coverage outlier (86.05%, lines 116-118, 129-131 — below the 90% per-module bar even though the suite total passes). Either add real-plot tests or trim dead branches.

## Medium

- [ ] 5 pre-existing test failures from the in-flight manuscript migration (`manuscript/` → `docs/manuscript/`); tests still read the legacy path:
  - tests/test_manuscript_contract.py::test_cover_art_is_declared_in_manuscript_config
  - tests/test_manuscript_contract.py::test_claim_boundary_language_stays_reduced_model_only
  - tests/test_manuscript_contract.py::test_bootstrap_caption_values_are_token_injected
  - tests/test_manuscript_contract.py::test_reference_ids_defined_and_used (FileNotFoundError: `manuscript/references.bib`)
  - tests/test_scripts.py::test_generate_manuscript_variables_check_mode
  Fix: repoint the test path constants/helpers to `docs/manuscript/` once the migration commit lands.

## Major

- [ ] Land the pending manuscript-migration commit: the working tree deletes legacy `manuscript/` and adds `docs/manuscript/`, but the test suite still references the legacy path, so the repo is not green in either state. Committing migration + test-path fix together restores a green baseline.
