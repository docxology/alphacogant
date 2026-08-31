# TODO — AlphaCOGANT backlog

Canonical backlog. Single source of truth for open work. Add entries as one line + file path(s).
Status as of 2026-08-31, Round 2 (verified by `uv run --no-project pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q` → 200 passed, 98.70% coverage; full log in REVIEW_LOG_2026-08-31.md).

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

- [ ] (Minor, owner-intent) Keep or drop the generated per-directory AGENTS.md/README.md skeleton pairs (src/, tests/, scripts/, models/) that remain untracked: they were auto-generated against the template-monorepo layout and do not describe this standalone repo. Either rewrite for this repo and commit, or delete. Acceptance: `git status --porcelain | grep -c '?? .*\(AGENTS\|README\).md'` returns 0 or the tracked set matches repo reality.
- [ ] (Minor, hygiene) `output/figure_scripts/` accumulates per-run subprocess logs; confirm they are covered by .gitignore (they are not tracked) and consider a `clean_output` step in the demo script. Acceptance: `git status --porcelain | grep output/` is empty.
