---
project: alphacogant
task: "Render AlphaFund recursive self-improvement as Active Inference in GNN via COGANT"
effort: E3
phase: complete
progress: 16/16
mode: ALGORITHM
started: 2026-06-23
updated: 2026-06-27
---

# AlphaCOGANT — Ideal State Artifact

## Problem

The AlphaFund whitepaper formalizes recursive self-improvement (RSI) as a portfolio
optimization problem with an Economic World Model (EWM), five capital channels, a
marginal-return objective, and a t-RSI certificate — but leaves the construction as
prose + proprietary surfaces. It is, term for term, an Active Inference agent, yet
nothing makes that correspondence explicit, executable, or integrity-checked. There
is no generative-model artifact, no filtration-safe inference engine, and no
reproducible link between the firm's claimed statistics and shipped computation.

## Vision

A reader who knows either Active Inference or AlphaFund instantly recognizes the
other in it: the five channels *are* hidden-state factors, the portfolio optimizer
*is* Expected Free Energy minimization, t-RSI *is* the EFE-improvement certificate —
and the epistemic/pragmatic value split finally says *why* a data-feed dollar and a
trading dollar are comparable. The artifact is a real prepared project (scaffolded
from `template_code_project`, importing the GNN modeling notation and the COGANT
translation pattern) with a GNN model file, a tested engine, and a manuscript whose
every number is gated against that engine.

## Out of Scope

Not a trading system, not financial advice, not a reproduction of AlphaFund's
proprietary execution-friction surface, fitted scaling exponents, or live track
record (all non-public). No live market data, no broker integration, no LLM
dependency in the engine. The GNN model is a legible reduced (two-level) encoding,
not a numerically-calibrated replica of AlphaFund's books.

## Constraints

- Scaffold from `template_code_project`; obey the thin-orchestrator pattern (logic in
  `src/`, scripts coordinate), `{{TOKEN}}` reproducibility, ≥90% `src/` coverage, no
  mocks, deterministic seeds.
- Engine is NumPy-only; all randomness via an explicit `numpy.random.Generator`.
- Active Inference sign convention fixed once: EFE `G` minimized, value `= -G`;
  epistemic value (a KL) is always ≥ 0.
- GNN model file conforms to GNN section structure (StateSpaceBlock, Connections,
  InitialParameterization, ActInfOntologyAnnotation).
- Lives local-only under `projects/working/alphacogant/` (confidentiality invariant —
  only `projects/templates/` is git-tracked).

## Goal

Deliver `projects/working/alphacogant/`: a GNN model file encoding AlphaFund's
five-channel EWM as an Active Inference generative model; a tested NumPy engine that
performs channel inference, computes the epistemic/pragmatic Expected-Free-Energy
split, the marginal-return vector, and the t-RSI certificate; a COGANT-style
structure→model bridge; and a manuscript that fully writes out the concept with every
cited numeric gated by `manuscript_variables`.

## Criteria

- [x] ISC-1: `models/alphafund_ewm.md` exists with all GNN sections and five channel factors.
- [x] ISC-2: GNN file's A_R/A_L/B_*/C_*/D_* are well-formed (column-stochastic / sum-to-1).
- [x] ISC-3: `src/alphacogant/channels.py` defines the 5 channels + 6 actions with roles.
- [x] ISC-4: `generative_model.default_model()` loads the GNN values and validates distributions.
- [x] ISC-5: `infer_states` moves belief toward `strong` after a high-reward+low-loss obs.
- [x] ISC-6: `expected_free_energy` returns pragmatic, epistemic, total with total == -(prag+epist).
- [x] ISC-7: epistemic value ≥ 0 for every action (asserted + tested).
- [x] ISC-8: funding stale `Theta` yields higher epistemic than `hold` (tested).
- [x] ISC-9: `marginal_return_vector` + `policy_posterior` (sums to 1) implemented.
- [x] ISC-10: `t_rsi` standardized distance + `certificate(value, delta)` gate implemented.
- [x] ISC-11: `bootstrap_t_rsi` is deterministic under a fixed seed (two runs identical, tested).
- [x] ISC-12: `cogant_bridge` maps firm structure→priors and re-emits a GNN summary naming all 5 channels + "ExpectedFreeEnergy".
- [x] ISC-13: `manuscript_variables.generate_variables()` emits all tokens used in prose as strings.
- [x] ISC-14: every `{{TOKEN}}` in `manuscript/*.md` is produced by the generator (no orphans).
- [x] ISC-15: `pytest tests/ --cov=src/alphacogant --cov-fail-under=90` passes with >0 collected.
- [x] ISC-16: Anti: no mocks / `unittest.mock` anywhere in `tests/`; no prose number absent from the generator.
- [x] ISC-17: Anti: the create/decay comparator is NOT green-by-construction — t-RSI changes sign across operating points (negative-control test enforces it).

## Test Strategy

| isc | type | check | threshold | tool |
| --- | --- | --- | --- | --- |
| ISC-2 | numeric | column sums | allclose 1.0 | pytest |
| ISC-5 | behavioral | belief shift sign | strong↑ | pytest |
| ISC-6/7 | invariant | total identity; KL≥0 | exact / ≥0 | pytest |
| ISC-11 | determinism | two seeded runs equal | array_equal | pytest |
| ISC-14 | drift | token set ⊆ generator keys | empty diff | grep + script |
| ISC-15 | gate | coverage | ≥90% | pytest --cov |
| ISC-16 | anti | grep mock/unittest.mock | 0 hits | grep |

## Features

| name | satisfies | depends_on | parallelizable |
| --- | --- | --- | --- |
| GNN model file | ISC-1,2 | — | yes (authored) |
| Engine (channels/gen-model/EFE/t-RSI/bridge) | ISC-3..12 | GNN file | yes (Forge) |
| manuscript_variables + scripts | ISC-13,14 | engine | no (after engine) |
| Manuscript prose | concept writeout | engine tokens | yes (authored) |
| Verification (coverage, token gate, no-mocks) | ISC-15,16 | all | no |

## Decisions

- 2026-06-23: Home = `projects/working/alphacogant/` (local-only; confidentiality invariant).
- 2026-06-23: Reduced 2-level GNN encoding for legibility/type-checkability; full continuous marginal-return formalism carried by the manuscript, not the toy matrices. Avoids over-claiming numeric fidelity to AlphaFund's proprietary books.
- 2026-06-23: Engine implemented by Forge (cross-vendor, GPT-5.4) against `src/alphacogant/SPEC.md`; primary author verifies on-disk + runs the gate (never inherits Forge's self-reported numbers — R8/GENERATOR-CHECK).

## Changelog

- conjecture: AlphaFund RSI ≡ Active Inference control; refutation route = a channel/term that has no AI counterpart; learned (so far): all five channels, the objective, and t-RSI map cleanly; the epistemic/pragmatic split *explains* AlphaFund's explore/exploit comparison rather than merely restating it.

## Verification

- ISC-15: `.venv/bin/python -m pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q` → `28 passed`, `Total coverage: 96.76%` (per-module: channels 100%, cogant_bridge 100%, free_energy 97.92%, generative_model 93.06%, manuscript_variables 100%, t_rsi 98.41%). Verified first-hand, not inherited from Forge.
- ISC-14: `scripts/z_generate_manuscript_variables.py --check` → `tokens generated: 13 | tokens used in prose: 10 | orphans: 0`. Injected sections grep for `{{` → none.
- ISC-17: `test_comparator_is_not_green_by_construction` passes — improving point create>decay (0.20>0.10), coasting create<decay (0.00<0.67); headline bootstrap t-RSI = −6.7983 (honestly negative under belief uncertainty), proving the comparator is sign-free. Caught and reverted Forge's green-by-construction `create = exp(prag); decay = max(0, create − exp(leaked))` (decay ≤ create always → t-RSI ≥ 0, certificate could never fail; its `HEADLINE_T_RSI` was a nonsensical 203.49).
- ISC-16: `grep -rn 'mock\|unittest.mock\|MagicMock' tests/` → 0 hits. Prose-vs-token: "Investments and Actuators (I, U)" / "Sensors and R&D (S, Z)" render consistent with the prose nouns.
- ISC-1/2: GNN file + `default_model()` validation (column sums ≈ 1) tested in `test_generative_model.py`.
- Demo: `scripts/run_alphacogant_demo.py` → `output/data/demo_summary.json` + `output/figures/value_decomposition.png` (epistemic vs pragmatic per channel).

## Combined-PDF render + typography (2026-06-23, E4)

Rendered the full combined PDF: `output/pdf/alphacogant_combined.pdf` — **21 pages, 0.90 MB**,
title page + 8 sections + 19 numbered Definitions + 7 figures. Verified on disk (not the
success message): 14 image objects (7 figures, none dropped), 0 literal `{{`/`{#fig`/`[@fig`
(all tokens + cross-refs resolved), 8 "fig. N" refs, headline t-RSI −4.9607 in text, body
font LMRoman9 (9pt confirmed). Visual spot-check of title + figure pages clean.

- **Smaller margins + font (user request):** `config.yaml metadata.geometry: margin=1.4cm`,
  `documentclass: extarticle`, `fontsize: 9pt`. Margins were already wired (geometry→pandoc -V);
  I added opt-in `fontsize` + `documentclass` support to `infrastructure/rendering/_pdf_combined_pandoc.py`
  (mirrors the geometry read; zero impact when unset; extarticle needed for sub-10pt). 766 rendering
  infra tests still pass.
- **Preamble:** replaced the markdown stub with a real xelatex/fontspec LaTeX preamble (math + table +
  hyperref/cleveref packages, latinmodern-math for unicode Θ/Ξ/λ, JuliaMono for code) — geometry/font
  live in config so geometry loads exactly once.
- **Cross-ref fix:** 8 prose figure references used definition syntax `Figure {#fig:X}` (would render
  literally) → corrected to pandoc-crossref `[@fig:X]`. Render flow: `z_generate_manuscript_variables.py`
  injects tokens into `output/manuscript/` (the renderer's preferred source dir); figures referenced
  `../output/figures/*.png`; render runs xelatex + pandoc-crossref.

## Comprehensive pass (2026-06-23, /workflows E4)

Added via a 10-agent workflow + my integration/verification: 7 engine-driven figures
(self-forecasting loop, AlphaFund↔AI dictionary, t-RSI create/decay densities,
certificate create-vs-decay, GNN factor graph, Θ alpha-decay, epistemic/pragmatic by
regime), a `08_formalisms.md` with 19 numbered Definitions + numbered Equations (each
bound to its AlphaFund whitepaper Definition AND the shipped code symbol), and full
token auto-injection (16 tokens, 0 orphans). Final gate (mine): 28 passed, 96.93%.

**3 integrity defects the workflow's own VERIFY agent reported `overall_pass=true` on,
caught by my visual + numeric verification:**
1. certificate figure plotted bootstrap t-RSI bars titled "fires and fails" — but ALL
   three regimes are negative (improving −4.96, coasting −617.97) so every bar FAILED,
   and the −617 outlier crushed the scale. Rebuilt to plot point-estimate create-vs-decay
   (the genuine not-green-by-construction signal: improving ADMIT, coasting REJECT).
2. prose claimed the two bootstrap t-RSIs "order oppositely / flip sign" — both negative;
   the opposite ordering is at the POINT estimate, not the bootstrap sign. Corrected §5 + §8;
   labeled −617.97 a degenerate near-zero-create-variance artifact.
3. value_by_regime plotted prior-vs-improving but the caption said improving-vs-coasting,
   and claimed "epistemic dominates when stale" — FALSE (epistemic 0.290 stale vs 0.316 fresh).
   Rebuilt to improving-vs-coasting with the honest claim (pragmatic rises when fresh;
   epistemic peak shifts Θ→Sensors). Also made trsi_densities use the canonical seed/n so its
   displayed t-RSI == the HEADLINE_T_RSI token, with a self-check assertion.

## Decisions (VERIFY addendum)

- 2026-06-23: R15 convergent-automation — Forge re-authored `t_rsi.py` and `test_t_rsi.py` mid-session ≥2× toward a green-by-construction comparator. Switched to own stable layer (rewrote both via full Write), behaviour-diffed (the defect: t-RSI structurally ≥0), re-ran the gate myself, and did NOT re-invoke Forge. See [[gotcha-crippled-comparator-authored-artifact]].
- 2026-06-23: Honest-null accepted over manufactured positive. The reduced 2-level encoding yields a modestly-negative bootstrap headline t-RSI even at the self-improving point; refused to tune `concentration`/matrices to flip the sign (seed-shopping). Documented in §5 + §6 as the instrument's integrity feature. See [[gotcha-single-seed-positive-is-cherry-pick]], [[gotcha-param-change-flips-value-prose-adjective-vs-token]].

## Performance + publish-readiness pass (2026-06-27, E4)

Picked up a large uncommitted refactor (flat modules → `model/efe/trsi/bridge/stats/tokens/viz`
subpackages, expanded manuscript, new docs/figures/tests). The blocker: the full test suite
effectively never finished — `generate_variables()` took **229s** (BOOTSTRAP_N=2560, ~7 bootstraps),
the determinism test called it twice, and there was no pytest timeout.

- **Bootstrap optimization (4.8× render; token values byte-identical).** Vectorized the EFE epistemic
  double-loop (`efe/free_energy.py::_epistemic_value`), hoisted redundant `validate_belief_map` out of
  `marginal_return_vector`, and shared one greedy trajectory across create/decay via
  `trsi/t_rsi.py::paired_bootstrap_samples` (also collapsed `compute_regime_statistics`' three same-seed
  bootstraps into one). `generate_variables()` 229s → **47.9s**; all 31→30 tokens byte-for-byte identical
  to a pre-change baseline (negative-control verified; Forge independently confirmed bit-for-bit). Added an
  equivalence test pinning `_epistemic_value` to a naive per-observation KL loop (max diff < 1e-12).
- **Test suite unblocked.** `generate_variables(n=None)` parameterizes the resample count; tests run at
  `FAST_N=32` (the contract is n-independent), production/manuscript keep BOOTSTRAP_N=2560. Added
  `pytest-timeout` (300s ceiling). Suite: **198 passed, ~98% coverage** in ~4 min.
- **Publish-readiness audit (5-dimension /workflow + cross-vendor Forge).** Fixed: a token literally named
  `TOKEN` (="AlphaCOGANT") that corrupted self-referential `{{TOKEN}}` prose into "AlphaCOGANT" in the PDF
  (HIGH — removed the token, reworded to "generated token"); a manuscript overclaim that the agent "selects
  Sensors" at the prior when the greedy policy actually **holds** (HIGH — engine-verified, reworded); the
  t-RSI "modestly negative" framing that understated a −13.26 *standardized* value whose magnitude scales as
  √n (reworded to robust-*sign* / n-dependent magnitude across 4 sites); a structurally-constant
  "exploration ratio" (2/6) made genuinely behavioral via the greedy trajectory; and the artifact-integrity
  contract (dropped a dead producer-fallback, made `--check` gate on issues, made the manifest timestamp
  SOURCE_DATE_EPOCH-aware). All HIGH/MEDIUM findings fixed and verified in the re-rendered PDF.
- **Standalone-release scaffolding** (target = public `docxology/alphacogant` + Zenodo DOI): added LICENSE
  (MIT), CITATION.cff, codemeta.json, .zenodo.json. DOI reservation + public push + Zenodo mint gated on
  explicit go-ahead (irreversible). See [[gotcha-crippled-comparator-authored-artifact]], [[gotcha-doc-claim-must-be-backed-by-shipped-code]].
