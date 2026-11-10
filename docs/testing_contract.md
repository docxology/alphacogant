# Exercise real computations and manuscript source contracts

AlphaCOGANT uses no-mock tests. Tests run real engine computations, scripts, and
figure generators with deterministic seeds.

## Run the real-engine suite

The ordered test/generation recipe is maintained in
[rendering and validation](rendering_and_validation.md); this page owns the
coverage expectations and scope of the checks.

The suite covers:

- probability normalization, constructor storage and invalid matrices
- independent informative filtering and complete joint-axis oracles
- scalar/batch Expected Free Energy invariants and explicit RNG continuation
- finite/extreme policy probabilities and paired rational statistic boundaries
- regime statistics and representable signed trajectory prefix totals
- real caller instrumentation preservation and truthful timing availability
- figure-script execution and live manuscript token computations
- saved-artifact tampering, rendered image/anchor identity and source preflight

## Protect consumed content and actual rendered products

Manuscript checks cover unique section-reference targets and bibliography keys.
They do not prove literal scientific claims or ban arbitrary source wording.
`tests/test_scripts.py` and artifact tests exercise actual generation and saved
consistency. Render-boundary tests use real files, PNGs and default-model
computations, not mocked tool output. The separate manuscript workflow runs the
real external renderers, checks generated-byte and PDF-pixel reproducibility,
and retains actual pages/figures for mandatory human inspection.

## Interpret coverage as software evidence only

The project target is at least 90 percent coverage over `src/alphacogant`.
Coverage does not establish correct observation chronology, financial calibration,
current figure generation or publication readiness. See the
[claim boundaries](claim_boundaries.md) and [artifact provenance limits](figures_and_artifacts.md).
