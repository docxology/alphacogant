# Exercise real computations and manuscript source contracts

AlphaCOGANT uses no-mock tests. Tests run real engine computations, scripts, and
figure generators with deterministic seeds.

## Run the real-engine suite

The ordered test/generation recipe is maintained in
[rendering and validation](rendering_and_validation.md); this page owns the
coverage expectations and scope of the checks.

The suite covers:

- probability normalization and invalid matrices
- inference and belief validation
- Expected Free Energy invariants
- policy posterior normalization
- t-RSI comparator behavior
- regime statistics
- simulation trajectories
- figure-script execution
- manuscript token coverage
- claim-boundary wording
- docs hub structure

## Protect render-facing syntax and artifact contracts

`tests/test_manuscript_contract.py` protects render-facing syntax and claim
boundaries. It rejects unlabeled display equations, raw LaTeX equation blocks,
hardcoded section-sign references, and overclaims that the reduced model is
identical to AlphaFund's proprietary system.

`tests/test_scripts.py` protects the artifact manifest and token generation
contracts.

`tests/test_docs_contract.py` protects the modular docs hub.

## Interpret coverage as software evidence only

The project target is at least 90 percent coverage over `src/alphacogant`.
Coverage does not establish correct observation chronology, financial calibration,
current figure generation or publication readiness. See the
[claim boundaries](claim_boundaries.md) and [artifact provenance limits](figures_and_artifacts.md).
