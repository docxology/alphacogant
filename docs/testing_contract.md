# Testing Contract

AlphaCOGANT uses no-mock tests. Tests run real engine computations, scripts, and
figure generators with deterministic seeds.

## Core suite

```bash
uv run --no-project pytest tests/ --cov=src/alphacogant --cov-fail-under=90 -q
```

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

## Contract tests

`tests/test_manuscript_contract.py` protects render-facing syntax and claim
boundaries. It rejects unlabeled display equations, raw LaTeX equation blocks,
hardcoded section-sign references, and overclaims that the reduced model is
identical to AlphaFund's proprietary system.

`tests/test_scripts.py` protects the artifact manifest and token generation
contracts.

`tests/test_docs_contract.py` protects the modular docs hub.

## Coverage

The project target is at least 90 percent coverage over `src/alphacogant`.
