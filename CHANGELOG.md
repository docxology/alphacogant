# Changelog

## 1.1.1 — 2026-10-16

- Select the modal joint reward/loss observation during synthetic rollouts so
  deterministic likelihoods cannot generate impossible marginal-mode pairs.

- Correct the inference reference test to admit floating-point differences up
  to 1e-15 between NumPy kernels operating on differently strided arrays.
  Linux CI exposed a one-ulp (1.1e-16) prediction difference. Coefficient
  preservation and deterministic trace replay continue to require exact equality.

## 1.1.0 — 2026-10-16

- Replace shape-only GNN summaries with lossless serialization of every A/B/C/D
  coefficient and a bounded literal-only finite-profile reader.
- Correct authored B tensors to their declared `[next,current,action]` axes;
  preserve the existing NumPy engine coefficients and research operating points.
- Export complete joint 32-state, nine-observation, six-action matrices in
  COGANT's authoritative matrix fence. Add coefficient-level and real COGANT
  runtime integration checks against a reviewed public revision.
- Add standalone `validate`, `export`, `infer`, and `verify-trace` commands.
  Observation-conditioned traces bind exact source bytes and coefficients,
  record filtering before action selection, and verify by deterministic replay.
- Stabilize information gain for point-mass Theta priors and policy softmax for
  extreme finite scores. Reject fractional/boolean indices and loop bounds,
  nonfinite bootstrap concentration, malformed confidence-interval samples,
  incomplete policy values, unknown channel keys, and unequal paired samples.
- Require manuscript sources before generating evidence tokens; installed-wheel
  callers can provide `manuscript_dir` explicitly instead of receiving silent
  zero counts.
- Make installation independent of a parent workspace, refresh the stale
  lockfile, include package metadata/specification, and add CI for Python 3.10,
  3.12 and 3.14 plus an installed-wheel smoke check.
- Apply the documented figure test marker, align release metadata, and document
  upstream scope, controller differences, artifact regeneration and release steps.

## 1.0.1 — 2026-06-27

Existing research release. Historical deposit:
[10.5281/zenodo.20977774](https://doi.org/10.5281/zenodo.20977774).
Later repository revisions added manuscript and evidence-ledger improvements.
The concept DOI [10.5281/zenodo.20976824](https://doi.org/10.5281/zenodo.20976824)
continues to identify the software across versions.
