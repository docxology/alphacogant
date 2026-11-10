# Fixed A/B/C/D and factorized observation filtering

`src/alphacogant/model/generative_model.py` owns the reduced Economic World
Model. The model is deliberately small: five factors, two levels per factor, six
actions, and two observation channels.
The matrices are illustrative authored parameters, not fitted corporate data.
See the [manuscript references](manuscript/references.bib) for methodological context.

## Likelihood, transition, preference and prior tensors

| Symbol | Engine field | Meaning |
| --- | --- | --- |
| `A_R` | reward likelihood | Maps Investments, Actuators, and Theta to reward observations. |
| `A_L` | loss likelihood | Maps Sensors and Theta to predictive-loss observations. |
| `B_k` | transition tensor per channel | Moves each channel under each funding action. |
| `C_R` | reward preference vector | Log-preference over reward buckets. |
| `C_L` | loss preference vector | Log-preference over loss buckets. |
| `D_k` | prior belief per channel | Initial belief over weak/strong channel state. |

## Admit finite probability arrays and complete beliefs

- Probability vectors must be finite, non-negative, and sum to one.
- Probability columns in likelihood and transition tensors must sum to one.
- Belief maps must include every channel and normalize before downstream use.
- Invalid observation indices are rejected before inference.

`EconomicWorldModel` converts valid integer/list coefficients to NumPy floating
storage once, without renormalizing accepted values. Probability admission uses
the implementation's floating tolerance; it is not an input-repair service.
B/D mappings are copied, while compatible float arrays/views remain shared with
their original writeability. Callers must preserve validated invariants after mutation.
`belief_prior(model)` returns detached copies of that model's D vectors.

## Filter one observation pair with a single factorized sweep

`infer_states()` performs one factorized filtering sweep. Each likelihood message
is averaged over the other prior factors; the resulting product discards
posterior dependence. Coupled likelihoods can change exact individual marginals,
even with independent transitions. Log-domain normalization supports rare
positive evidence; truly zero-support observations raise `ValueError`.

`bridge.inference.run_inference` repeats filter/score/select/predict for supplied
current observations. Timestamps and training-data cutoffs are not part of this
interface: correctly timed inputs remain the caller's responsibility. See the
[trace contract](cogant_bridge.md#source-bound-traces) for source binding and replay.
The [prediction-only diagnostic](trsi_statistics.md) does not filter observations.

## Interpret capability states without claiming learned coefficients

The model is not calibrated to AlphaFund proprietary surfaces. It is an authored
reduced Active Inference encoding used to examine a conceptual correspondence
and generate synthetic diagnostics. Z has no likelihood or cross-transition
effect and cannot affect EFE. Theta is a capability state, not learned matrix
coefficients; A/B/C/D remain fixed throughout a run.
