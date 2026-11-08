# Generative Model Contract

`src/alphacogant/model/generative_model.py` owns the reduced Economic World
Model. The model is deliberately small: five factors, two levels per factor, six
actions, and two observation channels.
The matrices are illustrative authored parameters, not fitted corporate data.
See the [manuscript references](manuscript/references.bib) for methodological context.

## Tensors

| Symbol | Engine field | Meaning |
| --- | --- | --- |
| `A_R` | reward likelihood | Maps Investments, Actuators, and Theta to reward observations. |
| `A_L` | loss likelihood | Maps Sensors and Theta to predictive-loss observations. |
| `B_k` | transition tensor per channel | Moves each channel under each funding action. |
| `C_R` | reward preference vector | Log-preference over reward buckets. |
| `C_L` | loss preference vector | Log-preference over loss buckets. |
| `D_k` | prior belief per channel | Initial belief over weak/strong channel state. |

## Validation rules

- Probability vectors must be finite, non-negative, and sum to one.
- Probability columns in likelihood and transition tensors must sum to one.
- Belief maps must include every channel and normalize before downstream use.
- Invalid observation indices are rejected before inference.

## Inference contract

`infer_states()` performs one factorized filtering sweep. Each likelihood message
is averaged over the other prior factors; the resulting product discards
posterior dependence. Coupled likelihoods can change exact individual marginals,
even with independent transitions. Log-domain normalization supports rare
positive evidence; truly zero-support observations raise `ValueError`.

## What the model is not

The model is not a calibrated copy of AlphaFund proprietary surfaces. It is an
illustrative reduced Active Inference encoding used to test the correspondence,
the sign-discriminating certificate, and the source-owned manuscript pipeline.
Z has no likelihood or cross-transition effect and cannot affect EFE in this
encoding. Theta is a capability state, not learned matrix coefficients.
