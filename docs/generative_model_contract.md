# Generative Model Contract

`src/alphacogant/model/generative_model.py` owns the reduced Economic World
Model. The model is deliberately small: five factors, two levels per factor, six
actions, and two observation channels.
This contract is the source-owned executable form of the model in the whitepaper's
filtered generative framing [1, 2, 3, 4].

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

`infer_states()` performs a one-step Bayesian filtering update from reward and
loss observations to posterior channel beliefs. For the current factor graph, the
mean-field update is exact per factor given the other factors. If future work adds
coupled transitions or higher-cardinality state, this exactness claim must be
revised [2, 4].

## What the model is not

The model is not a calibrated copy of AlphaFund proprietary surfaces. It is an
illustrative reduced Active Inference encoding used to test the correspondence,
the sign-discriminating certificate, and the source-owned manuscript pipeline.
