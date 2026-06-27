# Active Inference Dictionary

The manuscript's central move is a structural mapping from AlphaFund's
recursive-corporate-self-improvement language to Active Inference. This file is
the short operational dictionary.
The mapping is grounded in the original framing [1] and Active Inference method
sources [2, 3, 4].

| AlphaFund term | AlphaCOGANT / Active Inference term | Engine surface |
| --- | --- | --- |
| Economic World Model | Generative model over observations and hidden state | `model/generative_model.py` |
| Five capital channels | Hidden-state factors | `model/channels.py` |
| Channel capability | Two-level latent state per factor | `default_model()` |
| Capital allocation | Discrete funding action | `ACTIONS` |
| Portfolio optimizer objective | Negative Expected Free Energy action value | `efe/marginal_return_vector()` |
| Broker-ledger reward | Reward observation likelihood | `A_R` |
| Forecast loss | Loss observation likelihood | `A_L` |
| Alpha creation | Greedy policy value over passive holding | `trsi/create_rate()` |
| Alpha decay | Residual Theta staleness erosion | `trsi/decay_rate()` |
| t-RSI | Standardized create-minus-decay separation | `trsi/t_rsi()` |
| Certificate | Threshold predicate over t-RSI | `trsi/certificate()` |

## Important distinction

AlphaFund's whitepaper uses a continuous dollar-vector and a marginal-return
gradient. AlphaCOGANT uses a reduced discrete action space. The engine therefore
computes a negative-EFE value for each admissible funding move, not a continuous
portfolio derivative.

## Where this appears in the manuscript

- `manuscript/02_active_inference_mapping.md`
- `manuscript/05_epistemic_and_pragmatic_value.md`
- `manuscript/08_formalisms.md`
