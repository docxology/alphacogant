# Map corporate concepts to reduced Active Inference objects

The manuscript's central move is a structural mapping from AlphaFund's
recursive-corporate-self-improvement language to Active Inference. This file is
the short operational dictionary.
This is an analogy, not objective equivalence. The historical framing is recorded
in [whitepaper_concepts.md](whitepaper_concepts.md); methodological references
are identified in the [manuscript bibliography](manuscript/references.bib).

| AlphaFund term | AlphaCOGANT / Active Inference term | Engine surface |
| --- | --- | --- |
| Economic World Model | Generative model over observations and hidden state | `model/generative_model.py` |
| Five capital channels | Hidden-state factors | `model/channels.py` |
| Channel capability | Two-level latent state per factor | `default_model()` |
| Capital allocation | Discrete funding action | `ACTIONS` |
| Portfolio optimizer objective | Negative Expected Free Energy action value | `alphacogant.efe.free_energy.marginal_return_vector` |
| Broker-ledger reward | Reward observation likelihood | `A_R` |
| Forecast loss | Loss observation likelihood | `A_L` |
| Alpha creation | Mean greedy pragmatic value over passive hold | `alphacogant.trsi.t_rsi.create_rate` |
| Alpha decay | Residual Theta-refresh opportunity | `alphacogant.trsi.t_rsi.decay_rate` |
| t-RSI | Paired synthetic mean margin divided by its SE | `alphacogant.trsi.t_rsi.t_rsi` |
| Certificate | Scalar threshold predicate; no enforced gate | `alphacogant.trsi.t_rsi.certificate` |

## Discrete action values are not continuous dollar derivatives

AlphaFund's whitepaper uses a continuous dollar-vector and a marginal-return
gradient. AlphaCOGANT uses a reduced discrete action space. The engine therefore
computes a negative-EFE value for each admissible funding move, not a continuous
portfolio derivative.

## Read the argument or follow an implementation contract

- [Manuscript dictionary](manuscript/02_active_inference_mapping.md) develops the analogy.
- [EFE and policy](efe_and_policy.md) owns action scoring and Theta-information semantics.
- [Paired diagnostics](trsi_statistics.md) owns rates and the non-enforced predicate.
- [Numbered formalisms](manuscript/08_formalisms.md) distinguishes implemented objects from conceptual definitions.
