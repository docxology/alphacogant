# Corporate concepts and their reduced Active Inference counterparts {#sec:dictionary}

Active Inference casts any adaptive system as an agent that holds a generative
model of how hidden causes produce sensory data, infers those causes by minimizing
variational free energy, and selects actions that minimize *expected* free energy
over a planning horizon [@friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis]. Below, each AlphaFund construct is matched to its Active
Inference-inspired reduced object. The mapping is an analogy with explicit
omissions, not an identity of the economic and EFE equations.

| AlphaFund construct | Active Inference object | Correspondence |
| --- | --- | --- |
| Corporation tuple $\Xi_t = (I,S,U,\Theta,Z)$ | Factorized hidden state $s_t$ | Five capital channels = five hidden-state factors |
| Environment $E_t$ (prices, flow, macro) | External hidden causes | The part of the world the firm conditions on but does not fully control |
| Economic World Model $\widehat{W}_t$ | Generative model $P(o,s'\mid s,a)$ | Historical learned law; local matrices are fixed |
| Firm history $H_t$; channel histories $H_t^k$ | Observation sequence $o_{0:t}$ | Caller-supplied evidence, not authenticated corporate records |
| Firm filtration $\mathcal{F}_t = \sigma(H_t)$ | Belief-update information set | Intended no-peeking constraint; local ordering cannot validate input chronology |
| Action vector $a_t$ (dollars per channel) | Discrete control action | One channel funded or hold; no continuous dollar vector |
| Cumulative objective $J_t$ (expected $\sum \log$-equity) | Expected reward/loss preference analogy | Authored scores, not measured log-return or an objective identity |
| Marginal-return vector $g_t = \partial J_t/\partial a_t$ | Negative-EFE action value | Continuous marginal return becomes a discrete funding-move score |
| Equimarginal identity $\hat g^k_t/\sigma^k_t = \lambda^*_{S,t}$ | Conceptual comparison only | Policy precision is not a solved risk-adjusted shadow price |
| Sensors / R&D returns | Conceptual epistemic roles | Implemented information gain targets Theta, not fitted scaling laws |
| Investments / Actuators returns | Expected reward/loss preference | No broker ledger or Sharpe calibration |
| t-RSI = std. gap(create, decay) | Paired synthetic diagnostic | Computed separately from EFE action selection |
| Certificate of monotone improvement | Scalar threshold predicate only | No deployed-update admission mechanism |
| Drift detection + refit | Not implemented | Fixed matrices; no learned parameter update |

[@fig:dictionary] maps the {{NUM_CHANNELS}} corporate channel names to their
finite-state meanings and actual likelihood dependencies. A separate decision-term
key explains EFE, priors and the synthetic diagnostic across {{NUM_ACTIONS}}
discrete actions. This is a structural schematic, not a numerical action-value
comparison; executable support and limitations are described separately below.

![Channel/state/dependency dictionary and decision-term key for {{NUM_CHANNELS}} factors and {{NUM_ACTIONS}} discrete actions. Reward depends on I/U/Theta and loss on S/Theta; Z has no likelihood edge. The conceptual corporate names do not imply coefficient learning, financial calibration or an enforced t-RSI admission gate. This schematic displays no numerical action-value examples.](../output/figures/aif_dictionary.png){#fig:dictionary}

## Hidden corporate state motivates a generative-model-carrying agent

The corporation "sees the world only through its sensors," so both the environment
$E_t$ and the firm's own state $\Xi_t$ enter the EWM as **posteriors over noisy
observations**, never as latent ground truth. This is the defining posture of a
partially-observed Active Inference agent: there is a hidden state, a likelihood
mapping that state to observations, and a transition mapping that state forward
under action. AlphaFund's channel histories $H_t^k$ — the $(o^k_\tau, a^k_\tau,
R_{\tau+1})$ rows the historical account fits its row-laws on — motivate per-factor
evidence streams; the local engine does not fit those laws or acquire that history.

In the reduced engine, Theta is a latent capability label, not the numerical
parameters of $A$ or $B$. Funding Theta changes the distribution over this label;
it does not refit coefficients. Information gain about the label is therefore
not demonstrated improvement in a learned EWM [@westenhaver2026rsi].

## Factorized channel models retain tractability, not exact posteriors

AlphaFund decomposes the joint EWM into **channel-specific world models** [@dacosta2020synthesis]
$\widehat{W}_t^k$, each trained on its own channel history — a scaling law, a
market-impact curve, a refit-decay model, a search law. It is explicit that this
is a practical approximation: "cross-channel coupling re-enters when the
controller composes the rows." Our factorized representation is similarly a
tractability choice [@vandemeent2021ppl]. Coupled observation likelihoods enter
filtering and EFE evaluation, but the retained product belief discards posterior
dependence. Likelihood coupling does not prove economic supermodularity or restore
omitted transition interactions [@friston2015epistemic; @dacosta2020synthesis; @milgrom1990].

The GNN model file in [@sec:gnn] encodes exactly this: five factor blocks with per-factor
transition matrices $B_k$, two likelihood matrices $A_R, A_L$ that couple factors
at the observation, and an Expected-Free-Energy block whose epistemic and
pragmatic parts compose the rows back together.
