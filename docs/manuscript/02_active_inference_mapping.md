# The AlphaFund ↔ Active Inference dictionary {#sec:dictionary}

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
| Economic World Model $\widehat{W}_t$ | Generative model $P(o,s'\mid s,a)$ | Learned, filtration-respecting next-cycle law |
| Firm history $H_t$; channel histories $H_t^k$ | Observation sequence $o_{0:t}$ | The evidence inference conditions on |
| Firm filtration $\mathcal{F}_t = \sigma(H_t)$ | Belief-update information set | "No-peeking" — posterior at $t$ depends only on $\mathcal{F}_t$ |
| Action vector $a_t$ (dollars per channel) | Control state / policy $\pi$ | Capital allocation = the agent's action |
| Cumulative objective $J_t$ (expected $\sum \log$-equity) | Negative Expected Free Energy (pragmatic part) | Discounted log-return = preference satisfaction |
| Marginal-return vector $g_t = \partial J_t/\partial a_t$ | Negative-EFE action value | Continuous marginal return becomes a discrete funding-move score |
| Equimarginal identity $\hat g^k_t/\sigma^k_t = \lambda^*_{S,t}$ | Conceptual comparison only | Policy precision is not a solved risk-adjusted shadow price |
| Sensors / R&D returns | Conceptual epistemic roles | Implemented information gain targets Theta, not fitted scaling laws |
| Investments / Actuators returns | Expected reward/loss preference | No broker ledger or Sharpe calibration |
| t-RSI = std. gap(create, decay) | Paired synthetic diagnostic | Computed separately from EFE action selection |
| Certificate of monotone improvement | Scalar threshold predicate only | No deployed-update admission mechanism |
| Drift detection + refit | Not implemented | Fixed matrices; no learned parameter update |

[@fig:dictionary] lays the same correspondence out as a two-column dictionary
panel, pairing each AlphaFund construct with its Active Inference object across the
{{NUM_CHANNELS}} channels and {{NUM_ACTIONS}} actions of the engine. The panel
locates the conceptual pairings; executable support and limitations are described
separately below.

![Conceptual AlphaFund–Active Inference dictionary with explicit implementation boundaries across {{NUM_CHANNELS}} factors and {{NUM_ACTIONS}} actions. Only the annotated example action values are computed from the finite engine; the conceptual pairings are not numerical outputs or implemented financial guarantees.](../output/figures/aif_dictionary.png){#fig:dictionary}

## The firm as a generative-model-carrying agent

The corporation "sees the world only through its sensors," so both the environment
$E_t$ and the firm's own state $\Xi_t$ enter the EWM as **posteriors over noisy
observations**, never as latent ground truth. This is the defining posture of a
partially-observed Active Inference agent: there is a hidden state, a likelihood
mapping that state to observations, and a transition mapping that state forward
under action. AlphaFund's channel histories $H_t^k$ — the $(o^k_\tau, a^k_\tau,
R_{\tau+1})$ rows the firm fits its row-laws on — are exactly the per-factor
evidence streams an Active Inference agent accumulates.

In the reduced engine, Theta is a latent capability label, not the numerical
parameters of $A$ or $B$. Funding Theta changes the distribution over this label;
it does not refit coefficients. Information gain about the label is therefore
not demonstrated improvement in a learned EWM [@westenhaver2026rsi].

## Channel-specific world models = factorized generative models

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
