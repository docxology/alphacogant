# The AlphaFund ↔ Active Inference dictionary

Active Inference casts any adaptive system as an agent that holds a generative
model of how hidden causes produce sensory data, infers those causes by minimizing
variational free energy, and selects actions that minimize *expected* free energy
over a planning horizon. Below, each AlphaFund construct is matched to its Active
Inference counterpart. The match is structural, not metaphorical: the equations
coincide.

| AlphaFund construct | Active Inference object | Correspondence |
| --- | --- | --- |
| Corporation tuple $\Xi_t = (I,S,U,\Theta,Z)$ | Factorized hidden state $s_t$ | Five capital channels = five hidden-state factors |
| Environment $E_t$ (prices, flow, macro) | External hidden causes | The part of the world the firm conditions on but does not fully control |
| Economic World Model $\widehat{W}_t$ | Generative model $P(o,s'\mid s,a)$ | Learned, filtration-respecting next-cycle law |
| Firm history $H_t$; channel histories $H_t^k$ | Observation sequence $o_{0:t}$ | The evidence inference conditions on |
| Firm filtration $\mathcal{F}_t = \sigma(H_t)$ | Belief-update information set | "No-peeking" — posterior at $t$ depends only on $\mathcal{F}_t$ |
| Action vector $a_t$ (dollars per channel) | Control state / policy $\pi$ | Capital allocation = the agent's action |
| Cumulative objective $J_t$ (expected $\sum \log$-equity) | Negative Expected Free Energy (pragmatic part) | Discounted log-return = preference satisfaction |
| Marginal-return vector $g_t = \partial J_t/\partial a_t$ | $-\partial G/\partial a$ | Per-channel return = negative EFE gradient |
| Equimarginal identity $\hat g^k_t/\sigma^k_t = \lambda^*_{S,t}$ | Precision-weighted policy optimum | Risk-adjusted shadow price of capital |
| Sensors / R&D returns (data-scaling, search laws) | **Epistemic value** (information gain) | What reduces EWM predictive loss |
| Investments / Actuators returns (broker ledger, Sharpe) | **Pragmatic value** (expected utility) | What realizes preferred outcomes now |
| t-RSI = std. gap(create, decay) | Thresholded EFE-improvement statistic | Certificate gating each commit |
| Certificate of monotone improvement | Admissibility gate on a model update | Admit update iff value clears a margin |
| Drift detection + refit | Precision / model revision under surprise | Refit when observations leave support |

[@fig:dictionary] lays the same correspondence out as a two-column dictionary
panel, pairing each AlphaFund construct with its Active Inference object across the
{{NUM_CHANNELS}} channels and {{NUM_ACTIONS}} actions of the engine — the equations
coincide, so the panel is a map of where to find each AlphaFund symbol inside
`src/alphacogant`.

![The AlphaFund ↔ Active Inference dictionary: each whitepaper construct (equity, reward, EWM, filtration, marginal-return vector, certificate) paired with its Active Inference object and the realizing engine symbol across {{NUM_CHANNELS}} channels and {{NUM_ACTIONS}} actions, with each pairing's value computed by `free_energy.marginal_return_vector`.](../output/figures/aif_dictionary.png){#fig:dictionary}

## The firm as a generative-model-carrying agent

The corporation "sees the world only through its sensors," so both the environment
$E_t$ and the firm's own state $\Xi_t$ enter the EWM as **posteriors over noisy
observations**, never as latent ground truth. This is the defining posture of a
partially-observed Active Inference agent: there is a hidden state, a likelihood
mapping that state to observations, and a transition mapping that state forward
under action. AlphaFund's channel histories $H_t^k$ — the $(o^k_\tau, a^k_\tau,
R_{\tau+1})$ rows the firm fits its row-laws on — are exactly the per-factor
evidence streams an Active Inference agent accumulates.

The EWM's structural promise is that it is the **only** model of the future the
firm has access to, so improving it is itself an allocation with first-order
effect on the objective. In Active Inference terms: the generative model is
parameterized (by $\Theta$), those parameters are themselves hidden states subject
to inference and to control, and the value of acting to improve them is
quantified by the same expected-free-energy functional that values every other
action. AlphaFund's insistence that "every input that lowers predictive loss is
priced in dollars on the firm's books" is the economic image of Active Inference's
unification of perception, learning, and action under one objective.

## Channel-specific world models = factorized generative models

AlphaFund decomposes the joint EWM into **channel-specific world models**
$\widehat{W}_t^k$, each trained on its own channel history — a scaling law, a
market-impact curve, a refit-decay model, a search law. It is explicit that this
is a practical approximation: "cross-channel coupling re-enters when the
controller composes the rows." This is precisely the **mean-field / structured
factorization** of a generative model in Active Inference: the joint is
approximated as a product of per-factor distributions for tractable inference, and
coupling re-enters at the policy-evaluation step where Expected Free Energy is
computed over the joint predicted outcome. AlphaFund's supermodular cross-partials
— "a marginal dollar on channel $j$ raises the marginal value of a dollar on
channel $k$" — are the coupling terms that a fully factorized model drops and the
EFE computation restores.

The GNN model file in §3 encodes exactly this: five factor blocks with per-factor
transition matrices $B_k$, two likelihood matrices $A_R, A_L$ that couple factors
at the observation, and an Expected-Free-Energy block whose epistemic and
pragmatic parts compose the rows back together.
