# Generative-model inference under the firm filtration {#sec:inference}

## The generative model

The AlphaCOGANT generative model factorizes the next cycle as

$$
P(o_{R,t}, o_{L,t}, s_{t+1} \mid s_t, a_t)
  = A_R(o_{R,t} \mid s_t^I, s_t^U, s_t^\Theta)\;
    A_L(o_{L,t} \mid s_t^S, s_t^\Theta)\;
    \prod_{k \in \{I,S,U,\Theta,Z\}} B_k\!\left(s^k_{t+1} \mid s^k_t, a_t\right).
$$ {#eq:generative-factorization}

This is AlphaFund's "true corporate transition" $W$ — a property of the world the
firm cannot access — approximated by the firm's learned $\widehat{W}_t$, the EWM.
In this form it is a finite-state, controlled, partially-observed model in the
same sense as active-inference POMDPs [@friston2010freeenergy; @friston2015epistemic; @kaelbling1998pomdp; @friston2017process].
The likelihoods $A_R, A_L$ are how the latent corporate state generates the
observable broker ledger and forecast-evaluation panel; the transitions $B_k$ are
how a capital-allocation action moves each channel. Inference is the inverse
problem: recover a posterior $q(s_t)$ over the five channels from the observed
reward/loss history.

## Inference is the firm reading its own state

Because the firm "sees the world only through its sensors," it never observes
$\Xi_t$ directly; it observes $o_R, o_L$ and infers the channel capabilities that
best explain them. In the engine, `generative_model.infer_states` performs one
single-sweep factorized update from a bucketed reward/loss pair. Each channel's
likelihood is averaged over the other prior factors, then normalized. Coupled
likelihoods induce posterior dependence even with independent transitions, so the
retained product is an approximation, not exact joint filtering. Independent
joint enumeration in the numerical tests exposes this distinction
[@vandemeent2021ppl; @kaelbling1998pomdp; @friston2017process].

A worked reading: a high-reward, low-loss observation is most probable when the
production channels and the EWM are strong and fresh, so the posterior shifts mass
toward `strong` on $I, U, \Theta$ and toward `rich` on $S$. The firm has *inferred*
that it is in a compounding regime — not because it was told, but because the
ledger and the panel are jointly improbable under any weaker state. This is the
self-forecasting loop's "predict" step: query the world model for the current
state and its forward law before choosing an allocation.

The transition $B_\Theta$ illustrates the hypothesis that deployed parameters
decay toward stale unless refreshed; it is not an empirical estimate obtained by
this repository. [@fig:thetadecay]
traces the belief in fresh $\Theta$ forward under the two policies: under `hold` the
freshness mass bleeds away each cycle through the decay-leak probability
{{B_THETA_LEAK_PROB}}, while under `fund_Theta` it is pulled back toward fresh — the
mechanism that makes refit a first-order allocation rather than maintenance overhead [@westenhaver2026rsi].

![The $B_\Theta$ alpha-decay law: belief in fresh parameters traced over {{PLANNING_HORIZON}} cycles under `hold` (freshness decays via leak probability {{B_THETA_LEAK_PROB}}) versus `fund_Theta` (freshness restored), with the transition read from `generative_model.default_model`.](../output/figures/theta_decay.png){#fig:thetadecay}

## Inspectable step ordering and caller-owned chronology

AlphaFund's sharpest methodological claim is that an LLM is **not** an EWM, because
a model trained on a static corpus can "mix documents from before and after the
event it is asked to predict," so its context can contain future information.
Held-out validation only enforces the no-peeking property when the holdout is
strictly chronologically after the entire training corpus — a window that shrinks
as internet-scale models train on ever-more-recent data, leaving little data for
validation and high measured uncertainty.

In the local loop this discipline is an explicit operation order. The intended
filtration $\mathcal{F}_t = \sigma(H_t)$ is the information available at decision
time. The supplied current observation is filtered, an action is scored, and
$B_k$ predicts the next belief. Later observations cannot alter earlier records.
However, timestamps and training-data provenance are absent: a caller can supply
future-derived values as current observations, and the graph cannot detect that.
The motivating paper contrasts two loss functions —

$$
\mathcal{L}_{\text{LLM}}(\Theta) = \sum_i \ell\big(\hat p_\Theta(x_i \mid \text{ctx}_i), x_i\big)
\quad\text{(permutation-invariant over documents)},
$$ {#eq:llm-loss}

$$
\mathcal{L}_{\text{EWM}}(\Theta) = \sum_{\tau \in I_{\text{eval}}} \ell\big(\widehat{P}_\tau(o_{\tau+1}, R_{\tau+1} \mid \mathcal{F}_\tau, a_\tau), (o_{\tau+1}, R_{\tau+1})\big)
\quad\text{(information order matters)}
$$ {#eq:ewm-loss}

— are, respectively, a likelihood with no temporal index and the **predictive**
likelihood one might use in a time-ordered EWM [@westenhaver2026rsi; @friston2010freeenergy; @kaelbling1998pomdp; @friston2017process].
The local engine does not fit either loss. Its finite GNN reader validates model
arrays, not chronological data availability or the absence of training leakage.

A note AlphaFund itself makes survives the translation: a general LLM *wrapped* in
a strict post-cutoff evaluation harness can still serve as a component of the
generative model (e.g. a likelihood or proposal). The categorical claim is about
the bare model used as the primary EWM, not about a properly-filtered subroutine —
and in GNN such a subroutine is just another typed block whose connections are
checked.
