# Introduction {#sec:introduction}

## Recursive self-improvement is an economic control problem

The literature on recursive self-improvement (RSI) — from Yudkowsky's seed AI [@yudkowsky2008],
through Schmidhuber's Gödel Machines [@schmidhuber2007godel], to intelligence-explosion dynamics and
diminishing-returns analyses [@schmidhuber2007godel; @yudkowsky2008] — often studies self-improvement where
compute costs are secondary to asymptotics. The AlphaFund whitepaper removes that
assumption by placing survival and capital constraints into the optimization
problem [@westenhaver2026rsi; @kaelbling1998pomdp; @friston2017process]. Every FLOP and every bit of data costs money; a system that
spends more on self-improvement than it earns from the resulting improvement runs
out of resources and dies. AlphaFund therefore recasts RSI as constrained control
survival test and an auditable certificate [@westenhaver2026rsi; @dacosta2020synthesis]. RSI is therefore a
**stochastic control problem under a survival constraint** [@westenhaver2026rsi; @kaelbling1998pomdp; @friston2017process], and a corporation is
its cleanest instance: a legal object with perpetual succession that turns capital into
improved capability and improved capability back into capital [@coase1937].

In this light, the economics-of-modern-manufacturing framing is useful, because it
treats RSI economics as constrained multi-factor coordination where capital budget
structure, not just return maximization, determines what a firm can credibly fund [@milgrom1990].

AlphaFund formalizes the firm as constrained stochastic optimal control over a
production cycle. The state is a bundle of five capital **channels** — what the
firm holds (Investments), what it can see (Sensors), what it can do (Actuators),
what it knows (Parameters), and how it learns (R&D). The objective is the expected
discounted log-return on shareholders' equity subject to solvency [@kelly1956]. The controller is
a model-predictive convex program over an **Economic World Model (EWM)** — a learned,
filtration-respecting approximation to the true corporate transition law [@westenhaver2026rsi; @friston2017process]. The
firm's standing is summarized by **t-RSI**, the standardized distance between its
alpha-creation and alpha-decay rates.

## The thesis of this paper

Read the previous paragraph again with one substitution. A system that maintains a
**generative model** of a partially-observed world, infers hidden state from a
filtered history of observations, and selects actions that minimize **Expected Free
Energy** — trading off the value of preferred outcomes against the value of
information — is an **Active Inference** agent [@friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis; @friedman2026gnn; @friston2017process]. AlphaFund's
corporation can be represented in that form: the EWM is its generative model, the
five channels are its hidden-state factors, capital allocation is its action, and the
marginal-return vector the portfolio optimizer maximizes is the negative-EFE value of
each admissible funding move [@friston2015epistemic; @dacosta2020synthesis]. The two posteriors whose separation defines
t-RSI — alpha created per dollar and alpha decayed from the deployed book — are
represented here as the **pragmatic** and (the cost side of the) **epistemic** terms of
that free energy [@friston2010freeenergy; @dacosta2020synthesis].

**AlphaCOGANT** makes this correspondence concrete and executable. It does three
things:

1. **Renders the firm as a generative model in GNN.** Generalized Notation
   Notation is a text specification language for Active Inference generative
   models. We write AlphaFund's five-channel EWM as a GNN model file
   (`models/alphafund_ewm.md`): channel factors, likelihood and transition
   matrices, log-preferences, and an Expected-Free-Energy objective annotated with
   its epistemic and pragmatic parts [@friedman2026gnn]. The GNN pipeline is explicit about
   typed parsing, validation, rendering, and executable exports, so the model is
   readable and checkable before inference [@friedman2026gnn]. In this representation, the firm's
   temporal assumptions are explicit graph objects and therefore directly reviewable.

2. **Produces that model by the COGANT pattern.** COGANT is a codebase-to-GNN
   translator: it scans a system's structure and emits a GNN generative model of
   it. It also emits a reverse map from generated artifacts, which we use as a
   provenance check [@friedman2026cogant]. The differentiable corporation is a system whose every
   operational degree
   of freedom is, in AlphaFund's words, "API-complete" — a function call with a
   structured, causal record. That is precisely the substrate COGANT consumes. We
   use the COGANT translation step in miniature [@friedman2026cogant] to map a firm description onto
   the generative model's priors and maintain a reproducible model provenance chain.

3. **Computes inference, value, and the certificate.** A small, deterministic,
   fully-tested NumPy engine (`src/alphacogant/`) performs state inference over
   the channels, computes the Expected-Free-Energy decomposition into epistemic
   and pragmatic value [@friston2015epistemic; @dacosta2020synthesis], derives the marginal-return vector, and evaluates
   the t-RSI improvement certificate.

## Why route AlphaFund through Active Inference and GNN

AlphaFund already has a controller; what does the Active Inference framing add?
Two things the whitepaper itself asks for and Active Inference makes explicit in
this reduced model.

The first is **integrity of inference**. AlphaFund spends a full section arguing
that a language model is *not* an EWM, because held-out validation only enforces
**filtration discipline** — conditioning a forecast at time *t* only on
information available at *t* — when the holdout is strictly after the training
corpus [@westenhaver2026rsi]. Active Inference is built on this discipline: the generative model
factorizes over time and the agent's posterior at *t* is, by construction, a
function of the history filtration $\mathcal{F}_t = \sigma(H_t)$ and nothing
resolved later [@friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis; @kaelbling1998pomdp; @friston2017process]. This is where COGANT and the GNN graph contract are doing
most of the heavy lifting [@friedman2026gnn; @friedman2026cogant]. GNN makes this factorization explicit and
checkable [@friedman2026gnn].
The "no-peeking" property AlphaFund must bolt onto a wrapped LLM is the native
semantics of the object COGANT emits [@friedman2026cogant; @friston2017process]. This is the same hard constraint that
separates time-aware economic forecasting from post-fitted narrative claims [@westenhaver2026rsi; @friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis; @kaelbling1998pomdp].

The second is a **single legible objective**. AlphaFund's central move is to put a
researcher hire, a data feed, a GPU, and a position in AAPL on one dollar axis by
differentiating a common objective. Expected Free Energy *is* that common axis, and
its decomposition tells you *why* a channel is worth funding: because it pays in
preferred outcomes (pragmatic) or because it sharpens the model that prices all
future outcomes (epistemic) [@friston2015epistemic; @dacosta2020synthesis]. The chronic difficulty of comparing an
explore dollar to an exploit dollar dissolves into the same quantity Active
Inference has formalized for the past decade as a single value decomposition [@friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis; @lattimore2020bandits].

AlphaFund's self-forecasting loop — predict (query the EWM for the marginal-return
vector), optimize (the convex inner program), execute the funded action, and fold
the realized outcome back in as the next training row — is, in Active Inference
terms, the perception–action cycle that minimizes Expected Free Energy over the
{{PLANNING_HORIZON}}-cycle horizon [@kaelbling1998pomdp]. [@fig:loop] renders that loop with each
stage labelled by its engine symbol, so the reader can see the whole correspondence
before the formalism arrives [@friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis; @friston2017process].

![The self-forecasting loop as the Active Inference perception–action cycle: predict via `free_energy.marginal_return_vector`, select via `free_energy.policy_posterior`, execute the funded action over {{PLANNING_HORIZON}} cycles, and fold the realized reward/loss back through `generative_model.infer_states`.](../output/figures/self_forecasting_loop.png){#fig:loop}

The remainder of the paper builds the dictionary ([@sec:dictionary]), gives the
GNN-via-COGANT realization ([@sec:gnn]), works through generative-model inference
under the firm filtration ([@sec:inference]), derives the epistemic/pragmatic value
split and recovers t-RSI as the EFE certificate ([@sec:value]), and argues the
integrity case ([@sec:integrity]).
