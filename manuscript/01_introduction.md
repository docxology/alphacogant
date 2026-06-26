# Introduction

## Recursive self-improvement is an economic control problem

The literature on recursive self-improvement (RSI) — from Yudkowsky's seed AI,
through Schmidhuber's Gödel machines, to intelligence-explosion dynamics and the
skeptical diminishing-returns analyses — shares one simplifying assumption: that
the economic cost of the improvement step does not threaten the system's
existence. The AlphaFund whitepaper removes that assumption. Every FLOP and every
bit of data costs money; a system that spends more on self-improvement than it
earns from the resulting improvement runs out of resources and dies. RSI is
therefore a **stochastic control problem under a survival constraint**, and a
corporation is its cleanest instance: a legal object with perpetual succession
that turns capital into improved capability and improved capability back into
capital.

AlphaFund formalizes the firm as constrained stochastic optimal control over a
production cycle. The state is a bundle of five capital **channels** — what the
firm holds (Investments), what it can see (Sensors), what it can do (Actuators),
what it knows (Parameters), and how it learns (R&D). The objective is the expected
discounted log-return on shareholders' equity subject to solvency. The controller
is a model-predictive convex program over an **Economic World Model (EWM)** — a
learned, filtration-respecting approximation to the true corporate transition law.
The firm's standing is summarized by **t-RSI**, the standardized distance between
its alpha-creation and alpha-decay rates.

## The thesis of this paper

Read the previous paragraph again with one substitution. A system that maintains a
**generative model** of a partially-observed world, infers hidden state from a
filtered history of observations, and selects actions that minimize **Expected
Free Energy** — trading off the value of preferred outcomes against the value of
information — is an **Active Inference** agent. AlphaFund's corporation is exactly
this agent. The EWM is its generative model. The five channels are its
hidden-state factors. Capital allocation is its action. The marginal-return vector
the portfolio optimizer maximizes is the negative gradient of Expected Free
Energy. And the two posteriors whose separation defines t-RSI — alpha created per
dollar and alpha decayed from the deployed book — are the **pragmatic** and
(the cost side of the) **epistemic** terms of that very free energy.

**AlphaCOGANT** makes this correspondence concrete and executable. It does three
things:

1. **Renders the firm as a generative model in GNN.** Generalized Notation
   Notation is a text specification language for Active Inference generative
   models. We write AlphaFund's five-channel EWM as a GNN model file
   (`models/alphafund_ewm.md`): channel factors, likelihood and transition
   matrices, log-preferences, and an Expected-Free-Energy objective annotated with
   its epistemic and pragmatic parts.

2. **Produces that model by the COGANT pattern.** COGANT is a codebase-to-GNN
   translator: it scans a system's structure and emits a GNN generative model of
   it. The differentiable corporation is a system whose every operational degree
   of freedom is, in AlphaFund's words, "API-complete" — a function call with a
   structured, causal record. That is precisely the substrate COGANT consumes. We
   use the COGANT translation step in miniature to map a firm description onto the
   generative model's priors.

3. **Computes inference, value, and the certificate.** A small, deterministic,
   fully-tested NumPy engine (`src/alphacogant/`) performs state inference over
   the channels, computes the Expected-Free-Energy decomposition into epistemic
   and pragmatic value, derives the marginal-return vector, and evaluates the
   t-RSI improvement certificate.

## Why route AlphaFund through Active Inference and GNN

AlphaFund already has a controller; what does the Active Inference framing add?
Two things the whitepaper itself asks for and Active Inference supplies for free.

The first is **integrity of inference**. AlphaFund spends a full section arguing
that a language model is *not* an EWM, because held-out validation only enforces
**filtration discipline** — conditioning a forecast at time *t* only on
information available at *t* — when the holdout is strictly after the training
corpus. Active Inference is built on this discipline: the generative model
factorizes over time and the agent's posterior at *t* is, by construction, a
function of the history filtration $\mathcal{F}_t = \sigma(H_t)$ and nothing
resolved later. GNN makes the factorization explicit and checkable. The
"no-peeking" property AlphaFund must bolt onto a wrapped LLM is the native
semantics of the object COGANT emits.

The second is a **single legible objective**. AlphaFund's central move is to put a
researcher hire, a data feed, a GPU, and a position in AAPL on one dollar axis by
differentiating a common objective. Expected Free Energy *is* that common axis,
and its decomposition tells you *why* a channel is worth funding: because it pays
in preferred outcomes (pragmatic) or because it sharpens the model that prices all
future outcomes (epistemic). The chronic difficulty of comparing an exploit dollar
to an explore dollar dissolves into a quantity Active Inference has been computing
for a decade.

AlphaFund's self-forecasting loop — predict (query the EWM for the marginal-return
vector), optimize (the convex inner program), execute the funded action, and fold
the realized outcome back in as the next training row — is, in Active Inference
terms, the perception–action cycle that minimizes Expected Free Energy over the
{{PLANNING_HORIZON}}-cycle horizon. [@fig:loop] renders that loop with each
stage labelled by its engine symbol, so the reader can see the whole correspondence
before the formalism arrives.

![The self-forecasting loop as the Active Inference perception–action cycle: predict via `free_energy.marginal_return_vector`, select via `free_energy.policy_posterior`, execute the funded action over {{PLANNING_HORIZON}} cycles, and fold the realized reward/loss back through `generative_model.infer_states`.](../output/figures/self_forecasting_loop.png){#fig:loop}

The remainder of the paper builds the dictionary (§2), gives the GNN-via-COGANT
realization (§3), works through generative-model inference under the firm
filtration (§4), derives the epistemic/pragmatic value split and recovers t-RSI as
the EFE certificate (§5), and argues the integrity case (§6).
