# Corporate RSI motivates a reduced Active Inference analogy {#sec:introduction}

## Economic survival constrains the historical RSI argument

The literature on recursive self-improvement (RSI) — from Yudkowsky's seed AI [@yudkowsky2008],
through Schmidhuber's Gödel Machines [@schmidhuber2007godel], to intelligence-explosion dynamics and
diminishing-returns analyses [@schmidhuber2007godel; @yudkowsky2008] — often studies self-improvement where
compute costs are secondary to asymptotics. The AlphaFund whitepaper removes that
assumption by placing survival and capital constraints into the optimization
problem [@westenhaver2026rsi; @kaelbling1998pomdp; @friston2017process]. Every FLOP and every bit of data costs money; a system that
spends more on self-improvement than it earns from the resulting improvement runs
out of resources and dies. AlphaFund therefore recasts RSI as constrained control
with a survival test and an auditable certificate [@westenhaver2026rsi; @dacosta2020synthesis].
In that account, RSI is a **stochastic control problem under a survival constraint**
[@westenhaver2026rsi; @kaelbling1998pomdp; @friston2017process], and a corporation is
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

## The executable thesis is a bounded analogy, not objective equivalence

Read the previous paragraph again with one substitution. A system that maintains a
**generative model** of a partially-observed world, infers hidden state from a
filtered history of observations, and selects actions that minimize **Expected Free
Energy** — trading off the value of preferred outcomes against the value of
information — is an **Active Inference** agent [@friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis; @friedman2026gnn; @friston2017process]. AlphaFund's
corporation motivates a reduced representation: the EWM becomes a fixed
generative model, the channels become hidden-state factors, and allocation
becomes a discrete action [@friston2015epistemic; @dacosta2020synthesis].
Negative EFE is an illustrative action score, not an established dollar-return
gradient. The create/decay diagnostic is computed from pragmatic trajectories;
its decay term is not epistemic information gain.

**AlphaCOGANT** makes this correspondence concrete and executable. It does three
things:

1. **Renders the firm as a generative model in GNN.** Generalized Notation
   Notation is a text specification language for Active Inference generative
   models. We write AlphaFund's five-channel EWM as a GNN model file
   (`models/alphafund_ewm.md`): channel factors, likelihood and transition
   matrices, log-preferences, and an Expected-Free-Energy objective annotated with
   its epistemic and pragmatic parts [@friedman2026gnn]. The local parser checks
   fixed declarations, dependencies and numerical coefficients before inference.
   Time and ontology text remain descriptive metadata, not an executed temporal
   type system or proof of observation availability.

2. **Uses a COGANT-inspired structure-to-prior mapping.** A supplied firm
   description maps deterministically to channel priors; no proprietary codebase
   or causal financial record is extracted. The authored coefficients round-trip
   through a bounded GNN profile and export complete joint matrices for a pinned
   real COGANT runtime [@friedman2026cogant]. Interoperability does not imply
   equal filtering or policy objectives across engines.

3. **Computes inference, value, and a threshold diagnostic.** The NumPy engine
   performs a single factorized filtering sweep, scores one-step EFE, and
   separately computes paired create/decay trajectory summaries
   [@friston2015epistemic; @dacosta2020synthesis]. No model-update acceptance
   mechanism is implemented.

## Inspectable inference and a legible objective motivate the translation

AlphaFund already has a controller; what does the Active Inference framing add?
Two things the whitepaper itself asks for and Active Inference makes explicit in
this reduced model.

The first is **integrity of inference**. AlphaFund spends a full section arguing
that a language model is *not* an EWM, because held-out validation only enforces
**filtration discipline** — conditioning a forecast at time *t* only on
information available at *t* — when the holdout is strictly after the training
corpus [@westenhaver2026rsi]. The local inference loop filters the supplied current
observation before selecting an action; a future supplied observation does not
change earlier records. This inspectable order is narrower than verified
filtration integrity: the engine has no timestamps or training-data provenance
with which to detect an observation leaked from the future
[@friston2010freeenergy; @kaelbling1998pomdp; @friston2017process].

The second is a **single legible objective**. AlphaFund's central move is to put a
researcher hire, a data feed, a GPU, and a position in AAPL on one dollar axis by
differentiating a common objective. Expected Free Energy supplies a legible
preference/information decomposition in this model [@friston2015epistemic; @dacosta2020synthesis].
Adding terms expressed in nats does not establish their monetary calibration or
equivalence to a corporate portfolio objective [@friston2010freeenergy; @lattimore2020bandits].

AlphaFund's self-forecasting loop — predict (query the EWM for the marginal-return
vector), optimize (the convex inner program), execute the funded action, and fold
the realized outcome back in as the next training row — motivates a
perception–action analogy [@kaelbling1998pomdp]. Here action scoring is one-step;
{{PLANNING_HORIZON}} is the rollout/diagnostic horizon, not a multi-step optimizer.
[@fig:loop] is a conceptual schematic; the open-loop trajectory simulator does
not consume observations or refit coefficients [@friston2017process].

![Conceptual corporate self-forecasting loop. The local engine computes one-step action values and a selected-action example. Observation acquisition, external execution and coefficient learning are conceptual handoffs here, not operations performed by this diagram or the open-loop trajectory simulator.](../output/figures/self_forecasting_loop.png){#fig:loop}

The remainder of the paper builds the dictionary ([@sec:dictionary]), gives the
GNN-via-COGANT realization ([@sec:gnn]), works through generative-model inference
under the firm filtration ([@sec:inference]), derives the epistemic/pragmatic value
split and the separate t-RSI diagnostic ([@sec:value]), and evaluates the
implementation's integrity boundaries ([@sec:integrity]).
