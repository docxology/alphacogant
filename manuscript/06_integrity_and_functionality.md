# Functionality and integrity AlphaCOGANT brings

The user-facing question is not only "can the firm be modeled this way" but "what
does modeling it this way *give* AlphaFund." Four concrete things, each an integrity
property Active Inference / GNN enforces structurally rather than by convention.

## 1. Filtration integrity — the model cannot cheat on time

AlphaFund's existential methodological risk is a controller that looks, even
slightly, into the future — a backtest whose training data contains a later
retrospective, an evaluation window contaminated by post-cutoff information. Such a
controller posts a flattering t-RSI and then bleeds capital live. In a GNN
generative model the no-peeking property is a **typing constraint**: beliefs are
forward functions of the history filtration $\mathcal{F}_t$, and a connection that
would route a future observation into a past belief is not a valid edge. The GNN
type checker rejects it; the factor-graph `Connections` block makes it auditable by
eye. AlphaCOGANT thus converts AlphaFund's "we promise our holdout is strictly
post-corpus" into "the model is incapable of expressing the violation."

## 2. Auditable capital allocation — one objective, every dollar a gradient

A differentiable corporation is only as trustworthy as the legibility of the
objective its controller optimizes. Expected Free Energy is a single scalar with a
named decomposition: every allocation is $-\partial G/\partial a$, and every funded
channel's worth is reported as a pragmatic part (expected log-equity now) plus an
epistemic part (information that prices future equity). A reviewer can ask of any
dollar, "did it clear the shadow price, and was it bought for return or for
knowledge?" and get a number, not a narrative. This is the integrity AlphaFund
gestures at with "an auditable capital-allocation process"; the EFE decomposition
is what makes the audit mechanical.

## 3. Reproducibility-by-construction — every prose number is a gate

AlphaCOGANT inherits the template's discipline: every numeric the manuscript cites
is a `{{TOKEN}}` emitted by one function (`manuscript_variables.generate_variables`)
and cross-checked by one test, so a drifted constant, a deleted result, or an
out-of-sync narrative turns the build red before it can reach a PDF. Applied to a
firm that grades itself, this is not cosmetic: it means the headline t-RSI in the
document is provably the t-RSI the shipped engine computed from the shipped model,
not a number typed by an optimist. The same gate that protects the template's
optimization numbers protects AlphaCOGANT's create-rate, decay-rate, and certificate
threshold. Functionality (the engine runs and is ≥90%-covered, no mocks) and
integrity (the prose cannot diverge from the engine) are enforced by the same CI.

## 4. The certificate as a tamper-resistant commit gate

The certificate of monotone improvement is the operational integrity primitive: a
candidate $\Theta$ update is admitted into the deployed model **only** when t-RSI
clears the margin on every active channel. This is the Active Inference
admissibility test (accept a model revision only when its expected free energy is
reliably lower) and it is what distinguishes a self-improving corporation from one
that promotes noise. Because it is a function of beliefs the engine computes and
logs, the gate is reproducible and reviewable — a self-improvement step leaves an
auditable record of *why* it was admitted, in the same currency every other
decision uses.

## What this does and does not claim

AlphaCOGANT is a **modeling and integrity instrument**, not a trading system and
not financial advice. It does not reproduce AlphaFund's proprietary execution-
friction surface, its fitted scaling exponents, or its live track record — those
are not public and are deliberately out of scope. The reduced two-level GNN model
and the illustrative matrices are chosen for legibility and type-checkability, and
every numeric in this manuscript is a property of *that* model, generated and
gated by the engine. The transferable claim is structural and survives the
reduction: AlphaFund's recursive-self-improvement-as-portfolio-optimization is an
Active Inference control problem, it is expressible in GNN, it is producible by the
COGANT pattern from an API-complete firm, and casting it that way buys filtration
integrity, a legible objective, reproducibility-by-construction, and an auditable
commit gate for free.
