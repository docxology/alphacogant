# Functionality and integrity AlphaCOGANT brings {#sec:integrity}

The user-facing question is not only "can the firm be modeled this way" but "what
does modeling it this way *give* AlphaFund." Five concrete things, each an
integrity property that the Active Inference / GNN representation either makes
explicit or gates with source-owned computation.

## 1. Filtration integrity — the model cannot cheat on time

AlphaFund's existential methodological risk is a controller that looks, even
slightly, into the future — a backtest whose training data contains a later
retrospective, an evaluation window contaminated by post-cutoff information. Such a
controller posts a flattering t-RSI and then bleeds capital live. In a GNN
generative model the no-peeking property is a **typing constraint**: beliefs are
forward functions of the history filtration $\mathcal{F}_t$, and a valid factor
graph has no edge that routes a future observation into a past belief. The
factor-graph `Connections` block makes that temporal claim auditable by eye [1, 2, 24, 25].
AlphaCOGANT thus converts AlphaFund's "we promise our holdout is strictly
post-corpus" into a visible graph contract: every inference edge has to point
from information available at decision time [14, 18].

## 2. Auditable capital allocation — one objective, every move scored

A differentiable corporation is only as trustworthy as the legibility of the
objective its controller optimizes. Expected Free Energy is a single scalar with a
named decomposition: every admissible funding move is scored by negative EFE, and
every funded channel's worth is reported as a pragmatic part (expected log-equity
now) plus an epistemic part (information that prices future equity). A reviewer
can ask of any allocation, "did it clear the shadow price, and was it bought for
return or for knowledge?" and get a number, not a narrative. This is the integrity
AlphaFund gestures at with "an auditable capital-allocation process"; the EFE
decomposition is what makes the audit mechanical [3, 25].

## 3. Reproducibility-by-construction — every prose number is a gate

AlphaCOGANT inherits the template's discipline: every numeric the manuscript cites
is a generated token emitted by one function (`manuscript_variables.generate_variables`)
and cross-checked by one test, so a drifted constant, a deleted result, or an
out-of-sync narrative turns the build red before it can reach a PDF. Applied to a
firm that grades itself, this is not cosmetic: it means the headline t-RSI in the
document is provably the t-RSI the shipped engine computed from the shipped model,
not a number typed by an optimist. This is the same reproducibility mechanism
used across the manuscript pipeline [1], and the same mechanism is used in the
template for source-only provenance [5]. The same gate that protects the template's
optimization numbers protects AlphaCOGANT's create-rate, decay-rate, and certificate
threshold [6]. Functionality (the engine runs and is ≥90%-covered, no mocks) and
integrity (the prose cannot diverge from the engine) are enforced by the same CI.

## 4. Artifact provenance — every figure has a producer

The same contract now covers visual evidence. The variable-generation script reads
the manuscript's figure references after token injection and writes
`output/figures/figure_registry.json`, a registry of labels, source manuscript
files, captions, filenames, and producer scripts. The artifact manifest then hashes
the stable output surface and records issues when a registered figure is missing,
too small, not a PNG, duplicated by label or filename, or disconnected from a
producer script. The verification pass (`z_generate_manuscript_variables.py --check`)
fails the build when any such issue is present. This matters for a self-grading firm
because figures are often where a manuscript can launder stale computation into fresh
prose. AlphaCOGANT's contract instead requires the figure, its caption, and its
generating script to remain mutually visible.

## 5. The certificate as a tamper-resistant commit gate

The certificate of monotone improvement is the operational integrity primitive: a
candidate $\Theta$ update is admitted into the deployed model **only** when t-RSI
clears the margin on every active channel. This is the Active Inference
admissibility test [3, 24, 25] (accept a model revision only when its expected free energy is
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
reduction: AlphaFund's recursive-self-improvement-as-portfolio-optimization has
an Active Inference representation, it is expressible in GNN, it is producible by
the COGANT pattern from an API-complete firm, and casting it that way makes
filtration integrity, a legible objective, reproducibility-by-construction, and
an auditable commit gate explicit.
