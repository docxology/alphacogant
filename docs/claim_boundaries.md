# Distinguish executable model claims from financial evidence

This file is the high-level honesty contract for both docs and manuscript prose.

## Claims AlphaCOGANT makes

- AlphaFund's recursive-corporate-self-improvement framing motivates a reduced
  Active Inference analogy, not an equivalence of economic objectives.
- The five-channel Economic World Model can be expressed as a GNN-style
  generative model.
- The NumPy engine computes inference, Expected Free Energy, policy posterior,
  t-RSI, regime statistics, and figures from source-owned code.
- The reduced model demonstrates inspectable filter/score/predict ordering,
  source-owned artifact generation, and a separately evaluated sign-discriminating
  synthetic diagnostic. It does not authenticate caller-supplied chronology.

## Claims AlphaCOGANT does not make

- It does not reproduce AlphaFund's proprietary execution surfaces.
- It does not claim live trading performance.
- It does not provide financial advice.
- It does not claim that the two-level reduced model robustly certifies net
  self-improvement under uncertainty.
- It does not implement continuous dollar allocation.
- It does not implement endogenous model learning.
- It does not enforce the scalar certificate predicate as an action or model-update gate.
- It does not equate local factorized filtering or Theta-information EFE with
  upstream joint filtering or risk-plus-ambiguity policies.

## Use precise reduced-model language

Prefer:

- "reduced Active Inference analogy"
- "reduced model"
- "negative-EFE action value"
- "source-owned demonstration"

Avoid:

- "the same control problem"
- "type checker rejects it" unless a concrete type-checking command is actually
  part of the verified pipeline
- "every allocation is a gradient" for the current discrete-action engine
