# Extending the Model

This file records safe extension lanes. Each lane needs source changes, tests,
figures or tokens as appropriate, and a regenerated PDF.

## Continuous state

Replace two-level channel factors with continuous capability distributions, for
example Beta-distributed channel strength. This would improve dynamic range but
requires revising inference, figures, and limitations.

## Continuous actions

Replace the six discrete actions with a capital-allocation vector. Candidate
approaches:

- constrained softmax over channel weights
- Gumbel-softmax relaxation for differentiable discrete approximations
- normalizing-flow policy over allocation simplex

This is the path from negative-EFE action values to a true continuous marginal
return surface.

## Endogenous learning

Make model matrices update from observations. A minimal path is to make selected
transition probabilities Dirichlet or Beta posteriors. The manuscript's current
"no learning dynamics" limitation must be rewritten when this lands.

## Coupled transitions

Move from factorized `B_k` transitions to a coupled transition over joint channel
state. This would model supermodularity in state dynamics rather than only in
the observation likelihood.

## External capital amplification

Add an observation-volume or capital-growth state so cumulative pragmatic value
changes how much evidence the firm can buy next cycle.

## Required gates for any extension

- new or updated tests
- updated manuscript tokens when numbers change
- regenerated figures
- contained render and visual/reference checks from
  [rendering_and_validation.md](rendering_and_validation.md)
- optional template integration only when independently configured and authorized;
  do not run another checkout's stages as a standalone project requirement
