# Define new model contracts before extending the reduced engine

This file records possible extension lanes, not implemented features. Each lane
needs source changes, tests, regenerated evidence and revised claims as appropriate.

## Replace binary capability with continuous state

Replace two-level channel factors with continuous capability distributions, for
example Beta-distributed channel strength. This would change representational
resolution, not establish calibration; inference, figures and limitations need revision.

## Replace discrete moves with a continuous allocation policy

Replace the six discrete actions with a capital-allocation vector. Candidate
approaches:

- constrained softmax over channel weights
- Gumbel-softmax relaxation for differentiable discrete approximations
- normalizing-flow policy over allocation simplex

This would permit a continuous action-value surface. A monetary marginal-return
interpretation would still require a specified economic objective and calibration.

## Learn coefficients rather than only infer capability

Make model matrices update from observations. A minimal path is to make selected
transition probabilities Dirichlet or Beta posteriors. The manuscript's current
"no learning dynamics" limitation must be rewritten when this lands.

## Add coupled state transitions with an explicit joint contract

Move from factorized `B_k` transitions to a coupled transition over joint channel
state. This would model supermodularity in state dynamics rather than only in
the observation likelihood.

## Model external capital and evidence acquisition explicitly

Add an observation-volume or capital-growth state so cumulative pragmatic value
changes how much evidence the firm can buy next cycle.

## Regenerate evidence and revise claims for every extension

- new or updated tests
- updated manuscript tokens when numbers change
- regenerated figures
- actual figure inspection and, for manuscript/visual releases, contained PDF/HTML
  generation, page inspection and reference checks from [rendering_and_validation.md](rendering_and_validation.md)
- optional template integration only when independently configured and authorized;
  do not run another checkout's stages as a standalone project requirement
