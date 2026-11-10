# Minimize one-step EFE and rank discrete actions

`src/alphacogant/efe/free_energy.py` owns the project's action scoring.

## Minimized G versus maximized action value

Active Inference motivates minimizing Expected Free Energy `G`
([epistemic-value paper](https://doi.org/10.1080/17588928.2015.1020053)). AlphaCOGANT reports:

- `pragmatic`: expected log-preference value
- `epistemic`: expected information gain about the hidden Theta capability only
- `total`: the minimized quantity, `-(pragmatic + epistemic)`

The action value used for ranking is therefore `-total`.

## Score each action's predicted belief

For each action:

1. `predicted_belief()` advances the current belief through the action-specific
   transition tensor.
2. Reward and loss distributions are computed under the predicted belief.
3. Pragmatic value is computed from preferences `C_R` and `C_L`.
4. Epistemic value is computed as the expected KL divergence between posterior
   and prior beliefs over Theta.
5. `marginal_return_vector()` returns the negative-EFE value for every action.

## Softmax policy precision is separate from perturbation concentration

`policy_posterior()` applies a precision-weighted softmax to the action values.
`gamma` must be positive and finite. Figure probabilities are evaluated along a
greedy prediction-only rollout; they do not establish an explore/exploit switch.

`gamma` is not the Dirichlet concentration used in synthetic belief perturbations.
`simulate_trajectory` repeatedly selects and predicts without filtering observations
or learning coefficients. The [paired diagnostic](trsi_statistics.md) separately
contrasts greedy and hold trajectories; its threshold predicate never controls
action selection.

## Discrete scoring is not a continuous portfolio solver

This is the reduced-model image of AlphaFund's continuous marginal-return
formalism, not a continuous optimizer, convex solve or dollar-calibrated gradient.

## Admit finite action scores and valid indices

Actions and observations must be integer indices (booleans and floats are
rejected). Precomputed policy values must cover all six actions and be finite.
The softmax centers before scaling to retain close common-offset scores. When
an opposite-sign gap overflows but a small precision makes the scaled gap finite,
it recomputes only that gap safely instead of erasing valid probability mass.
KL calculations mask impossible Theta entries before division;
point-mass beliefs therefore produce finite, nonnegative information gain
without invalid floating-point warnings. Sampling bounds and concentration
admission belong to the [paired diagnostic contract](trsi_statistics.md).

Prediction-only trajectory and summary totals must also remain representable.
Finite per-cycle scores can still overflow their cumulative pragmatic value;
the simulator raises `ValueError` at the first failing cycle rather than
publishing an infinite total. This does not alter genuine constant-margin
t-RSI infinities.
