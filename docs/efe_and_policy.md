# Expected Free Energy and Policy

`src/alphacogant/efe/free_energy.py` owns the project's action scoring.

## Sign convention

Active Inference motivates minimizing Expected Free Energy `G`
([epistemic-value paper](https://doi.org/10.1080/17588928.2015.1020053)). AlphaCOGANT reports:

- `pragmatic`: expected log-preference value
- `epistemic`: expected information gain about the hidden Theta capability only
- `total`: the minimized quantity, `-(pragmatic + epistemic)`

The action value used for ranking is therefore `-total`.

## Policy evaluation

For each action:

1. `predicted_belief()` advances the current belief through the action-specific
   transition tensor.
2. Reward and loss distributions are computed under the predicted belief.
3. Pragmatic value is computed from preferences `C_R` and `C_L`.
4. Epistemic value is computed as the expected KL divergence between posterior
   and prior beliefs over Theta.
5. `marginal_return_vector()` returns the negative-EFE value for every action.

## Policy posterior

`policy_posterior()` applies a precision-weighted softmax to the action values.
`gamma` must be positive and finite. Figure probabilities are evaluated along a
greedy prediction-only rollout; they do not establish an explore/exploit switch.

## Claim boundary

This is a discrete action scorer. It is the reduced-model image of AlphaFund's
continuous marginal-return formalism, not a continuous optimizer.

## Numerical admission

Actions and observations must be integer indices (booleans and floats are
rejected). Precomputed policy values must cover all six actions and be finite.
The softmax centers before scaling so extreme finite scores produce valid
probabilities. KL calculations mask impossible Theta entries before division;
point-mass beliefs therefore produce finite, nonnegative information gain
without invalid floating-point warnings. Loop horizons and bootstrap draw
counts must be positive integers; Dirichlet concentration must also be finite.
