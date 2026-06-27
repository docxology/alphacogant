# Expected Free Energy and Policy

`src/alphacogant/efe/free_energy.py` owns the project's action scoring.

## Sign convention

Active Inference minimizes Expected Free Energy `G` [2, 3]. AlphaCOGANT reports:

- `pragmatic`: expected log-preference value
- `epistemic`: expected information gain [3, 4]
- `total`: the minimized quantity, `-(pragmatic + epistemic)`

The action value used for ranking is therefore `-total`.

## Policy evaluation

For each action:

1. `_predict_belief()` advances the current belief through the action-specific
   transition tensor.
2. Reward and loss distributions are computed under the predicted belief.
3. Pragmatic value is computed from preferences `C_R` and `C_L`.
4. Epistemic value is computed as the expected KL divergence between posterior
   and prior beliefs over Theta.
5. `marginal_return_vector()` returns the negative-EFE value for every action.

## Policy posterior

`policy_posterior()` applies a precision-weighted softmax to the action values.
`gamma` must be positive and finite. The posterior is used in figures to show
how mass shifts from exploration to exploitation as beliefs change.

## Claim boundary

This is a discrete action scorer. It is the reduced-model image of AlphaFund's
continuous marginal-return formalism, not a continuous optimizer.
