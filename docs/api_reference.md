# API Reference

This is a compact map of stable engine functions used by scripts, tests, and the
manuscript.

## Model

- `default_model()` builds the reduced Economic World Model.
- `belief_prior(model)` returns the prior belief map.
- `infer_states(model, obs_R, obs_L, prior)` performs one filtering update.
- `validate_belief_map(belief, context=...)` normalizes and validates beliefs.

## EFE

- `expected_free_energy(model, belief, action)` returns pragmatic, epistemic,
  and total EFE.
- `marginal_return_vector(model, belief)` returns negative-EFE values for all
  actions.
- `policy_posterior(model, belief, gamma=1.0)` returns a softmax posterior over
  actions.

## t-RSI

- `create_rate(model, belief, horizon=DEFAULT_HORIZON)` computes greedy value
  over passive holding.
- `decay_rate(model, belief, horizon=DEFAULT_HORIZON)` computes residual Theta
  erosion.
- `bootstrap_t_rsi(model, belief, rng, ...)` returns deterministic bootstrap
  summary values.
- `certificate(t_rsi_value, delta)` applies the threshold gate.

## Statistics

- `bootstrap_ci(samples, confidence=0.95)` returns a percentile interval.
- `compute_regime_statistics(...)` returns a complete regime profile.
- `compare_regimes(...)` compares improving and coasting regimes.
- `break_even_profile(...)` estimates paired `create > decay` event mass.

## Bridge

- `firm_structure_to_channels(spec)` maps firm counts to priors.
- `model_to_gnn_summary(model)` emits a GNN-style summary.
- `parse_gnn_summary(text)` extracts sections, factors, connections, and ontology.

## Tokens

- `generate_variables()` returns the full manuscript token dictionary.
