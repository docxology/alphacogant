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

## Finite GNN and replayable inference (1.1.0)

Public imports from `alphacogant` now include `model_from_gnn`,
`model_to_cogant_gnn`, `joint_matrices`, `run_inference` and `verify_inference`.
`model_to_gnn_summary` now preserves every numeric coefficient; the lightweight
`parse_gnn_summary` structural return type is unchanged. Read the
[bridge contract](cogant_bridge.md) for axes, budgets, observation modes and
policy differences from COGANT. CLI: `uv run --locked alphacogant --help`.

`generate_variables(n=None, *, manuscript_dir=None)` requires actual manuscript
sources for definition/figure counts. In a source checkout it uses
`docs/manuscript/`; installed-wheel callers must supply that directory explicitly.
Missing manuscript sources raise `FileNotFoundError` before computation or artifact
writes, rather than producing zero counts. Artifact generation scripts require
the source checkout.

## Versioned agent and profiling surfaces

`alphacogant.api.Client.call`, `capabilities`, and `handle_request` provide the
[agent protocol](agent_api.md). `alphacogant.profiling.profile_workload` returns
[bounded performance evidence](performance.md) with source/runtime provenance.
