# Python entry points and contract owners

This is an import map, not a second protocol specification. The engine functions
listed below are also exported from `alphacogant`, except
`paired_bootstrap_samples`, which is imported from its defining module.

| Defining module | Entry points | Detailed semantics |
| --- | --- | --- |
| `alphacogant.model.generative_model` | `EconomicWorldModel`, `default_model`, `belief_prior`, `infer_states`, `validate_belief_map` | [A/B/C/D and single-sweep factorized filtering](generative_model_contract.md) |
| `alphacogant.model.channels` | `CHANNELS`, `ACTIONS`, `action_index`, `channel_index` | [Model axes and channels](generative_model_contract.md) |
| `alphacogant.efe.free_energy` | `expected_free_energy`, `efe_vector`, `marginal_return_vector`, `greedy_action`, `policy_posterior`, `predicted_belief`, `static_pragmatic_value` | [One-step EFE signs and policy precision](efe_and_policy.md) |
| `alphacogant.trsi.t_rsi` | `create_rate`, `decay_rate`, `paired_bootstrap_samples`, `t_rsi`, `bootstrap_t_rsi`, `certificate` | [Paired diagnostic, not an enforced gate](trsi_statistics.md) |
| `alphacogant.stats.statistics` | `bootstrap_ci`, `compute_regime_statistics`, `compare_regimes`, `break_even_profile` | [Synthetic intervals and event mass](trsi_statistics.md) |
| `alphacogant.stats.simulation` | `simulate_trajectory`, `summarize_trajectory` | [Prediction-only trajectory scope](efe_and_policy.md) |
| `alphacogant.bridge.cogant_bridge` | `firm_structure_to_channels`, `model_to_gnn_summary`, `parse_gnn_summary` | [Authored priors and numeric serialization](cogant_bridge.md) |
| `alphacogant.bridge.gnn` | `model_from_gnn`, `model_to_cogant_gnn`, `joint_matrices` | [Finite-profile import and joint export](cogant_bridge.md) |
| `alphacogant.bridge.inference` | `run_inference`, `verify_inference` | [Source-bound filter/score/predict traces](cogant_bridge.md) |
| `alphacogant.tokens.manuscript_variables` | `generate_variables` | [Source-to-output ownership](methods_and_artifacts.md) |

## Generate evidence tokens from manuscript sources

`generate_variables(n=None, *, manuscript_dir=None)` requires actual manuscript
sources for definition/figure counts. A source checkout defaults to
`docs/manuscript/`; installed-wheel callers must supply that directory explicitly.
Missing sources raise `FileNotFoundError` before computation or artifact writes.
Artifact-generation scripts require the source checkout. An explicit `n` changes
the synthetic sample recipe and its reported results, not the empirical data volume.

## Integrate bounded requests or collect performance evidence

Use `from alphacogant.api import Client, capabilities, handle_request` for the
[versioned agent protocol](agent_api.md), which owns schemas, errors, budgets and
CLI behavior. `alphacogant.profiling.profile_workload` produces
[bounded performance evidence](performance.md). Neither surface is an external
execution service or a financial controller.
