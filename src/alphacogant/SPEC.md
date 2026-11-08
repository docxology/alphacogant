# AlphaCOGANT core — implementation spec

A small, deterministic, **NumPy-only** Active Inference engine that realizes the
GNN model in `models/alphafund_ewm.md`: the AlphaFund five-channel Economic World
Model as a partially-observed Active Inference agent. No mocks; all tests use real
arrays and real computation; fixed seeds. Target ≥90% coverage on `src/`.

## Channels (the corporation tuple Ξ_t)

Five capital channels, fixed order `("I", "S", "U", "Theta", "Z")`:
- `I` Investments, `S` Sensors, `U` Actuators, `Theta` Parameters, `Z` R&D.
Channel role labels are conceptual. Theta is a hidden capability state, not the
learned numerical matrices; Z has no likelihood/value coupling in this profile.
Each is a 2-level factor `{weak=0, strong=1}`. Actions index `0..5` =
`(fund_I, fund_S, fund_U, fund_Theta, fund_Z, hold)`.

## Modules

### `model/channels.py`
- `CHANNELS: tuple[str,...]` and `ACTIONS: tuple[str,...]` constants.
- `Channel` dataclass: name, role in `{"pragmatic","epistemic","both"}`, 2-level.
- `channel_index(name) -> int`, `action_index(name) -> int`.

### `model/generative_model.py`
- `EconomicWorldModel` dataclass holding `A_R (3,2,2,2)`, `A_L (3,2,2)`,
  `B` dict of `{channel: (2,2,6)}`, `C_R (3,)`, `C_L (3,)`, `D` dict `{channel:(2,)}`.
- `default_model() -> EconomicWorldModel` returning the matrices in the GNN file
  (load the same numeric values; normalize columns to valid distributions).
- Validation: every A/B column is a normalized probability distribution
  (`assert allclose(sum, 1)`), every D sums to 1. Raise `ValueError` on malformed.
- `belief_prior() -> dict[channel, np.ndarray]` from D.
- `infer_states(model, obs_R, obs_L, prior) -> dict[channel, np.ndarray]`: one
  single-sweep factorized update from a bucketed reward/loss observation.
  Coupled likelihoods induce posterior dependence, so this is not exact joint
  filtering or exact joint marginals. Log-domain normalization admits rare
  positive evidence while rejecting truly impossible observations.

### `efe/free_energy.py` — the heart
- `expected_free_energy(model, belief, action) -> EFEResult` where `EFEResult`
  has `pragmatic: float`, `epistemic: float`, `total: float` and
  `total == -(pragmatic + epistemic)` (G is negated value; lower G = better).
  - `pragmatic` = expected authored reward/loss log-preference scores
    `E_q[C_R(o_R) + C_L(o_L)]` under the predicted post-action belief.
    These scores are illustrative, not measured log-equity growth.
  - `epistemic` = expected information gain about the hidden `Theta` factor:
    `E_o[KL(q(Theta'|o,a) || q(Theta'|a))]`. Must be ≥0. The matrices are
    fixed; this is not parameter learning. Z has no likelihood/value effect.
- `efe_vector(model, belief) -> list[EFEResult]`: the decomposition for every
  action, validating the belief exactly once (hot path for simulation and
  policy scoring).
  Scalar, batched and static preference evaluation reject nonfinite computed
  scores with `ValueError` before policy selection; finite coefficients alone
  do not guarantee representable arithmetic.
- `marginal_return_vector(model, belief) -> dict[action, float]`: `g_t` = the
  negative-EFE score for each discrete action, not an equity-gradient or
  per-dollar derivative. Its argmax includes `hold`; no convex optimizer is run.
- `greedy_action(model, belief) -> int`: the argmax action itself.
- `predicted_belief(model, belief, action) -> dict[channel, np.ndarray]`: the
  one-step B-rollout `B_k[:, :, action] @ belief_k` (no validation; caller
  passes a validated belief).
- `policy_posterior(model, belief, gamma=1.0, *, values=None) -> np.ndarray (6,)`:
  softmax over `-G` (precision gamma). Sums to 1. `values` may carry
  precomputed marginal returns to avoid a second EFE sweep.

### `trsi/t_rsi.py`
- `create_rate(model, belief) -> float`: horizon-mean greedy pragmatic value
  minus horizon-mean hold value, evaluated on H post-transition beliefs b1…bH.
- `decay_rate(model, belief) -> float`: mean nonnegative Theta-refresh
  opportunity gap on those same H greedy post-transition beliefs.
- `t_rsi(create_samples, decay_samples) -> float`: standardized distance
  `(mean(create) - mean(decay)) / se(create - decay)` where the samples are
  **paired** (entry `i` of each array comes from the same perturbed trajectory,
  as `paired_bootstrap_samples` produces) and `se(x)` is
  `std(x, ddof=1)/sqrt(n)`. The independence-assuming pooled SE
  `sqrt(se(create)**2 + se(decay)**2)` is deliberately NOT used: pairing
  requires covariance. Current correlation and paired margin SE are generated
  from the same canonical samples rather than fixed historical literals.
  Return signed `inf` only if the margin SE is exactly 0 and the margin mean is
  nonzero; return 0 for n<2; raise if the arrays are non-empty but of unequal
  length (unpaired input).
  Scale-stable reductions prevent finite small/large samples from producing
  spurious zero/infinite dispersion. Synthetic perturbation count controls
  Monte Carlo SE, not empirical corporate confidence.
- `bootstrap_t_rsi(model, belief, rng, n=500) -> dict` returning
  `{"t_rsi", "create_mean", "decay_mean", "create_se", "decay_se"}` from a
  deterministic-seeded bootstrap over per-channel belief perturbations.
- `certificate(t_rsi_value, delta) -> bool`: scalar threshold predicate only;
  neither simulator enforces action/model-update admission or monotone EFE improvement.

### `bridge/cogant_bridge.py` — codebase→GNN→EWM provenance
- `firm_structure_to_channels(spec: dict) -> dict`: map a small firm-description
  dict (counts of data feeds, venues, models, researchers, book size) onto the
  five channel prior beliefs `D` (more feeds → stronger `S` prior, etc.). Pure,
  deterministic, documented mapping. This is the COGANT translation step in
  miniature: structure → generative-model priors.
- `model_to_gnn_summary(model) -> str`: emit a **full** GNN-style block
  (StateSpaceBlock, Connections, InitialParameterization,
  ActInfOntologyAnnotation) from an `EconomicWorldModel`, proving the round-trip
  from NumPy arrays back to GNN text. Every section a GNN processor needs is
  present.
- `parse_gnn_summary(text) -> dict`: lightweight structural inspection (factors,
  connections, ontology); this function does not validate or reconstruct arrays.
- `bridge.gnn.model_from_gnn(text) -> EconomicWorldModel`: reconstruct all arrays
  from this project's finite literal-only GNN profile without renormalization.
  Reject missing/duplicate sections and parameters, wrong shapes, nonfinite or
  nonstochastic values and inputs over one megabyte. All B tensors use
  `[next,current,action]` axes. Tests compare every authored coefficient to the engine.
- `bridge.gnn.joint_matrices(model) -> dict`: exact product compilation over the
  lexicographic `(I,S,U,Theta,Z)` state axis and `(reward,loss)` observation axis.
  Return complete A[9,32], B[32,32,6], C[9], D[32].
- `bridge.gnn.model_to_cogant_gnn(model) -> str`: add the complete joint matrices
  in COGANT's authoritative `gnn-matrices` fence. Real upstream integration
  checks coefficient equality and execution, not policy equivalence.
- `bridge.inference.run_inference(source, *, steps=8, observations=None) -> dict`:
  filter current observations, select the lowest-EFE action, and predict next
  beliefs. Record exact source/parameter digests and the single-sweep mean-field
  approximation. Modal model rollout without observations is explicitly labeled.
- `bridge.inference.verify_inference(source, trace) -> None`: replay all fields
  and reject stale source or changed computation. Traces admit 1–10,000 steps.
  t-RSI remains a separately evaluated certificate, not a gate enforced by this loop.

### `model/operating_points.py`
- `IMPROVING` / `COASTING`: the two canonical operating-point belief maps (weak
  channels + stale parameters vs. strong channels + fresh parameters). The single
  source of truth — every figure script, test, and the manuscript-variables
  generator imports from here instead of re-declaring literals.
- `IMPROVING_RAW` / `COASTING_RAW`: the same beliefs as `(weak, strong)` tuples,
  for figure scripts that need the raw form.
- `as_belief(raw) -> dict`: convert a raw tuple spec into the channel→array
  belief map (plain conversion; engine boundaries validate).
- `BOOTSTRAP_SEED`, `BOOTSTRAP_N`, `BOOTSTRAP_CONCENTRATION`: the canonical
  bootstrap constants used by `manuscript_variables.generate_variables` and the
  `fig_trsi_densities` figure.

### `tokens/manuscript_variables.py`
- `generate_variables() -> dict[str, str]`: every numeric the manuscript prose
  cites as a `{{TOKEN}}` is produced here from the live model + a deterministic
  demo trajectory (headline t-RSI, create_mean, decay_mean, epistemic/pragmatic of
  the funded policy, channel count, action count, planning horizon, etc.). Used by
  `scripts/z_generate_manuscript_variables.py`.

### `stats/simulation.py`
- `simulate_trajectory(model, initial_belief, horizon, *, policy, seed) -> TrajectoryResult`:
  predict `horizon` cycles under greedy, hold, fund_theta, or stochastic control.
  Each cycle records the pre-action belief, selected action, EFE decomposition
  and marginal-return vector. Matrices remain fixed; the rollout consumes no
  observations and demonstrates neither learned improvement nor an inevitable
  exploration-to-exploitation switch.
- `TrajectoryResult`: frozen dataclass with `cycles` (list of `CycleRecord`),
  `belief_history` (channel→[p_strong per cycle]), `action_history`, and
  `cumulative_pragmatic`. `action_counts()` returns how many times each action
  was selected.
- `summarize_trajectory(trajectory) -> dict`: compact summary for the manuscript
  (first/last funded channel, belief deltas, total pragmatic/epistemic value,
  exploration ratio, dominant action).

### `stats/sensitivity.py`
- `sweep_concentration(model, belief, *, concentrations, seed, n) -> dict`: bootstrap
  t-RSI across a range of Dirichlet concentrations, showing how robust the headline
  number is to the firm's belief precision.
- `sweep_theta_freshness(model, base_belief, *, theta_values, seed, n, concentration) -> dict`:
  bootstrap t-RSI as the Theta-freshness prior is swept from stale to fresh, showing
  how the certificate responds to the firm's belief about its own parameter quality.

### `stats/statistics.py`
- `bootstrap_ci(samples, confidence) -> BootstrapCI`: percentile interval summary
  for generated rate samples.
- `cohen_d(samples_a, samples_b) -> float`: pooled-std Cohen's d between two
  sample sets.
- `compute_regime_statistics(...) -> RegimeStatistics`: bootstrap CIs, EFE
  decomposition, funded action, and effect-size inputs for one operating point.
- `compare_regimes(...) -> RegimeComparison`: improving-vs-coasting table and
  Cohen's d summaries.
- `break_even_profile(...) -> BreakEvenProfile`: paired bootstrap estimate of
  `P(create_rate > decay_rate)` plus the create-minus-decay margin interval. The
  same belief perturbation feeds both rates, so the probability is an event-level
  complement to the standardized t-RSI distance.

## Tests (`tests/`, no mocks, fixed seeds)
- `test_generative_model.py`: matrices valid distributions; `infer_states` moves
  belief toward `strong` after a high-reward + low-loss observation.
- `test_free_energy.py`: epistemic ≥ 0 always; funding `Theta` when it is stale
  yields higher epistemic than `hold`; pragmatic is highest when production
  channels strong; `policy_posterior` sums to 1; `total == -(prag+epist)`.
- `test_t_rsi.py`: t-RSI positive when create_mean > decay_mean; standardization
  pairing-aware (same marginal means, tighter paired margins → larger |t-RSI|);
  certificate fires iff `t_rsi >= delta`; bootstrap deterministic under fixed
  seed (two runs identical).
- `test_cogant_bridge.py`: richer firm structure → stronger channel priors;
  `model_to_gnn_summary` mentions all five channels and "ExpectedFreeEnergy".
- `test_manuscript_variables.py`: every token is a non-empty string; a curated set
  of expected token keys is present (mirror the template's gate).

## Invariants (hard)
- Active Inference sign convention: EFE `G` is minimized; value `= -G`. Document it
  once and keep it consistent everywhere.
- Epistemic value is never negative (KL ≥ 0).
- All randomness via an explicit `numpy.random.Generator`; no global seed; same
  seed → identical output (determinism test enforces it).
