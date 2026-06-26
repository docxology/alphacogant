# AlphaCOGANT core — implementation spec

A small, deterministic, **NumPy-only** Active Inference engine that realizes the
GNN model in `models/alphafund_ewm.md`: the AlphaFund five-channel Economic World
Model as a partially-observed Active Inference agent. No mocks; all tests use real
arrays and real computation; fixed seeds. Target ≥90% coverage on `src/`.

## Channels (the corporation tuple Ξ_t)

Five capital channels, fixed order `("I", "S", "U", "Theta", "Z")`:
- `I` Investments (pragmatic, production), `S` Sensors (epistemic, filtration),
- `U` Actuators (pragmatic, execution), `Theta` Parameters (the EWM itself),
- `Z` R&D (epistemic, compounds the loop).
Each is a 2-level factor `{weak=0, strong=1}`. Actions index `0..5` =
`(fund_I, fund_S, fund_U, fund_Theta, fund_Z, hold)`.

## Modules

### `channels.py`
- `CHANNELS: tuple[str,...]` and `ACTIONS: tuple[str,...]` constants.
- `Channel` dataclass: name, role in `{"pragmatic","epistemic","both"}`, 2-level.
- `channel_index(name) -> int`, `action_index(name) -> int`.

### `generative_model.py`
- `EconomicWorldModel` dataclass holding `A_R (3,2,2,2)`, `A_L (3,2,2)`,
  `B` dict of `{channel: (2,2,6)}`, `C_R (3,)`, `C_L (3,)`, `D` dict `{channel:(2,)}`.
- `default_model() -> EconomicWorldModel` returning the matrices in the GNN file
  (load the same numeric values; normalize columns to valid distributions).
- Validation: every A/B column is a normalized probability distribution
  (`assert allclose(sum, 1)`), every D sums to 1. Raise `ValueError` on malformed.
- `belief_prior() -> dict[channel, np.ndarray]` from D.
- `infer_states(model, obs_R, obs_L, prior) -> dict[channel, np.ndarray]`: one
  Bayesian filtering update of the per-channel posterior from a bucketed reward
  obs and loss obs (mean-field factorized; exact for this factor graph).

### `free_energy.py` — the heart
- `expected_free_energy(model, belief, action) -> EFEResult` where `EFEResult`
  has `pragmatic: float`, `epistemic: float`, `total: float` and
  `total == -(pragmatic + epistemic)` (G is negated value; lower G = better).
  - `pragmatic` = expected log-preference `E_q[ln P(o_R|C_R)] + E_q[ln P(o_L|C_L)]`
    under the predicted post-action belief (this is the create-rate contribution /
    expected log-equity).
  - `epistemic` = expected information gain about `Theta` (and `S`,`Z` via their
    effect on predicted obs): `D_KL[q(s'|o,a) || q(s'|a)]` summed over predicted
    observations, weighted by predicted obs probability. Must be ≥ 0.
- `marginal_return_vector(model, belief) -> dict[action, float]`: `g_t` = the
  negative-EFE per action = expected log-equity growth per channel-dollar. The
  argmax is the channel the portfolio optimizer funds this cycle.
- `policy_posterior(model, belief, gamma=1.0) -> np.ndarray (6,)`: softmax over
  `-G` (precision gamma). Sums to 1.

### `t_rsi.py`
- `create_rate(model, belief) -> float`: pragmatic value of the best funded policy
  (gross alpha-creation rate).
- `decay_rate(model, belief) -> float`: expected pragmatic loss per cycle from the
  `B_Theta` fresh→stale leak under non-funding (alpha-decay).
- `t_rsi(create_samples, decay_samples) -> float`: standardized distance
  `(mean(create) - mean(decay)) / sqrt(se(create)**2 + se(decay)**2)`; `se` is
  `std(x, ddof=1)/sqrt(n)`. Return `inf` only if both SEs are exactly 0 and
  means differ; guard n<2.
- `bootstrap_t_rsi(model, belief, rng, n=500) -> dict` returning
  `{"t_rsi", "create_mean", "decay_mean", "create_se", "decay_se"}` from a
  deterministic-seeded bootstrap over per-channel belief perturbations.
- `certificate(t_rsi_value, delta) -> bool`: the monotone-improvement gate.

### `cogant_bridge.py` — codebase→GNN→EWM provenance
- `firm_structure_to_channels(spec: dict) -> dict`: map a small firm-description
  dict (counts of data feeds, venues, models, researchers, book size) onto the
  five channel prior beliefs `D` (more feeds → stronger `S` prior, etc.). Pure,
  deterministic, documented mapping. This is the COGANT translation step in
  miniature: structure → generative-model priors.
- `model_to_gnn_summary(model) -> str`: emit a short GNN-style block (factor list,
  ontology annotation) from an `EconomicWorldModel`, proving the round-trip.

### `operating_points.py`
- `IMPROVING` / `COASTING`: the two canonical operating-point belief maps (weak
  channels + stale parameters vs. strong channels + fresh parameters). The single
  source of truth — every figure script, test, and the manuscript-variables
  generator imports from here instead of re-declaring literals.
- `IMPROVING_RAW` / `COASTING_RAW`: the same beliefs as `(weak, strong)` tuples,
  for figure scripts that need the raw form.
- `as_belief(raw) -> dict`: convert a raw tuple spec into a validated belief map.
- `BOOTSTRAP_SEED`, `BOOTSTRAP_N`, `BOOTSTRAP_CONCENTRATION`: the canonical
  bootstrap constants used by `manuscript_variables.generate_variables` and the
  `fig_trsi_densities` figure.

### `manuscript_variables.py`
- `generate_variables() -> dict[str, str]`: every numeric the manuscript prose
  cites as a `{{TOKEN}}` is produced here from the live model + a deterministic
  demo trajectory (headline t-RSI, create_mean, decay_mean, epistemic/pragmatic of
  the funded policy, channel count, action count, planning horizon, etc.). Used by
  `scripts/z_generate_manuscript_variables.py`.

## Tests (`tests/`, no mocks, fixed seeds)
- `test_generative_model.py`: matrices valid distributions; `infer_states` moves
  belief toward `strong` after a high-reward + low-loss observation.
- `test_free_energy.py`: epistemic ≥ 0 always; funding `Theta` when it is stale
  yields higher epistemic than `hold`; pragmatic is highest when production
  channels strong; `policy_posterior` sums to 1; `total == -(prag+epist)`.
- `test_t_rsi.py`: t-RSI positive when create_mean > decay_mean; standardization
  monotone in separation; certificate fires iff `t_rsi >= delta`; bootstrap
  deterministic under fixed seed (two runs identical).
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
