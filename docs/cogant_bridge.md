# COGANT Bridge

AlphaCOGANT is an authored reduced economic model. Its bridge adapts COGANT's
codebase-to-GNN design locally; it does not run repository extraction or infer
AlphaFund's proprietary dynamics. The [upstream contract](upstream_compatibility.md)
records the public GNN and COGANT revisions reviewed for this release.

## Firm structure to priors

`firm_structure_to_channels()` accepts finite, nonnegative counts and transforms
them through a logistic-scaled `log1p` map:

| Channel | Accepted count keys |
| --- | --- |
| `I` | `book_size`, `book`, `aum` |
| `S` | `data_feeds`, `feeds` |
| `U` | `venues`, `routes` |
| `Theta` | `models`, `ewm_models` |
| `Z` | `researchers`, `research_staff` |

## Numeric round trip

`model_to_gnn_summary()` emits every A/B/C/D coefficient with round-trip float
precision, declarations, connections, ontology annotations, and the standard
GNN sections. `model_from_gnn()` restores the complete `EconomicWorldModel`.
It reads this project's bounded finite profile, not arbitrary GNN programs.
It rejects missing or duplicate sections and coefficients, invalid literals,
nonfinite values, wrong axes, nonstochastic distributions, contradictory joint
matrix extensions, and sources exceeding 1,000,000 UTF-8 bytes. Parameter
assignments are scanned at the current position, never by retrying every suffix
of an identifier; multiline numeric literals and comments remain supported.
Unsupported assignment content and integers outside floating-point range are
ordinary validation errors. Parameters are data; the parser never evaluates code
or silently renormalizes an imported model.

`parse_gnn_summary()` remains a lightweight structural inspector for callers
that only need factors, connections and ontology. It does not validate a model.
The authored file is `models/alphafund_ewm.md`; coefficient-level tests bind it
to `default_model()`. All B tensors use `[next, current, action]` axes. Older
source text displayed action-major B rows despite declaring action-last axes;
version 1.1.0 corrects that layout without changing engine coefficients.

### Fixed-profile validation limits

`StateSpaceBlock` must declare exactly the exported finite-profile variables,
including both `(3,1)` observation arrays, the six-action policy and EFE vectors,
and scalar action/t-RSI outputs. Parameters, hidden factors and real-valued
outputs require `type=float`; observation/action declarations accept `int` or
`float` notation, while runtime observations and action indices remain integers.
`Connections` must contain exactly the exporter's fixed edge set, in any order:
missing, additional, duplicate and reversed edges are rejected. This validates
the fixed profile, not arbitrary GNN dependency graphs. Role-to-value annotations
such as `sZ>g_epistemic` are not supported computational edges.

`ModelParameters` requires `num_channel_factors: 5`, `num_actions: 6`, and
`policy_horizon: 1`. If present, `num_levels_per_factor`, `num_reward_levels`,
and `num_loss_levels` must equal 2, 3, and 3 respectively. Other metadata,
including `planning_horizon`, is descriptive. `Equations`, `Time`, ontology and
joint-axis prose are not executed or checked as a temporal/type system; they
cannot override the fixed numerical implementation. The dense joint fence
itself is checked against that implementation. Validation does not establish
no-peeking, timestamps, corpus cutoffs or the timeliness of observations:
callers must supply correctly timed data.

## Complete joint export

`joint_matrices()` enumerates the 32 complete states `(I,S,U,Theta,Z)` and nine
joint observations `(reward,loss)`, in lexicographic order with the last index
fastest. Actions follow `model.channels.ACTIONS`. It compiles:

- `A[(r,l),s] = A_R[r,I,U,Theta] * A_L[l,S,Theta]`
- `B[next,current,a] = product_k B_k[next_k,current_k,a]`
- `C[(r,l)] = C_R[r] + C_L[l]`
- `D[s] = product_k D_k[s_k]`

`model_to_cogant_gnn()` adds these complete matrices in COGANT's authoritative
`gnn-matrices` fence, with no truncation or invented columns. Independent
coefficient enumeration and an optional real COGANT parser/runtime test verify
the representation. There is no COGANT runtime dependency for ordinary use.

COGANT filters the full joint state and uses one-step risk plus ambiguity.
AlphaCOGANT makes one mean-field filtering sweep and uses expected log preference
plus information gain about Theta. Equal exported kernels therefore do not
establish equal posteriors, policy scores, or selected action sequences.

## Source-bound traces

`bridge.inference.run_inference()` conditions on the current observation before
scoring an action and predicting the next belief. Without supplied observations,
it chooses the modal joint reward/loss prediction and labels the trace
`model_rollout`. Joint selection avoids impossible pairs when the two marginal
modes disagree under deterministic likelihoods.
The trace records exact UTF-8 source and coefficient digests, the filtering
approximation, one-step policy horizon, beliefs, observation pairs, all EFE
components, actions and next predictions. `verify_inference()` replays all fields;
a changed source, score, action or posterior fails verification.
Nonfinite computed action-score components fail before choosing an action or
publishing a trace; numeric overflow is not encoded as a successful inference.

The CLI exposes `validate`, `export`, `infer`, and `verify-trace`; run
`uv run --locked alphacogant --help`. Supplied observations are integer pairs
`[reward, loss]`, each in `[0,2]`. Direct Python and CLI inference retain their
10,000-step research limit; API generation and verification are limited to
1000 steps. JSONL request frames and file-backed trace JSON have a 32,000,000-byte
budget; source remains limited to 1,000,000 bytes. The JSONL budget accommodates
full source-plus-trace replay at the API's maximum 1000 steps, not every research
trace. File outputs are exclusive unless `--overwrite` is specified. These are
model computations, not market observations or a proof of economic adequacy.
