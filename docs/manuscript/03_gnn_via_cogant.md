# Authored GNN and bounded COGANT matrix interoperability {#sec:gnn}

## GNN declares model structure; the local parser enforces a finite profile

Generalized Notation Notation (GNN) is a text-based specification language for
Active Inference generative models, with a processing pipeline that transforms a
GNN `.md` file into executable simulations (PyMDP, RxInfer.jl, JAX, Active
Inference.jl, and others), visualizations, type-checked validations, and reports.
This pipeline and model contract are the source of our representation [@friedman2026gnn].
A GNN model is a Markdown document with structured sections: a `StateSpaceBlock`
declaring tensors (the $A$ likelihood, $B$ transitions, $C$ preferences, $D$
priors, hidden states $s$, observations $o$, policy $\pi$, Expected Free Energy
$G$), a `Connections` block giving the factor-graph edges, an
`InitialParameterization` with concrete numbers, an `Equations` block, and an
`ActInfOntologyAnnotation` binding every symbol to its Active Inference role.

GNN supplies an inspectable declaration, coefficient and dependency contract.
This project's parser enforces its supported fixed declarations, graph and
numeric profile, and the compiler preserves axes in complete joint matrices.
Equations, time and ontology annotations are descriptive, not an executed
temporal type system or proof of filtration integrity. Reviewers can inspect
the state factors, likelihoods, transitions and action edges without mistaking
that inspectability for a verified financial controller.

## COGANT inspires a structure-to-prior mapping, not corporate extraction

COGANT is a **codebase-to-GNN translator** [@friedman2026cogant]: it scans a system's structure
(program graph, modules, call edges), builds a state-space factor graph, and
exports a GNN generative model of the system, which it then renders, visualizes,
and validates against the GNN package. The conceptual claim of AlphaCOGANT is that
the differentiable corporation is a *natural* COGANT input, because AlphaFund's
**third structural fact** is that the firm is "API-complete": data ingestion,
model training, capital allocation, trade execution, and asset acquisition are all
function calls, each producing a structured record that "doubles as the causal
record needed for a derivative against it." Such API structure motivates a
COGANT-style representation; scanability alone cannot identify economic
transition coefficients or causal effects.

The local authored translation has three stages:

1. **Structure → channels.** A firm description — counts of data feeds, execution
   venues, deployed models, researchers, and book size — maps onto the five
   channel factors and their prior beliefs $D_k$. Counts are transformed by a
   logistic-scaled `log1p` map, not interpreted as validated capability measurements.
   This is `alphacogant.bridge.cogant_bridge.firm_structure_to_channels`.
   This is a hand-authored illustrative mapping inspired by COGANT
   [@friedman2026cogant], not inferred financial dynamics.

2. **Channels → GNN model.** The five factors, their likelihood matrices ($A_R$
   for the broker-ledger reward readout, $A_L$ for the predictive-loss readout),
   their per-action transition matrices $B_k$, the log-preferences $C_R, C_L$, and
   the Expected-Free-Energy block are emitted as a GNN file
   (`models/alphafund_ewm.md`). `cogant_bridge.model_to_gnn_summary` emits every
   live coefficient; `bridge.gnn.model_from_gnn` reconstructs and validates the
   bounded finite profile without changing its axes.

3. **GNN model → rollouts and value.** The local NumPy engine performs
   single-sweep factorized inference and Theta-information EFE scoring.
   Complete joint-matrix export is independently parsed and executed by pinned
   COGANT. That runtime's joint filtering and risk-plus-ambiguity objective are
   not policy-equivalent to the local engine.

## The authored file fixes factors, observation likelihoods and action axes

`models/alphafund_ewm.md` encodes the five whitepaper channels as hidden-state
factors `sI, sS, sU, sTheta, sZ`, each with two capability levels
$\{\text{weak}, \text{strong}\}$. The control vector has six actions —
`fund_I, fund_S, fund_U, fund_Theta, fund_Z, hold` — so a cycle's decision is
"which channel receives a discrete funding move." Two bucketed observations are
modeled: reward `o_R` and predictive loss `o_L`, inspired by the historical
broker-ledger and forecast-evaluation roles. Neither is calibrated to a measured
log-equity series. The likelihood $A_R$ makes high reward probable only
when the production channels $I, U$ are strong **and** the parameters $\Theta$ are
fresh; $A_L$ makes low predictive loss probable only when Sensors are rich **and**
$\Theta$ is fresh. The transition $B_\Theta$ illustrates parameter-capability
decay toward stale under non-refit actions and refresh when Theta is funded.
These coefficients are illustrative, not independently measured corporate rates.

[@fig:factorgraph] draws the reduced model's computational dependencies:
the {{NUM_CHANNELS}} hidden-state factors `sI, sS, sU, sTheta, sZ`, the likelihoods
$A_R, A_L$, and per-factor transitions $B_k$ under {{NUM_ACTIONS}} actions.
The EFE annotations distinguish expected preference from Theta-information gain.
Z has no likelihood or value edge. The visible dependency structure does not
authenticate when caller-supplied observations became available.

![Computational factor graph of the authored reduced model: {{NUM_CHANNELS}} hidden-state factors, reward likelihood $A_R$ coupling $I,U,\Theta$, loss likelihood $A_L$ coupling $S,\Theta$, and per-factor transitions $B_k$ under {{NUM_ACTIONS}} actions. EFE minimizes G = -(pragmatic + Theta-information gain). Z has no likelihood/value edge; conceptual R&D roles are not computational dependencies. Structure and coefficients come from `generative_model.default_model`.](../output/figures/gnn_factor_graph.png){#fig:factorgraph}

The reduced encoding is legible and numerically validated. Neither this engine
nor its interchange tests implement a continuous production portfolio controller
[@westenhaver2026rsi]. The result is a bounded executable model, not numerical
fidelity to proprietary surfaces or proof of upstream-wide language support.
