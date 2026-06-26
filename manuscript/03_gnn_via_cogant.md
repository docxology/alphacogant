# Technical and computational realization: GNN via COGANT

## What GNN is, and why it fits

Generalized Notation Notation (GNN) is a text-based specification language for
Active Inference generative models, with a processing pipeline that transforms a
GNN `.md` file into executable simulations (PyMDP, RxInfer.jl, JAX, Active
Inference.jl, and others), visualizations, type-checked validations, and reports.
A GNN model is a Markdown document with structured sections: a `StateSpaceBlock`
declaring tensors (the $A$ likelihood, $B$ transitions, $C$ preferences, $D$
priors, hidden states $s$, observations $o$, policy $\pi$, Expected Free Energy
$G$), a `Connections` block giving the factor-graph edges, an
`InitialParameterization` with concrete numbers, an `Equations` block, and an
`ActInfOntologyAnnotation` binding every symbol to its Active Inference role.

GNN is the right target for AlphaFund because it is **executable, typed, and
filtration-explicit**. The EWM is not a diagram; it is an object the firm rolls
forward under candidate actions. GNN compiles to exactly such rollouts, and its
type-checker enforces that the declared tensors are well-formed probability
objects — the static discipline AlphaFund needs for an auditable controller.

## What COGANT is, and the translation step

COGANT is a **codebase-to-GNN translator**: it scans a system's structure
(program graph, modules, call edges), builds a state-space factor graph, and
exports a GNN generative model of the system, which it then renders, visualizes,
and validates against the GNN package. The conceptual claim of AlphaCOGANT is that
the differentiable corporation is a *natural* COGANT input, because AlphaFund's
**third structural fact** is that the firm is "API-complete": data ingestion,
model training, capital allocation, trade execution, and asset acquisition are all
function calls, each producing a structured record that "doubles as the causal
record needed for a derivative against it." A system whose operational degrees of
freedom are all API calls with causal records is a system whose structure COGANT
can scan and whose dynamics COGANT can express as $B_k$ transitions.

The translation has three stages, mirrored by the engine in `src/alphacogant/`:

1. **Structure → channels.** A firm description — counts of data feeds, execution
   venues, deployed models, researchers, and book size — maps onto the five
   channel factors and their prior beliefs $D_k$. More feeds strengthen the
   Sensors prior; a larger validated book strengthens Investments; a fresher
   model strengthens Parameters. This is `cogant_bridge.firm_structure_to_channels`.
   It is the COGANT scan in miniature: system structure becomes generative-model
   priors.

2. **Channels → GNN model.** The five factors, their likelihood matrices ($A_R$
   for the broker-ledger reward readout, $A_L$ for the predictive-loss readout),
   their per-action transition matrices $B_k$, the log-preferences $C_R, C_L$, and
   the Expected-Free-Energy block are emitted as a GNN file
   (`models/alphafund_ewm.md`). `cogant_bridge.model_to_gnn_summary` performs the
   round-trip check: an `EconomicWorldModel` re-emits a GNN-style ontology block,
   proving structure was preserved.

3. **GNN model → rollouts and value.** The GNN pipeline (or, here, the NumPy
   engine that implements the same semantics) performs inference and policy
   evaluation: posteriors over the channels, Expected Free Energy per policy split
   into epistemic and pragmatic value, the marginal-return vector, and the t-RSI
   certificate.

## The model file

`models/alphafund_ewm.md` encodes the five whitepaper channels as hidden-state
factors `sI, sS, sU, sTheta, sZ`, each with two capability levels
$\{\text{weak}, \text{strong}\}$. The control vector has six actions —
`fund_I, fund_S, fund_U, fund_Theta, fund_Z, hold` — so a cycle's decision is
"which channel receives the marginal dollar." Two observations are modeled: the
realized log-equity reward `o_R` (the only channel read directly off the broker
ledger, per AlphaFund's Investments row) and the EWM predictive loss `o_L` (the
forecast-evaluation panel). The likelihood $A_R$ makes high reward probable only
when the production channels $I, U$ are strong **and** the parameters $\Theta$ are
fresh; $A_L$ makes low predictive loss probable only when Sensors are rich **and**
$\Theta$ is fresh. The transition $B_\Theta$ encodes AlphaFund's central empirical
finding — that deployed parameters **decay** (alpha-decay) toward stale under any
non-refit action — and refreshes only when $\Theta$ is funded.

[@fig:factorgraph] draws the GNN factor graph this file declares: the
{{NUM_CHANNELS}} hidden-state factors `sI, sS, sU, sTheta, sZ`, the two likelihood
matrices $A_R, A_L$ wiring factors to the reward and loss observations, the
per-factor transitions $B_k$ under the {{NUM_ACTIONS}} actions, and the
Expected-Free-Energy block — the `Connections` block made visible, so a future
observation conditioning a past belief is an edge one can see is absent.

![GNN factor graph of the five-channel firm: {{NUM_CHANNELS}} hidden-state factors, likelihoods $A_R$ (coupling $I, U, \Theta$ to reward $o_R$) and $A_L$ (coupling $S, \Theta$ to loss $o_L$), per-factor transitions $B_k$ over {{NUM_ACTIONS}} actions, and the EFE objective block, with factor structure read from `generative_model.default_model`.](../output/figures/gnn_factor_graph.png){#fig:factorgraph}

The reduced two-level encoding keeps the GNN file legible and type-checkable; the
continuous marginal-return formalism of §5 is what the engine and the whitepaper
use in production. The point of the file is not numerical fidelity to AlphaFund's
proprietary surfaces (which are not public); it is to demonstrate that the firm's
control problem **is** a well-formed Active Inference model with an Expected-Free-
Energy objective, expressible in a language that compiles to executable inference.
