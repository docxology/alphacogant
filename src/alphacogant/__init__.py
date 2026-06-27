"""AlphaCOGANT — the AlphaFund self-improving corporation as an Active Inference agent.

The five capital channels (Investments, Sensors, Actuators, Parameters, R&D) are
hidden-state factors of a partially-observed generative model (the Economic World
Model); capital allocation is the control vector; Expected Free Energy is the
portfolio optimizer's marginal-return objective, decomposed into pragmatic value
(expected log-equity / alpha-creation) and epistemic value (information gain on the
EWM / what Sensors and R&D buy); t-RSI is the standardized create-vs-decay distance
that gates each self-improvement commit.

Subpackage layout:
  model/    — channels, generative model, operating points
  efe/      — Expected Free Energy computation, marginal returns, policy posterior
  trsi/     — create/decay rates, standardized distance, certificate, bootstrap
  bridge/   — COGANT firm structure → priors → GNN summary round-trip
  viz/      — shared plot style and color palettes
  stats/    — simulation, sensitivity analysis, bootstrap CIs, regime comparison
  tokens/   — deterministic manuscript {{TOKEN}} generation

See ``SPEC.md`` for the module contract and ``models/alphafund_ewm.md`` for the GNN
specification this package realizes.
"""

# ── Model subpackage ─────────────────────────────────────────────────────────
# ── Bridge subpackage ────────────────────────────────────────────────────────
from alphacogant.bridge import (
    firm_structure_to_channels,
    model_to_gnn_summary,
    parse_gnn_summary,
)

# ── EFE subpackage ───────────────────────────────────────────────────────────
from alphacogant.efe import (
    EFEResult,
    expected_free_energy,
    marginal_return_vector,
    policy_posterior,
    static_pragmatic_value,
)
from alphacogant.model import (
    ACTIONS,
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CHANNEL_ROLES,
    CHANNELS,
    COASTING,
    COASTING_RAW,
    IMPROVING,
    IMPROVING_RAW,
    Channel,
    EconomicWorldModel,
    action_index,
    as_belief,
    belief_prior,
    channel_index,
    default_model,
    infer_states,
    validate_belief_map,
)

# ── Stats subpackage ─────────────────────────────────────────────────────────
from alphacogant.stats import (
    BootstrapCI,
    BreakEvenProfile,
    CycleRecord,
    RegimeComparison,
    RegimeStatistics,
    TrajectoryResult,
    bootstrap_ci,
    break_even_profile,
    compare_regimes,
    compute_regime_statistics,
    simulate_trajectory,
    summarize_trajectory,
    sweep_concentration,
    sweep_theta_freshness,
)

# ── Tokens subpackage ─────────────────────────────────────────────────────────
from alphacogant.tokens import (
    PLANNING_HORIZON,
    generate_variables,
)

# ── t-RSI subpackage ─────────────────────────────────────────────────────────
from alphacogant.trsi import (
    DEFAULT_HORIZON,
    bootstrap_t_rsi,
    certificate,
    create_rate,
    decay_rate,
    t_rsi,
)
from alphacogant.viz import plot_style as plot_style

__version__ = "1.0.0"

__all__ = [
    # model
    "ACTIONS",
    "BOOTSTRAP_CONCENTRATION",
    "BOOTSTRAP_N",
    "BOOTSTRAP_SEED",
    "CHANNELS",
    "CHANNEL_ROLES",
    "COASTING",
    "COASTING_RAW",
    "Channel",
    "EconomicWorldModel",
    "IMPROVING",
    "IMPROVING_RAW",
    "action_index",
    "as_belief",
    "belief_prior",
    "channel_index",
    "default_model",
    "infer_states",
    "validate_belief_map",
    # efe
    "EFEResult",
    "expected_free_energy",
    "marginal_return_vector",
    "policy_posterior",
    "static_pragmatic_value",
    # trsi
    "DEFAULT_HORIZON",
    "bootstrap_t_rsi",
    "certificate",
    "create_rate",
    "decay_rate",
    "t_rsi",
    # bridge
    "firm_structure_to_channels",
    "model_to_gnn_summary",
    "parse_gnn_summary",
    # stats
    "BreakEvenProfile",
    "BootstrapCI",
    "CycleRecord",
    "RegimeComparison",
    "RegimeStatistics",
    "TrajectoryResult",
    "break_even_profile",
    "bootstrap_ci",
    "compare_regimes",
    "compute_regime_statistics",
    "simulate_trajectory",
    "summarize_trajectory",
    "sweep_concentration",
    "sweep_theta_freshness",
    "plot_style",
    # tokens
    "PLANNING_HORIZON",
    "generate_variables",
    # metadata
    "__version__",
]
