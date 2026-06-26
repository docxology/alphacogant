"""AlphaCOGANT — the AlphaFund self-improving corporation as an Active Inference agent.

The five capital channels (Investments, Sensors, Actuators, Parameters, R&D) are
hidden-state factors of a partially-observed generative model (the Economic World
Model); capital allocation is the control vector; Expected Free Energy is the
portfolio optimizer's marginal-return objective, decomposed into pragmatic value
(expected log-equity / alpha-creation) and epistemic value (information gain on the
EWM / what Sensors and R&D buy); t-RSI is the standardized create-vs-decay distance
that gates each self-improvement commit.

See ``SPEC.md`` for the module contract and ``models/alphafund_ewm.md`` for the GNN
specification this package realizes.
"""

from alphacogant.channels import ACTIONS, CHANNELS, CHANNEL_ROLES, Channel, action_index, channel_index
from alphacogant.cogant_bridge import firm_structure_to_channels, model_to_gnn_summary, parse_gnn_summary
from alphacogant.free_energy import EFEResult, expected_free_energy, marginal_return_vector, policy_posterior, static_pragmatic_value
from alphacogant.generative_model import (
    EconomicWorldModel,
    belief_prior,
    default_model,
    infer_states,
    validate_belief_map,
)
from alphacogant.operating_points import COASTING, COASTING_RAW, IMPROVING, IMPROVING_RAW, as_belief
from alphacogant.sensitivity import sweep_concentration, sweep_theta_freshness
from alphacogant.simulation import CycleRecord, TrajectoryResult, simulate_trajectory, summarize_trajectory
from alphacogant.statistics import (
    BootstrapCI,
    RegimeComparison,
    RegimeStatistics,
    bootstrap_ci,
    compare_regimes,
    compute_regime_statistics,
)
from alphacogant.t_rsi import (
    DEFAULT_HORIZON,
    bootstrap_t_rsi,
    certificate,
    create_rate,
    decay_rate,
    t_rsi,
)

__version__ = "0.3.0"

__all__ = [
    # channels
    "ACTIONS",
    "CHANNELS",
    "CHANNEL_ROLES",
    "Channel",
    "action_index",
    "channel_index",
    # generative_model
    "EconomicWorldModel",
    "belief_prior",
    "default_model",
    "infer_states",
    "validate_belief_map",
    # cogant_bridge
    "firm_structure_to_channels",
    "model_to_gnn_summary",
    "parse_gnn_summary",
    # free_energy
    "EFEResult",
    "expected_free_energy",
    "marginal_return_vector",
    "policy_posterior",
    "static_pragmatic_value",
    # t_rsi
    "DEFAULT_HORIZON",
    "bootstrap_t_rsi",
    "certificate",
    "create_rate",
    "decay_rate",
    "t_rsi",
    # operating_points
    "COASTING",
    "COASTING_RAW",
    "IMPROVING",
    "IMPROVING_RAW",
    "as_belief",
    # simulation
    "CycleRecord",
    "TrajectoryResult",
    "simulate_trajectory",
    "summarize_trajectory",
    # sensitivity
    "sweep_concentration",
    "sweep_theta_freshness",
    # statistics
    "BootstrapCI",
    "RegimeComparison",
    "RegimeStatistics",
    "bootstrap_ci",
    "compare_regimes",
    "compute_regime_statistics",
    # metadata
    "__version__",
]
