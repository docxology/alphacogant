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
from alphacogant.free_energy import EFEResult, expected_free_energy, marginal_return_vector, policy_posterior, static_pragmatic_value
from alphacogant.generative_model import (
    EconomicWorldModel,
    belief_prior,
    default_model,
    infer_states,
    validate_belief_map,
)
from alphacogant.operating_points import COASTING, COASTING_RAW, IMPROVING, IMPROVING_RAW, as_belief
from alphacogant.t_rsi import (
    DEFAULT_HORIZON,
    bootstrap_t_rsi,
    certificate,
    create_rate,
    decay_rate,
    t_rsi,
)

__version__ = "0.1.0"

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
    # metadata
    "__version__",
]
