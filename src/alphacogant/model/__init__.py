"""Model subpackage: channels, generative model, and operating points."""

from alphacogant.model.channels import (
    ACTIONS,
    CHANNEL_ROLES,
    CHANNELS,
    Channel,
    action_index,
    channel_index,
)
from alphacogant.model.generative_model import (
    EconomicWorldModel,
    belief_prior,
    default_model,
    infer_states,
    validate_belief_map,
)
from alphacogant.model.operating_points import (
    BOOTSTRAP_CONCENTRATION,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COASTING,
    COASTING_RAW,
    IMPROVING,
    IMPROVING_RAW,
    as_belief,
)

__all__ = [
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
]
