"""Expected Free Energy subpackage: EFE computation, marginal returns, policy posterior."""

from alphacogant.efe.free_energy import (
    EFEResult,
    efe_vector,
    expected_free_energy,
    greedy_action,
    marginal_return_vector,
    policy_posterior,
    predicted_belief,
    static_pragmatic_value,
)

__all__ = [
    "EFEResult",
    "efe_vector",
    "expected_free_energy",
    "greedy_action",
    "marginal_return_vector",
    "policy_posterior",
    "predicted_belief",
    "static_pragmatic_value",
]
