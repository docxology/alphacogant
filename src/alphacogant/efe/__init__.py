"""Expected Free Energy subpackage: EFE computation, marginal returns, policy posterior."""

from alphacogant.efe.free_energy import (
    EFEResult,
    expected_free_energy,
    marginal_return_vector,
    policy_posterior,
    static_pragmatic_value,
)

__all__ = [
    "EFEResult",
    "expected_free_energy",
    "marginal_return_vector",
    "policy_posterior",
    "static_pragmatic_value",
]
