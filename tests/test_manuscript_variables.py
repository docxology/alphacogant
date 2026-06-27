"""Tests for deterministic manuscript token generation."""

from __future__ import annotations

from alphacogant.efe.free_energy import expected_free_energy, marginal_return_vector
from alphacogant.model.channels import CHANNELS
from alphacogant.model.generative_model import belief_prior, default_model
from alphacogant.model.operating_points import BOOTSTRAP_N
from alphacogant.tokens.manuscript_variables import (
    PLANNING_HORIZON,
    generate_variables,
    resolve_bootstrap_n,
)

# Small bootstrap count: the token contract (keys, determinism, formatting,
# behaviour) is independent of the resample count, so tests run fast while the
# published manuscript values use the full default ``BOOTSTRAP_N``.
FAST_N = 32


def test_resolve_bootstrap_n_defaults_to_published_value() -> None:
    assert resolve_bootstrap_n(None) == BOOTSTRAP_N
    assert resolve_bootstrap_n(FAST_N) == FAST_N


def test_generate_variables_contains_required_non_empty_strings() -> None:
    variables = generate_variables(n=FAST_N)
    required = {
        "HEADLINE_T_RSI",
        "CREATE_RATE_MEAN",
        "DECAY_RATE_MEAN",
        "FUNDED_CHANNEL",
        "FUNDED_EPISTEMIC",
        "FUNDED_PRAGMATIC",
        "NUM_CHANNELS",
        "NUM_ACTIONS",
        "PLANNING_HORIZON",
        "BOOTSTRAP_N",
        "EPISTEMIC_CHANNELS",
        "PRAGMATIC_CHANNELS",
    }
    assert required.issubset(variables)
    assert all(isinstance(value, str) and value for value in variables.values())


def test_generate_variables_is_deterministic_and_consistent() -> None:
    first = generate_variables(n=FAST_N)
    second = generate_variables(n=FAST_N)
    assert first == second
    assert first["FUNDED_CHANNEL"] in CHANNELS
    assert first["NUM_CHANNELS"] == str(len(CHANNELS))
    assert first["NUM_ACTIONS"] == "6"
    assert first["PLANNING_HORIZON"] == str(PLANNING_HORIZON)
    assert first["BOOTSTRAP_N"] == str(FAST_N)

    model = default_model()
    prior = belief_prior(model)
    returns = marginal_return_vector(model, prior)
    funded_action = max(range(5), key=lambda action: returns[action])
    funded_result = expected_free_energy(model, prior, funded_action)
    assert first["FUNDED_CHANNEL"] == CHANNELS[funded_action]
    assert first["FUNDED_PRAGMATIC"] == f"{funded_result.pragmatic:.4f}"
    assert first["FUNDED_EPISTEMIC"] == f"{funded_result.epistemic:.4f}"
