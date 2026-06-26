"""Tests for deterministic manuscript token generation."""

from __future__ import annotations

from alphacogant.channels import CHANNELS
from alphacogant.free_energy import expected_free_energy, marginal_return_vector
from alphacogant.generative_model import belief_prior, default_model
from alphacogant.manuscript_variables import PLANNING_HORIZON, generate_variables


def test_generate_variables_contains_required_non_empty_strings() -> None:
    variables = generate_variables()
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
        "EPISTEMIC_CHANNELS",
        "PRAGMATIC_CHANNELS",
        "TOKEN",
    }
    assert required.issubset(variables)
    assert all(isinstance(value, str) and value for value in variables.values())


def test_generate_variables_is_deterministic_and_consistent() -> None:
    first = generate_variables()
    second = generate_variables()
    assert first == second
    assert first["FUNDED_CHANNEL"] in CHANNELS
    assert first["NUM_CHANNELS"] == str(len(CHANNELS))
    assert first["NUM_ACTIONS"] == "6"
    assert first["PLANNING_HORIZON"] == str(PLANNING_HORIZON)

    model = default_model()
    prior = belief_prior(model)
    returns = marginal_return_vector(model, prior)
    funded_action = max(range(5), key=lambda action: returns[action])
    funded_result = expected_free_energy(model, prior, funded_action)
    assert first["FUNDED_CHANNEL"] == CHANNELS[funded_action]
    assert first["FUNDED_PRAGMATIC"] == f"{funded_result.pragmatic:.4f}"
    assert first["FUNDED_EPISTEMIC"] == f"{funded_result.epistemic:.4f}"
