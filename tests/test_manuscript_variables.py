"""Tests for deterministic manuscript token generation."""

from __future__ import annotations

from alphacogant.efe.free_energy import expected_free_energy, marginal_return_vector
from alphacogant.model.channels import CHANNELS
from alphacogant.tokens.manuscript_variables import (
    PLANNING_HORIZON,
    generate_variables,
)

# Keep the real seeded computation small; published values use the canonical recipe.
FAST_N = 32


def test_generate_variables_is_deterministic_and_consistent(model, prior) -> None:
    first = generate_variables(n=FAST_N)
    second = generate_variables(n=FAST_N)
    assert first == second
    assert first["FUNDED_CHANNEL"] in CHANNELS
    assert first["NUM_CHANNELS"] == str(len(CHANNELS))
    assert first["NUM_ACTIONS"] == "6"
    assert first["PLANNING_HORIZON"] == str(PLANNING_HORIZON)
    assert first["BOOTSTRAP_N"] == str(FAST_N)

    returns = marginal_return_vector(model, prior)
    funded_action = max(range(5), key=lambda action: returns[action])
    funded_result = expected_free_energy(model, prior, funded_action)
    assert first["FUNDED_CHANNEL"] == CHANNELS[funded_action]
    assert first["FUNDED_PRAGMATIC"] == f"{funded_result.pragmatic:.4f}"
    assert first["FUNDED_EPISTEMIC"] == f"{funded_result.epistemic:.4f}"



def test_generate_variables_requires_available_manuscript_sources(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="Manuscript sources are required"):
        generate_variables(n=2, manuscript_dir=tmp_path)


def test_generate_variables_counts_explicit_source_directory(tmp_path):
    (tmp_path / "00_example.md").write_text(
        "**Definition D1** Test.\n![Example](example.png){#fig:example}\n"
    )
    result = generate_variables(n=2, manuscript_dir=tmp_path)
    assert result["NUM_DEFINITIONS"] == "1"
    assert result["NUM_FIGURES"] == "1"
