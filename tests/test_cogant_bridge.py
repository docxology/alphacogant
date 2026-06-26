"""Tests for COGANT bridge helpers."""

from __future__ import annotations

import numpy as np
import pytest

from alphacogant.cogant_bridge import firm_structure_to_channels, model_to_gnn_summary


def test_richer_firm_structure_strengthens_channel_priors() -> None:
    lean = firm_structure_to_channels(
        {"data_feeds": 1, "venues": 1, "models": 1, "researchers": 1, "book_size": 1}
    )
    rich = firm_structure_to_channels(
        {"data_feeds": 20, "venues": 18, "models": 12, "researchers": 10, "book_size": 25}
    )
    for channel in lean:
        assert rich[channel][1] >= lean[channel][1]
        assert np.allclose(rich[channel].sum(), 1.0)


def test_firm_structure_validation_errors() -> None:
    with pytest.raises(ValueError, match="mapping"):
        firm_structure_to_channels(["feeds", "venues"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing"):
        firm_structure_to_channels({"data_feeds": 1})
    with pytest.raises(ValueError, match="non-negative"):
        firm_structure_to_channels(
            {"data_feeds": -1, "venues": 1, "models": 1, "researchers": 1, "book_size": 1}
        )
    with pytest.raises(ValueError, match="finite"):
        firm_structure_to_channels(
            {
                "feeds": float("inf"),
                "routes": 1,
                "ewm_models": 1,
                "research_staff": 1,
                "aum": 1,
            }
        )


def test_model_to_gnn_summary_mentions_all_channels_and_efe(model) -> None:
    summary = model_to_gnn_summary(model)
    for token in ("I", "S", "U", "Theta", "Z", "ExpectedFreeEnergy"):
        assert token in summary
