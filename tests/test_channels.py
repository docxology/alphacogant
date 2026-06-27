"""Tests for channel metadata and lookup helpers."""

from __future__ import annotations

import pytest

from alphacogant.model.channels import ACTIONS, CHANNELS, Channel, action_index, channel_index


def test_channel_metadata_and_lookup() -> None:
    channel = Channel(name="Theta", role="both")
    assert channel.levels == 2
    assert channel_index("I") == 0
    assert channel_index("Z") == len(CHANNELS) - 1
    assert action_index("fund_Theta") == 3
    assert action_index("hold") == len(ACTIONS) - 1


def test_channel_validation_errors() -> None:
    with pytest.raises(ValueError, match="Unknown channel"):
        channel_index("Q")
    with pytest.raises(ValueError, match="Unknown channel name"):
        Channel(name="Q", role="pragmatic")
    with pytest.raises(ValueError, match="Unknown action"):
        action_index("invest")
    with pytest.raises(ValueError, match="role"):
        Channel(name="I", role="control")
    with pytest.raises(ValueError, match="2 levels"):
        Channel(name="I", role="pragmatic", levels=3)
