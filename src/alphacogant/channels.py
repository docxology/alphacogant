"""Channel and action metadata for the AlphaCOGANT Active Inference engine."""

from __future__ import annotations

from dataclasses import dataclass

CHANNELS: tuple[str, ...] = ("I", "S", "U", "Theta", "Z")

ACTIONS: tuple[str, ...] = (
    "fund_I",
    "fund_S",
    "fund_U",
    "fund_Theta",
    "fund_Z",
    "hold",
)

CHANNEL_ROLES: dict[str, str] = {
    "I": "pragmatic",
    "S": "epistemic",
    "U": "pragmatic",
    "Theta": "both",
    "Z": "epistemic",
}


@dataclass(frozen=True)
class Channel:
    """Metadata for a two-level hidden-state factor."""

    name: str
    role: str
    levels: int = 2

    def __post_init__(self) -> None:
        if self.name not in CHANNELS:
            raise ValueError(f"Unknown channel name {self.name!r}; expected one of {CHANNELS}.")
        if self.role not in {"pragmatic", "epistemic", "both"}:
            raise ValueError(
                "Channel role must be 'pragmatic', 'epistemic', or 'both'."
            )
        if self.levels != 2:
            raise ValueError("AlphaCOGANT channels are binary factors and must have 2 levels.")


def channel_index(name: str) -> int:
    """Return the index of a channel in the fixed factor order."""
    if name not in CHANNELS:
        raise ValueError(f"Unknown channel {name!r}; expected one of {CHANNELS}.")
    return CHANNELS.index(name)


def action_index(name: str) -> int:
    """Return the index of an action in the fixed control order."""
    if name not in ACTIONS:
        raise ValueError(f"Unknown action {name!r}; expected one of {ACTIONS}.")
    return ACTIONS.index(name)


__all__ = [
    "ACTIONS",
    "CHANNELS",
    "CHANNEL_ROLES",
    "Channel",
    "action_index",
    "channel_index",
]
