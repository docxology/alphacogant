"""Shared scalar admission checks for numerical APIs."""

from __future__ import annotations

import numpy as np


def positive_integer(value: int, name: str) -> None:
    """Reject boolean, fractional, nonfinite and nonpositive loop bounds."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
