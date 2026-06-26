"""Sensitivity analysis: how t-RSI responds to belief precision.

The manuscript reports t-RSI at a single operating point with a single
Dirichlet concentration.  This module sweeps concentration and (optionally)
the prior's Theta-freshness to show *how robust* the headline number is to
the firm's belief precision — the modelling choice that most directly
controls how tightly the bootstrap perturbations hug the operating belief.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from alphacogant.generative_model import EconomicWorldModel, validate_belief_map
from alphacogant.operating_points import BOOTSTRAP_SEED
from alphacogant.t_rsi import bootstrap_t_rsi


def sweep_concentration(
    model: EconomicWorldModel,
    belief: Mapping[str, np.ndarray],
    *,
    concentrations: np.ndarray | None = None,
    seed: int = BOOTSTRAP_SEED,
    n: int = 256,
) -> dict[str, np.ndarray]:
    """Bootstrap t-RSI across a range of Dirichlet concentrations.

    Returns a dict with arrays ``concentrations``, ``t_rsi``,
    ``create_mean``, ``decay_mean``, ``create_se``, ``decay_se``.
    """
    if concentrations is None:
        concentrations = np.array([2.0, 4.0, 8.0, 12.0, 20.0, 40.0, 80.0])
    normalized = validate_belief_map(belief, context="belief")
    t_rsi_vals = np.empty_like(concentrations, dtype=float)
    create_means = np.empty_like(concentrations, dtype=float)
    decay_means = np.empty_like(concentrations, dtype=float)
    create_ses = np.empty_like(concentrations, dtype=float)
    decay_ses = np.empty_like(concentrations, dtype=float)

    for idx, conc in enumerate(concentrations):
        result = bootstrap_t_rsi(
            model,
            normalized,
            np.random.default_rng(seed),
            n=n,
            concentration=float(conc),
        )
        t_rsi_vals[idx] = result["t_rsi"]
        create_means[idx] = result["create_mean"]
        decay_means[idx] = result["decay_mean"]
        create_ses[idx] = result["create_se"]
        decay_ses[idx] = result["decay_se"]

    return {
        "concentrations": concentrations,
        "t_rsi": t_rsi_vals,
        "create_mean": create_means,
        "decay_mean": decay_means,
        "create_se": create_ses,
        "decay_se": decay_ses,
    }


def sweep_theta_freshness(
    model: EconomicWorldModel,
    base_belief: Mapping[str, np.ndarray],
    *,
    theta_values: np.ndarray | None = None,
    seed: int = BOOTSTRAP_SEED,
    n: int = 256,
    concentration: float = 12.0,
) -> dict[str, np.ndarray]:
    """Bootstrap t-RSI as the Theta-freshness prior is swept.

    This shows how t-RSI changes as the firm's belief about its own
    parameter quality moves from stale to fresh.
    """
    if theta_values is None:
        theta_values = np.linspace(0.05, 0.95, 19)
    normalized = validate_belief_map(base_belief, context="base_belief")
    t_rsi_vals = np.empty_like(theta_values, dtype=float)
    create_means = np.empty_like(theta_values, dtype=float)
    decay_means = np.empty_like(theta_values, dtype=float)

    for idx, p_fresh in enumerate(theta_values):
        perturbed = {channel: normalized[channel].copy() for channel in normalized}
        perturbed["Theta"] = np.array([1.0 - p_fresh, p_fresh])
        result = bootstrap_t_rsi(
            model,
            perturbed,
            np.random.default_rng(seed),
            n=n,
            concentration=concentration,
        )
        t_rsi_vals[idx] = result["t_rsi"]
        create_means[idx] = result["create_mean"]
        decay_means[idx] = result["decay_mean"]

    return {
        "theta_p_fresh": theta_values,
        "t_rsi": t_rsi_vals,
        "create_mean": create_means,
        "decay_mean": decay_means,
    }


__all__ = [
    "sweep_concentration",
    "sweep_theta_freshness",
]
