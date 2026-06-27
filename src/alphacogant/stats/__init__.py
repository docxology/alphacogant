"""Statistics subpackage: simulation, sensitivity analysis, bootstrap CIs, regime comparison."""

from alphacogant.stats.sensitivity import (
    sweep_concentration,
    sweep_theta_freshness,
)
from alphacogant.stats.simulation import (
    CycleRecord,
    TrajectoryResult,
    simulate_trajectory,
    summarize_trajectory,
)
from alphacogant.stats.statistics import (
    BootstrapCI,
    BreakEvenProfile,
    RegimeComparison,
    RegimeStatistics,
    bootstrap_ci,
    break_even_profile,
    compare_regimes,
    compute_regime_statistics,
)

__all__ = [
    "BreakEvenProfile",
    "BootstrapCI",
    "CycleRecord",
    "RegimeComparison",
    "RegimeStatistics",
    "TrajectoryResult",
    "break_even_profile",
    "bootstrap_ci",
    "compare_regimes",
    "compute_regime_statistics",
    "simulate_trajectory",
    "summarize_trajectory",
    "sweep_concentration",
    "sweep_theta_freshness",
]
