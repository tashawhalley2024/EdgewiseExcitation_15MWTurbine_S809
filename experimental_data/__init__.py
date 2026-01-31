# ============================================================
# Experimental Validation Data Package
# ============================================================
# S809 Pitching Airfoil - Dynamic Stall Measurements
# Source: Figure 4, Ref. [41] (OSU wind tunnel)
# Reduced frequency: k = 0.079
# ============================================================

from .exp_validation_loader import (
    load_experimental_data,
    get_conditions,
    ExperimentalDataset,
    ExperimentalConditions,
    plot_experimental_hysteresis,
    compute_rms_error,
    compute_hysteresis_area,
)

__all__ = [
    'load_experimental_data',
    'get_conditions',
    'ExperimentalDataset',
    'ExperimentalConditions',
    'plot_experimental_hysteresis',
    'compute_rms_error',
    'compute_hysteresis_area',
]
