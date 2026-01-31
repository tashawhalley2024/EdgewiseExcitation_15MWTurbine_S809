# ============================================================
# Experimental Validation Data Loader for Dynamic Stall Models
# ============================================================
# This module loads and provides access to digitised experimental
# data from Figure 4 of the WES dynamic stall paper (Ref. [41]).
#
# Usage:
#   from experimental_data.exp_validation_loader import load_experimental_data
#   exp = load_experimental_data()
#   alpha_cl, cl_exp = exp['CL']['alpha_deg'], exp['CL']['data']
# ============================================================

from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os


# ============================================================
# Metadata for the experimental dataset
# ============================================================

@dataclass
class ExperimentalConditions:
    """Metadata for the S809 pitching airfoil experiment."""
    airfoil: str = "S809"
    source: str = "Figure 4, Ref. [41] (OSU wind tunnel)"
    motion: str = "Sinusoidal pitch oscillation"
    reduced_frequency_k: float = 0.079
    mean_alpha_deg: float = 14.0
    amplitude_deg: float = 10.0
    reynolds_number: float = 1.0e6
    data_type: str = "DYNAMIC (hysteresis loop)"
    
    def __repr__(self) -> str:
        return (
            f"Experimental Conditions:\n"
            f"  Airfoil: {self.airfoil}\n"
            f"  Source: {self.source}\n"
            f"  Motion: {self.motion}\n"
            f"  Reduced frequency k: {self.reduced_frequency_k}\n"
            f"  Mean AoA: {self.mean_alpha_deg}°\n"
            f"  Amplitude: {self.amplitude_deg}°\n"
            f"  Reynolds number: {self.reynolds_number:.2e}\n"
            f"  Data type: {self.data_type}"
        )


# ============================================================
# Data loading functions
# ============================================================

def _get_data_dir() -> Path:
    """Get the directory containing experimental data files."""
    return Path(__file__).parent


def load_csv_with_comments(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV file, skipping comment lines starting with #.
    Returns (alpha_deg, data) arrays.
    """
    alpha_list = []
    data_list = []
    
    with open(filepath, "r", encoding="utf-8-sig") as f:

        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            # Skip header line
            if 'alpha' in line.lower():
                continue
            # Parse data
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    alpha_list.append(float(parts[0]))
                    data_list.append(float(parts[1]))
                except ValueError:
                    continue
    
    return np.array(alpha_list), np.array(data_list)


@dataclass
class ExperimentalDataset:
    """Container for a single experimental coefficient dataset."""
    alpha_deg: np.ndarray
    data: np.ndarray
    coefficient: str  # 'CL', 'CD', or 'CM'
    
    @property
    def alpha_rad(self) -> np.ndarray:
        """Return angle of attack in radians."""
        return np.deg2rad(self.alpha_deg)
    
    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.alpha_deg)
    
    @property
    def alpha_range(self) -> Tuple[float, float]:
        """Min and max alpha in degrees."""
        return float(np.min(self.alpha_deg)), float(np.max(self.alpha_deg))
    
    def interpolate(self, alpha_deg_query: np.ndarray) -> np.ndarray:
        """
        Interpolate experimental data at specified alpha values.
        Note: Use with caution - hysteresis loops are not single-valued!
        """
        # Sort by alpha for interpolation
        idx = np.argsort(self.alpha_deg)
        return np.interp(alpha_deg_query, self.alpha_deg[idx], self.data[idx])


def load_experimental_data(data_dir: Optional[Path] = None) -> Dict[str, ExperimentalDataset]:
    """
    Load all experimental validation data.
    
    Returns:
        Dictionary with keys 'CL', 'CD', 'CM' containing ExperimentalDataset objects.
    
    Example:
        exp = load_experimental_data()
        print(exp['CL'].alpha_deg, exp['CL'].data)
    """
    if data_dir is None:
        data_dir = _get_data_dir()
    else:
        data_dir = Path(data_dir)
    
    files = {
        'CL': 'S809_PitchingAirfoil_Exp_CL_k0p079_Fig4.csv',
        'CD': 'S809_PitchingAirfoil_Exp_CD_k0p079_Fig4.csv',
        'CM': 'S809_PitchingAirfoil_Exp_CM_k0p079_Fig4.csv',
    }
    
    result = {}
    for coef, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            alpha, data = load_csv_with_comments(filepath)
            result[coef] = ExperimentalDataset(
                alpha_deg=alpha,
                data=data,
                coefficient=coef
            )
        else:
            print(f"Warning: Could not find {filepath}")
    
    return result


def get_conditions() -> ExperimentalConditions:
    """Return the experimental conditions metadata."""
    return ExperimentalConditions()


# ============================================================
# Plotting utilities for validation
# ============================================================

def plot_experimental_hysteresis(
    exp_data: Dict[str, ExperimentalDataset],
    model_results: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    figsize: Tuple[float, float] = (12, 4)
):
    """
    Plot experimental hysteresis loops with optional model comparisons.
    
    Args:
        exp_data: Dictionary from load_experimental_data()
        model_results: Optional dict of {model_name: {'alpha_deg': array, 'cl': array, ...}}
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    coefs = ['CL', 'CD', 'CM']
    labels = [r'$C_L$', r'$C_D$', r'$C_M$']
    
    for ax, coef, label in zip(axes, coefs, labels):
        if coef in exp_data:
            exp = exp_data[coef]
            ax.scatter(exp.alpha_deg, exp.data, s=8, c='black', 
                      label='Experiment', alpha=0.6, zorder=10)
        
        # Plot model results if provided
        if model_results is not None:
            colors = plt.cm.tab10.colors
            for i, (model_name, results) in enumerate(model_results.items()):
                key = coef.lower()
                if 'alpha_deg' in results and key in results:
                    ax.plot(results['alpha_deg'], results[key], 
                           label=model_name, color=colors[i % len(colors)], lw=1.5)
        
        ax.set_xlabel(r'$\alpha$ (deg)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle('S809 Pitching Airfoil - Dynamic Stall Validation (k=0.079)', fontsize=11)
    plt.tight_layout()
    return fig, axes


# ============================================================
# Quick validation metrics
# ============================================================

def compute_rms_error(
    exp: ExperimentalDataset,
    model_alpha_deg: np.ndarray,
    model_data: np.ndarray
) -> float:
    """
    Compute RMS error between model and experiment.
    Interpolates model to experimental alpha points.
    """
    # Sort model data for interpolation
    idx = np.argsort(model_alpha_deg)
    model_interp = np.interp(exp.alpha_deg, model_alpha_deg[idx], model_data[idx])
    
    error = exp.data - model_interp
    return float(np.sqrt(np.mean(error**2)))


def compute_hysteresis_area(alpha_deg: np.ndarray, coef: np.ndarray) -> float:
    """
    Compute the area enclosed by a hysteresis loop.
    Positive area indicates counter-clockwise loop (typical for CL).
    """
    # Use shoelace formula
    n = len(alpha_deg)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += alpha_deg[i] * coef[j]
        area -= alpha_deg[j] * coef[i]
    return abs(area) / 2.0


# ============================================================
# Main demo / test
# ============================================================

if __name__ == "__main__":
    # Load data
    print("Loading experimental validation data...")
    exp = load_experimental_data()
    cond = get_conditions()
    
    print("\n" + "="*60)
    print(cond)
    print("="*60)
    
    for coef, dataset in exp.items():
        print(f"\n{coef} data:")
        print(f"  Points: {dataset.n_points}")
        print(f"  Alpha range: {dataset.alpha_range[0]:.1f}° to {dataset.alpha_range[1]:.1f}°")
        print(f"  Data range: {dataset.data.min():.4f} to {dataset.data.max():.4f}")
    
    # Optional: plot
    try:
        import matplotlib.pyplot as plt
        plot_experimental_hysteresis(exp)
        plt.show()
    except ImportError:
        print("\nMatplotlib not available for plotting.")
