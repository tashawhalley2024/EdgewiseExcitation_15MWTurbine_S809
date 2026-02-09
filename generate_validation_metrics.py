# ============================================================
# Quantitative Validation Metrics for Dynamic Stall Models
# ============================================================
# Computes RMSE, MAE, and peak error for CL, CD, CM
# against NREL/OSU S809 experimental data (k = 0.079)
# Generates a grouped bar chart for the report.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experimental_data import load_experimental_data, get_conditions
from dynstall_combined import (
    AeroDynPolar,
    OyeDynamicStallCL,
    LeishmanBeddoesFirstOrder,
    IAGDynamicStallCL,
    NoDynamicStall,
    run_pitch_sine,
)


def rearrange_from_trough(alpha: np.ndarray, data: np.ndarray):
    """
    Rearrange a cyclic signal so it starts from the minimum-alpha point.
    This ensures the loop goes trough → peak → trough, matching the
    experimental data structure.
    """
    trough_idx = int(np.argmin(alpha))
    return (
        np.concatenate([alpha[trough_idx:], alpha[:trough_idx]]),
        np.concatenate([data[trough_idx:], data[:trough_idx]]),
    )


def split_upstroke_downstroke(alpha: np.ndarray, data: np.ndarray):
    """
    Split a trough→peak→trough hysteresis loop into upstroke and downstroke.
    Returns (alpha_up, data_up, alpha_down, data_down), each sorted by alpha.
    """
    peak_idx = int(np.argmax(alpha))

    # Upstroke: from trough to peak (inclusive)
    a_up = alpha[: peak_idx + 1]
    d_up = data[: peak_idx + 1]
    # Downstroke: from peak to next trough
    a_dn = alpha[peak_idx:]
    d_dn = data[peak_idx:]

    # Sort each branch by ascending alpha for interpolation
    idx_up = np.argsort(a_up)
    idx_dn = np.argsort(a_dn)

    return a_up[idx_up], d_up[idx_up], a_dn[idx_dn], d_dn[idx_dn]


def branch_interpolation(
    exp_alpha: np.ndarray,
    exp_data: np.ndarray,
    model_alpha: np.ndarray,
    model_data: np.ndarray,
):
    """
    Interpolate model onto experimental alpha points, matching
    upstroke-to-upstroke and downstroke-to-downstroke separately.

    Model data is first rearranged to start from its trough so
    the phase structure matches the experimental loop (trough → peak → trough).
    Interpolation is restricted to the overlapping alpha range.
    """
    # Rearrange model cycle to start from trough
    model_alpha, model_data = rearrange_from_trough(model_alpha, model_data)

    # Split both into branches
    ea_up, ed_up, ea_dn, ed_dn = split_upstroke_downstroke(exp_alpha, exp_data)
    ma_up, md_up, ma_dn, md_dn = split_upstroke_downstroke(model_alpha, model_data)

    # Restrict to overlapping alpha range per branch to avoid extrapolation
    lo_up = max(ea_up[0], ma_up[0])
    hi_up = min(ea_up[-1], ma_up[-1])
    mask_up = (ea_up >= lo_up) & (ea_up <= hi_up)

    lo_dn = max(ea_dn[0], ma_dn[0])
    hi_dn = min(ea_dn[-1], ma_dn[-1])
    mask_dn = (ea_dn >= lo_dn) & (ea_dn <= hi_dn)

    interp_up = np.interp(ea_up[mask_up], ma_up, md_up)
    interp_dn = np.interp(ea_dn[mask_dn], ma_dn, md_dn)

    model_interp = np.concatenate([interp_up, interp_dn])
    exp_combined = np.concatenate([ed_up[mask_up], ed_dn[mask_dn]])

    return exp_combined, model_interp


def compute_metrics(exp_data, model_interp):
    """Compute RMSE, MAE, and peak absolute error."""
    error = exp_data - model_interp
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    peak = float(np.max(np.abs(error)))
    return rmse, mae, peak


def main():
    # ============================================================
    # 1. Load experimental data
    # ============================================================
    print("Loading experimental validation data...")
    exp = load_experimental_data()

    # ============================================================
    # 2. Load polar and set up models
    # ============================================================
    polar_path = Path(__file__).resolve().parent / "OSU_S809_static_LEGR_1000K.dat"
    polar = AeroDynPolar.from_file(str(polar_path))
    c = 1.0

    models = {
        "Static": NoDynamicStall(),
        "Øye": OyeDynamicStallCL(c=c, Tf0=6.0, use_bergami_cd=True),
        "BL": LeishmanBeddoesFirstOrder(c=c, alpha_crit_rad=np.deg2rad(15.0)),
        "IAG": IAGDynamicStallCL(c=c, ks=0.06, alpha_crit_rad=np.deg2rad(15.0)),
    }

    # ============================================================
    # 3. Run pitch simulation
    # ============================================================
    U = 30.0
    k = 0.079
    freq_hz = k * U / (np.pi * c)
    alpha_mean_deg = 18.5
    alpha_amp_deg = 10.4
    dt = 1e-3
    n_cycles = 3
    t_end = n_cycles / freq_hz

    results = {}
    for name, model in models.items():
        print(f"  Running {name}...")
        r = run_pitch_sine(
            model=model,
            polar=polar,
            dt=dt,
            t_end=t_end,
            alpha_mean_deg=alpha_mean_deg,
            alpha_amp_deg=alpha_amp_deg,
            freq_hz=freq_hz,
            U=U,
        )
        results[name] = r

    # ============================================================
    # 4. Compute metrics
    # ============================================================
    n_per_cycle = int(1.0 / freq_hz / dt)

    coefficients = ["CL", "CD", "CM"]
    model_keys = {"CL": "cl", "CD": "cd", "CM": "cm"}

    # Store: metrics[model_name][coef] = (rmse, mae, peak)
    metrics = {}
    for name, r in results.items():
        metrics[name] = {}
        # Last cycle of model data
        alpha_last = r["alpha_deg"][-n_per_cycle:]
        for coef in coefficients:
            if coef not in exp:
                continue
            model_data_last = r[model_keys[coef]][-n_per_cycle:]
            exp_alpha = exp[coef].alpha_deg
            exp_data_vals = exp[coef].data

            exp_combined, model_interp = branch_interpolation(
                exp_alpha, exp_data_vals, alpha_last, model_data_last
            )
            rmse, mae, peak = compute_metrics(exp_combined, model_interp)
            metrics[name][coef] = (rmse, mae, peak)

    # Print table
    print("\n" + "=" * 72)
    print(f"{'Model':<10} {'Coeff':<6} {'RMSE':>8} {'MAE':>8} {'Peak':>8}")
    print("-" * 72)
    for name in models:
        for coef in coefficients:
            if coef in metrics[name]:
                rmse, mae, peak = metrics[name][coef]
                print(f"{name:<10} {coef:<6} {rmse:8.4f} {mae:8.4f} {peak:8.4f}")
    print("=" * 72)

    # ============================================================
    # 5. Generate grouped bar chart
    # ============================================================
    model_names = list(models.keys())
    n_models = len(model_names)
    n_coefs = len(coefficients)

    # Colours matching run_validation.py
    colors = {
        "Static": "gray",
        "Øye": "C0",
        "BL": "C1",
        "IAG": "C2",
    }

    bar_width = 0.18
    x = np.arange(n_coefs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ------- Left panel: RMSE -------
    ax = axes[0]
    for j, name in enumerate(model_names):
        vals = [metrics[name][c][0] for c in coefficients]   # RMSE
        bars = ax.bar(
            x + j * bar_width, vals, bar_width,
            label=name, color=colors[name], edgecolor="black", linewidth=0.4,
        )
        # Annotate values
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7,
            )
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([r"$C_L$", r"$C_D$", r"$C_M$"], fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("Root-Mean-Square Error", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    # ------- Right panel: Peak absolute error -------
    ax = axes[1]
    for j, name in enumerate(model_names):
        vals = [metrics[name][c][2] for c in coefficients]   # Peak
        bars = ax.bar(
            x + j * bar_width, vals, bar_width,
            label=name, color=colors[name], edgecolor="black", linewidth=0.4,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7,
            )
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([r"$C_L$", r"$C_D$", r"$C_M$"], fontsize=11)
    ax.set_ylabel("Peak Absolute Error", fontsize=11)
    ax.set_title("Maximum Absolute Error", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    fig.suptitle(
        r"S809 Validation Error Metrics — $k = 0.079$, "
        r"$\alpha = 18.5°\pm 10.4°$",
        fontsize=12,
    )
    plt.tight_layout()

    output_path = Path(__file__).parent / "validation_error_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
