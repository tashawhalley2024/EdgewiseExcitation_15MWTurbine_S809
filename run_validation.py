# ============================================================
# Experimental Validation Plot for Dynamic Stall Models
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Global plotting style (rcParams)
# ============================================================
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 14,     # ← 缩小 legend
    "legend.frameon": False,   # ← 去边框
    "figure.titlesize": 26,
    "lines.linewidth": 1.5,
})

from experimental_data import load_experimental_data, get_conditions
from dynstall_combined import (
    AeroDynPolar,
    OyeDynamicStallCL,
    LeishmanBeddoesFirstOrder,
    IAGDynamicStallCL,
    NoDynamicStall,
    run_pitch_sine,
)


def main():
    # ============================================================
    # 1. Load experimental data
    # ============================================================
    exp = load_experimental_data()
    cond = get_conditions()
    print(cond)

    # ============================================================
    # 2. Load polar and set up models
    # ============================================================
    polar_path = Path(__file__).resolve().parent / "OSU_S809_static_LEGR_1000K.dat"
    polar = AeroDynPolar.from_file(str(polar_path))

    c = 1.0
    models = {
        "Static (No DS)": NoDynamicStall(),
        "Øye": OyeDynamicStallCL(c=c, Tf0=6.0, use_bergami_cd=True),
        "Leishman-Beddoes": LeishmanBeddoesFirstOrder(
            c=c, alpha_crit_rad=np.deg2rad(15.0)
        ),
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
        results[name] = run_pitch_sine(
            model=model,
            polar=polar,
            dt=dt,
            t_end=t_end,
            alpha_mean_deg=alpha_mean_deg,
            alpha_amp_deg=alpha_amp_deg,
            freq_hz=freq_hz,
            U=U,
        )

    # ============================================================
    # 4. Plot validation comparison
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    colors = {
        "Static (No DS)": "gray",
        "Øye": "C0",
        "Leishman-Beddoes": "C1",
        "IAG": "C2",
    }

    n_per_cycle = int(1.0 / freq_hz / dt)

    # ---------- CL ----------
    ax = axes[0]
    ax.plot(exp["CL"].alpha_deg, exp["CL"].data, "k--", alpha=0.5)
    ax.scatter(exp["CL"].alpha_deg, exp["CL"].data,
               s=10, c="black", alpha=0.5, label="Experiment")
    for name, r in results.items():
        ax.plot(r["alpha_deg"][-n_per_cycle:], r["cl"][-n_per_cycle:],
                color=colors[name], label=name)
    ax.set_title("Lift Coefficient")
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_L$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", handlelength=2.0, labelspacing=0.4)

    # ---------- CD ----------
    ax = axes[1]
    ax.plot(exp["CD"].alpha_deg, exp["CD"].data, "k--", alpha=0.5)
    ax.scatter(exp["CD"].alpha_deg, exp["CD"].data,
               s=10, c="black", alpha=0.5, label="Experiment")
    for name, r in results.items():
        ax.plot(r["alpha_deg"][-n_per_cycle:], r["cd"][-n_per_cycle:],
                color=colors[name], label=name)
    ax.set_title("Drag Coefficient")
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_D$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", handlelength=2.0, labelspacing=0.4)

    # ---------- CM ----------
    ax = axes[2]
    ax.plot(exp["CM"].alpha_deg, exp["CM"].data, "k--", alpha=0.5)
    ax.scatter(exp["CM"].alpha_deg, exp["CM"].data,
               s=10, c="black", alpha=0.5, label="Experiment")
    for name, r in results.items():
        if "cm" in r:
            ax.plot(r["alpha_deg"][-n_per_cycle:], r["cm"][-n_per_cycle:],
                    color=colors[name], label=name)
    ax.set_title("Pitching Moment")
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_M$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", handlelength=2.0, labelspacing=0.4)

    # ---------- Figure title ----------
    fig.suptitle(
        f"S809 Dynamic Stall Validation — k = {k}, "
        f"α = {alpha_mean_deg}° ± {alpha_amp_deg}°"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(Path(__file__).parent / "validation_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
    
