# ============================================================
# Experimental Validation Plot for Dynamic Stall Models
# ============================================================
# Compares Oye, LB, and IAG models against experimental data
# from Figure 4 (S809 pitching airfoil, k=0.079)
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


def main():
    # ============================================================
    # 1. Load experimental data
    # ============================================================
    print("Loading experimental validation data...")
    exp = load_experimental_data()
    cond = get_conditions()
    print(cond)
    
    # ============================================================
    # 2. Load polar and set up models
    # ============================================================
    print("\nLoading S809 polar...")
    # Get polar file path relative to script location
    script_dir = Path(__file__).parent.parent
    polar_path = (Path(__file__).resolve().parent / "OSU_S809_static_LEGR_1000K.dat")

    polar = AeroDynPolar.from_file(str(polar_path))
    
    # Chord length (normalised)
    c = 1.0
    
    # Models to compare
    models = {
        "Static (No DS)": NoDynamicStall(),
        "Øye": OyeDynamicStallCL(c=c, Tf0=6.0, use_bergami_cd=True),
        "Leishman-Beddoes": LeishmanBeddoesFirstOrder(c=c, alpha_crit_rad=np.deg2rad(15.0)),
        "IAG": IAGDynamicStallCL(c=c, ks=0.06, alpha_crit_rad=np.deg2rad(15.0)),
    }
    
    # ============================================================
    # 3. Run pitch simulation matching experimental conditions
    # ============================================================
    # Experimental conditions from Figure 4:
    # - k = 0.079 (reduced frequency)
    # - Mean alpha ~ 14 deg
    # - Amplitude ~ 10 deg
    # - k = pi * f * c / U  =>  f = k * U / (pi * c)
    
    U = 30.0  # freestream velocity (m/s)
    k = 0.079  # reduced frequency
    freq_hz = k * U / (np.pi * c)  # compute frequency from k
    
    # Match experimental data range (8° to 29°)
    alpha_mean_deg = 18.5
    alpha_amp_deg = 10.4
    
    dt = 1e-3
    n_cycles = 3
    t_end = n_cycles / freq_hz
    
    print(f"\nSimulation parameters:")
    print(f"  Reduced frequency k = {k}")
    print(f"  Frequency = {freq_hz:.3f} Hz")
    print(f"  U = {U} m/s")
    print(f"  Alpha = {alpha_mean_deg} ± {alpha_amp_deg} deg")
    print(f"  Running {n_cycles} cycles...")
    
    # Run all models
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
    # 4. Plot validation comparison
    # ============================================================
    print("\nGenerating validation plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Colours for models
    colors = {
        "Static (No DS)": "gray",
        "Øye": "C0",
        "Leishman-Beddoes": "C1",
        "IAG": "C2",
    }
    
    # --- CL subplot ---
    ax = axes[0]
    # Plot experimental data as dashed line with scatter points
    ax.plot(exp['CL'].alpha_deg, exp['CL'].data, 
            'k--', linewidth=1, alpha=0.5)
    ax.scatter(exp['CL'].alpha_deg, exp['CL'].data,
               s=15, c='black', alpha=0.5, zorder=5, label='Experiment')
    for name, r in results.items():
        # Use last cycle only
        n_per_cycle = int(1.0 / freq_hz / dt)
        ax.plot(r['alpha_deg'][-n_per_cycle:], r['cl'][-n_per_cycle:], 
                label=name, color=colors[name], lw=1.5)
    ax.set_xlabel(r'$\alpha$ (deg)', fontsize=11)
    ax.set_ylabel(r'$C_L$', fontsize=11)
    ax.set_title('Lift Coefficient')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- CD subplot ---
    ax = axes[1]
    # Plot experimental data as dashed line with scatter points
    ax.plot(exp['CD'].alpha_deg, exp['CD'].data, 
            'k--', linewidth=1, alpha=0.5)
    ax.scatter(exp['CD'].alpha_deg, exp['CD'].data,
               s=15, c='black', alpha=0.5, zorder=5, label='Experiment')
    for name, r in results.items():
        n_per_cycle = int(1.0 / freq_hz / dt)
        ax.plot(r['alpha_deg'][-n_per_cycle:], r['cd'][-n_per_cycle:], 
                label=name, color=colors[name], lw=1.5)
    ax.set_xlabel(r'$\alpha$ (deg)', fontsize=11)
    ax.set_ylabel(r'$C_D$', fontsize=11)
    ax.set_title('Drag Coefficient')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- CM subplot ---
    ax = axes[2]
    # Plot experimental data as dashed line with scatter points
    ax.plot(exp['CM'].alpha_deg, exp['CM'].data, 
            'k--', linewidth=1, alpha=0.5)
    ax.scatter(exp['CM'].alpha_deg, exp['CM'].data,
               s=15, c='black', alpha=0.5, zorder=5, label='Experiment')
    # Plot model CM results
    for name, r in results.items():
        n_per_cycle = int(1.0 / freq_hz / dt)
        if 'cm' in r:
            ax.plot(r['alpha_deg'][-n_per_cycle:], r['cm'][-n_per_cycle:], 
                    label=name, color=colors[name], lw=1.5)
    ax.set_xlabel(r'$\alpha$ (deg)', fontsize=11)
    ax.set_ylabel(r'$C_M$', fontsize=11)
    ax.set_title('Pitching Moment')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'S809 Dynamic Stall Validation — k = {k}, '
                 f'α = {alpha_mean_deg}° ± {alpha_amp_deg}°', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "validation_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
