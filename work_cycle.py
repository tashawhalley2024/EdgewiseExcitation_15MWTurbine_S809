# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 10:59:03 2026

@author: cvhoh
"""
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, find_peaks

from scipy.integrate import simpson

# -----------------------------
# PATHS
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parent

DATASET_DIR = (
    REPO_ROOT
    / "Dataset"
    / "EdgewiseExcited_IEA15MW_S809"
    / "AeroParameters_at_93m"
)

OUT_DIR = REPO_ROOT / "moving_fft_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ_CSV = OUT_DIR / "excitation_frequencies.csv"


# -----------------------------
# CASES / OPTIONS
# -----------------------------
CASES = [
    "Polar_Instability_OSU_Frequency",
    "Polar_Instability_Resonance",
]
WINDS = ["10ms", "30ms", "50ms"]
STRUCTS = ["Flex", "Rigid"]
MODELS = ["Static", "Oye2", "IAGModel", "BeddoesIncomp"]

# FFT pick range (Hz) for f_e detection from alpha
FMAX_FFT_PICK = 5.0

# Plot cap fallback if auto cap fails
FMAX_PLOT_FALLBACK = 10.0

#stft settings defined in seconds 
# Needs optimization for values
WINDOW_SEC = 15 #seconds per stft window
OVERLAP_SEC = 14 #seconds overlap
# WINDOW_SEC = 5 #seconds per stft window
# OVERLAP_SEC = 4.5 #seconds overlap


#log power epsilon floor
EPS = 1e-12

#fixed colour scale for comparisons (edit so use -70 and -10 for no epsilon)
COLOUR_LIM_MIN = -70
COLOUR_LIM_MAX = -10


# -----------------------------
# NORMALISATION CONSTANTS
# -----------------------------
ALPHA_AMPLITUDE_DEG = 10.45  # degrees
ALPHA_AMPLITUDE_RAD = ALPHA_AMPLITUDE_DEG * np.pi / 180
ALPHA_NORM = ALPHA_AMPLITUDE_RAD
CL_NORM = 2 * np.pi * ALPHA_AMPLITUDE_RAD
CD_NORM = CL_NORM * np.sin(ALPHA_AMPLITUDE_RAD)
CM_NORM = CL_NORM * (1/4)

import sys
sys.path.append('/DNV')

def load_dat_file(filepath: Path):
    """
    File format:
    # time[s], AoA[rad], cl[-], cd[-], cm[-]
    """
    data = np.loadtxt(filepath, comments="#")
    t = data[:, 0]
    alpha = data[:, 1]
    cl = data[:, 2]
    cd = data[:, 3]
    cm = data[:, 4]
    return t, alpha, cl, cd, cm

#reuse format from fourier analysis

#from EdgewiseExcitation_15MWTurbine_S809.Dataset.Our_DynStall_Model.AeroParameters_at_93m.Polar_Instability_Resonance #here i want 10ms,20ms,30ms then Rigid_DeddoesIncomp, Rigid_IAGModel, Rigid_None, Rigid_Oye2

from scipy import integrate
import numpy as np
import csv
from pathlib import Path


directory = (REPO_ROOT / "Dataset" / "EdgewiseExcited_IEA15MW_S809" / "AeroParameters_at_93m" / "Polar_Instability_Resonance")
data = {}

for n in directory.glob("*.dat"):
    with open(n,newline='') as datfile:
        data[n] = np.getfromtxt(datfile, delimiter=' ')

def work_per_cycle (cl,cd,cm,t,alpha,pitch_centre_x,pitch_centre_y,int_range,model):
    #convert to radians
    alpha_rad = np.deg2rad(alpha)
    #calculate x and y force componenets from lift and drag
    Cy = cl*np.cos(alpha_rad) + cd*np.sin(alpha_rad)
    Cx = cl*np.sin(alpha_rad) + cd *np.cos(alpha_rad)

    

    #calculate total moment by pitch moment+force x distance
    M = cm +Cy*pitch_centre_x+Cx*pitch_centre_y

    print('alpha_rad:',alpha_rad)
    print('moment force:',M)
    #plt.plot(alpha,M,label=model)
    #plt.legend()
    #plt.show()

    #find integral over each cycle
    N_Integrate = int((t[-1]-t[0])/int_range)
    n_pts_each_cycle = int(len(t)/N_Integrate)
    print('n_pts_each_cycle:', n_pts_each_cycle)

    t_binned = ()
    work_binned = ()
    i_low =0
    i_high = n_pts_each_cycle
    for i in range(0,N_Integrate):
        t_mean = 0.5*(t[i_low]+t[i_high-1])
        work_i = integrate.simpson(M[i_low:i_high], alpha_rad[i_low:i_high])

        print('work i :', work_i)

        i_low = i_high
        i_high = i_low +n_pts_each_cycle

        t_binned = np.append(t_binned,t_mean)
        work_binned = np.append(work_binned, work_i)

    return t_binned,work_binned

    
    

def main():

    missing_count = 0

    for case in CASES:
        for wind in WINDS:
            for struct in STRUCTS:
                for model in MODELS:
                    fname = f"{struct}_{model}.dat"
                    fpath = DATASET_DIR / case / wind / fname

                    if not fpath.exists():
                        missing_count += 1
                        continue

                    print("Processing:", fpath)

                    t, alpha, cl, cd, cm = load_dat_file(fpath)


                    excitation = 0.687 
                    int_range = 1/excitation
                    chord = 2.771723
                    pitch_center_x = 0/chord
                    pitch_center_y = 0/chord

                    print('int rage:', int_range)

                    t_binned,work_binned = work_per_cycle(cl,cd,cm,t,alpha, pitch_center_x,pitch_center_y,int_range,model)                                   
                    print('model:', model)
                    plt.plot(t_binned,work_binned,label=model)
                    
                plt.legend()
                plt.title(f"{case} – Work per cycle vs time {wind}")
                plt.ylabel('Work per Cycle (W) [non dimensionalised]')
                plt.xlabel('time (t) [s]')
                plt.rcParams.update({
    # Base text
    "font.size": 18,

    # Axes
    "axes.titlesize": 24,
    "axes.labelsize": 22,

    # Ticks
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,

    # Legends / annotations
    "legend.fontsize": 20,

    # Figure-level titles (if used)
    "figure.titlesize": 26,

    # Line widths (helps freq lines stand out when downscaled)
    "lines.linewidth": 1.5,
})
                plt.show()

if __name__ == "__main__":
    main()
    