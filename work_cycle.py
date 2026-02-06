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
MODELS = ["None", "Oye2", "IAGModel", "BeddoesIncomp"]

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

def normalise_data(alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray, cm: np.ndarray):
    """
    Normalise alpha, cl, and cd by their respective normalisation constants.
    
    Args:
        alpha: Angle of attack in radians
        cl: Lift coefficient
        cd: Drag coefficient
    
    Returns:
        Normalised alpha, cl, cd
    """
    alpha_normalised = alpha / ALPHA_NORM
    cl_normalised = cl / CL_NORM
    cd_normalised = cd / CD_NORM
    cm_normalised = cm / CM_NORM
    
    return alpha_normalised, cl_normalised, cd_normalised, cm_normalised

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

def work_per_cycle (cl,cd,cm):
    
    chord = 600
    for cm_value,cl_value,cd_value in cm,cl,cd:
        #integrate across data
        W = cl_value*chord+cd_value*chord+cm_value

        return W

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
                    

                    # NORMALIsE the data
                    alpha, cl, cd, cm = normalise_data(alpha, cl, cd,cm)
    

                    work_data =  work_per_cycle(cl,cd,cm)

                    plt.plot(t, work_data)                                      

if __name__ == "__main__":
    main()
    