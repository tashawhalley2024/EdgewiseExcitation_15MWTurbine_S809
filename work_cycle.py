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
from fourier_analysis import load_dat_file

from fourier_analysis import normalise_data

#reuse format from fourier analysis

#from EdgewiseExcitation_15MWTurbine_S809.Dataset.Our_DynStall_Model.AeroParameters_at_93m.Polar_Instability_Resonance #here i want 10ms,20ms,30ms then Rigid_DeddoesIncomp, Rigid_IAGModel, Rigid_None, Rigid_Oye2

from scipy import integrate
import numpy as np
import csv
from pathlib import Path
import os
import pathlib


directory = (REPO_ROOT / "Dataset" / "EdgewiseExcited_IEA15MW_S809" / "AeroParameters_at_93m" / "Polar_Instability_Resonance")
data = {}

for n in directory.glob("*.dat"):
    with open(n,newline='') as datfile:
        data[n] = np.getfromtxt(datfile, delimiter=' ')

def work_per_cycle (cl,cd,cm):

    

    for cm_value,cl_value,cd_value in cm,cl,cd:
        #integrate across data
        W = cl_value*chord+cd_value*chord+cm_value

        return W


"""def cm(cl:np.ndarray,
       cd: np.ndarray,
       chord):
    #fix this, how do i calculate it?
    D=0
    r= 0
    
    C= (cl+cd)/(chord**2 * (chord**2/(D*r)))
    
    return -0.3

def w_cycle (cd: np.ndarray,
             cl: np.ndarray, 
             alpha: np.ndarray,
             wind,
             chord,
             cm
             ):
    
    rho = 1.225
    q = 0.5 * rho * wind**2
    S = 0.6
    c = chord
    
    M = q * S * c * cm   # Nm
    
    # Work over time history
    W = np.trapz(M, alpha)
    
    return W"""
    
def main():
    csv_rows = [["case", "wind", "struct", "model", "file", "fe_hz", "fmax_plot_hz"]]

    wrote_anything = False
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

                    t, alpha, cl, cd = load_dat_file(fpath)
                    

                    # NORMALIsE the data
                    alpha, cl, cd = normalise_data(alpha, cl, cd)
                    
                    wind_val = float(wind.replace("ms", ""))

                              work_data =  work_per_cycle(cl,cd)
                                        
                                        
                                         

if __name__ == "__main__":
    main()
    