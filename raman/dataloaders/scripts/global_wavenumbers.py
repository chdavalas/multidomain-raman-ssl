import pandas as pd
import numpy as np
import os
import scipy
import logging
import json
from pathlib import Path

# WARNING: In general when adding new data we need to make sure that the json file is updated with the new global wavenumber range. 
# This script can be used to recompute the global wavenumber range if needed.

# If the datasets is within the precomputed range, no further computations are needed, but if the new dataset has wavenumbers outside the precomputed range, 
# then we need to update the global wavenumber range and save it to the json file.

# This script computes the global wavenumber range across multiple Raman spectroscopy datasets and saves it to a JSON file.
# R6G, GBL and Raman Sugars are within the range, therefore they are not included in the computation. The script can be extended to include them if needed.

def global_wavenumber_range():
    # Initialize global min and max wavenumbers
    current_glob_min = 0
    current_glob_max = 1

    all_wavenumbers = np.array([])
    
    logger = logging.getLogger(__name__)
    # Set the logging level to INFO
    logger.setLevel(logging.INFO)


    # Adenine dataset
    df = pd.read_csv("~/armadillo/datasets/Adenine/ILSdata.csv", header=None)
    adenine_wavenumbers = df.iloc[0, 9:].to_numpy(dtype=np.float32)

    current_glob_min = np.min(adenine_wavenumbers)
    current_glob_max = np.max(adenine_wavenumbers)

    all_wavenumbers = np.concatenate((all_wavenumbers, adenine_wavenumbers))

    #bacteria dataset
    bacteria_wavenumbers = np.load(Path("~/armadillo/datasets/Bacteria/wavenumbers.npy").expanduser(), allow_pickle=True)
    current_glob_min = min(current_glob_min, np.min(bacteria_wavenumbers))
    current_glob_max = max(current_glob_max, np.max(bacteria_wavenumbers))

    all_wavenumbers = np.concatenate((all_wavenumbers, bacteria_wavenumbers))

    #liver dataset
    current_glob_min = min(current_glob_min, 500)
    current_glob_max = max(current_glob_max, 2000)

    all_wavenumbers = np.concatenate((all_wavenumbers, np.arange(500, 2001)))

    # SangXiancheng dataset  
    current_glob_min = min(current_glob_min, 30)
    current_glob_max = max(current_glob_max, 1599)

    all_wavenumbers = np.concatenate((all_wavenumbers, np.arange(30, 1600)))

    # Raman Sugars dataset
    # Acquisition range: 142 cm-1 - 3684.8 cm-1 from reading file in github dataset
    current_glob_min = min(current_glob_min, 142)
    current_glob_max = max(current_glob_max, 3684.8)
    all_wavenumbers = np.concatenate((all_wavenumbers, np.arange(142, 3685)))

    #Hyperspectral datasets
    axis_file = Path('~/armadillo/datasets/hyperspectral/axis.txt').expanduser()
    hyp_wavenumbers = np.loadtxt(axis_file, dtype=float, comments='#', unpack=True).astype(np.float32)

    all_wavenumbers = np.concatenate((all_wavenumbers, hyp_wavenumbers))

    #Wheat lines dataset
    mat = scipy.io.loadmat(Path("~/armadillo/datasets/wheat-lines/Data.mat").expanduser())
    wheat_lines_wavenumbers = np.unique(mat['Calx'][0]).astype(np.float32)

    current_glob_min = min(current_glob_min, float(np.min(wheat_lines_wavenumbers)))
    current_glob_max = max(current_glob_max, float(np.max(wheat_lines_wavenumbers)))

    all_wavenumbers = np.concatenate((all_wavenumbers, wheat_lines_wavenumbers))

    # GBL Wavenumbers
    for root, _, files in os.walk("~/armadillo/datasets/GBL"):
        for file in files:
            if file.lower().endswith('.csv'):
                # Skip files containing 'summary' in the filename (case-insensitive)
                if 'summary' in file.lower():
                    print(f"Skipping summary file: {file}")
                    continue
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                gbl_wavenumbers = df["Raman shift"].to_numpy(dtype=np.float64)

                current_glob_min = min(current_glob_min, np.min(gbl_wavenumbers))
                current_glob_max = max(current_glob_max, np.max(gbl_wavenumbers))

                all_wavenumbers = np.concatenate((all_wavenumbers, gbl_wavenumbers))

    # R6G
    class_dir = os.path.join("~/armadillo/datasets/R6G_dataset/", "R6G/Mixtures")
    for filename in os.listdir(Path(class_dir).expanduser()):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(class_dir, filename)).dropna()
        r6g_wavenumbers = df["Raman shift"].to_numpy(dtype=np.float64)

        current_glob_min = min(current_glob_min, np.min(r6g_wavenumbers))
        current_glob_max = max(current_glob_max, np.max(r6g_wavenumbers))

    all_wavenumbers = np.concatenate((all_wavenumbers, r6g_wavenumbers))

    #Deep raman wavenumbers
    # Range from paper 50-3200 cm-1, across all samples
    current_glob_min = min(current_glob_min, 50)
    current_glob_max = max(current_glob_max, 3200)
    all_wavenumbers = np.concatenate((all_wavenumbers, np.arange(50, 3201)))

    # Save the global wavenumber range to a file
    global_wavenumber_range = {
        "min": current_glob_min,
        "max": current_glob_max,
        "mean": np.unique(np.mean(all_wavenumbers))[0],
        "std": np.unique(np.std(all_wavenumbers))[0],
    }
    file_path = Path(__file__).parent.parent / "global_wavenumber_range.json"
    with open(file_path, "w") as f:
        json.dump(global_wavenumber_range, f, indent=4)

if __name__ == "__main__":
    global_wavenumber_range()