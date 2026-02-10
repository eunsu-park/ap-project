"""Split and classify merged dataset for training.

This script splits the merged dataset into train/validation sets and
extracts target information (ap_index_nt based classification) for each file.
"""

import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
ROOT = "/Volumes/AP-PROJECT/datasets"
DATASET_NAME = "original_64"
DIR_LOAD = f"{ROOT}/{DATASET_NAME}"

# OMNI columns to use (files with nan/None in these columns will be excluded)
OMNI_KEYS = [
    "bx_gse_gsm_nt",
    "by_gsm_nt",
    "bz_gsm_nt",
    "b_magnitude_of_avg_field_vector_nt",
    "plasma_flow_speed_km_s",
    "proton_density_n_cm3",
    "proton_temperature_k",
    "kp_index",
    "ap_index_nt",
    "dst_index_nt",
    "f10_7_index_sfu",
    "sunspot_number_r"
]

# OMNI data structure: 80 points total (3-hour cadence)
# - Input: 56 points (7 days × 8 points/day) → index 0-55
# - Target: 24 points (3 days × 8 points/day) → index 56-79
INPUT_POINTS = 56
TARGET_POINTS = 24
POINTS_PER_DAY = 8

# Classification threshold
AP_THRESHOLD = 48


def has_valid_data(filepath: str) -> bool:
    """Checks if file has valid data (no nan/None in OMNI_KEYS columns).

    Args:
        filepath: Path to the HDF5 file.

    Returns:
        True if all OMNI_KEYS columns have no nan/None values.
    """
    with h5py.File(filepath, 'r') as f:
        for key in OMNI_KEYS:
            if key not in f['omni']:
                return False
            data = f['omni'][key][:]
            if np.any(np.isnan(data)) or np.any(data is None):
                return False
    return True


def extract_target_info(filepath: str) -> dict:
    """Extracts target information from a merged HDF5 file.

    Args:
        filepath: Path to the HDF5 file.

    Returns:
        Dictionary with file_name and target metrics.
    """
    filename = os.path.basename(filepath)

    with h5py.File(filepath, 'r') as f:
        ap_index = f['omni']['ap_index_nt'][:]

    # Target data: index 56-79 (24 points)
    target_data = ap_index[INPUT_POINTS:]

    # Calculate max values for each day
    max_day_1 = float(np.nanmax(target_data[:POINTS_PER_DAY]))      # 8 points
    max_day_2 = float(np.nanmax(target_data[:POINTS_PER_DAY * 2]))  # 16 points
    max_day_3 = float(np.nanmax(target_data[:POINTS_PER_DAY * 3]))  # 24 points

    # Classification (1 if >= 48, else 0)
    class_day_1 = 1 if max_day_1 >= AP_THRESHOLD else 0
    class_day_2 = 1 if max_day_2 >= AP_THRESHOLD else 0
    class_day_3 = 1 if max_day_3 >= AP_THRESHOLD else 0

    return {
        'file_name': filename,
        'max_day_1': max_day_1,
        'max_day_2': max_day_2,
        'max_day_3': max_day_3,
        'class_day_1': class_day_1,
        'class_day_2': class_day_2,
        'class_day_3': class_day_3,
    }


def is_validation(filename: str) -> bool:
    """Checks if file belongs to validation set (2024 or later).

    Args:
        filename: Filename in format YYYYMMDDHH.h5

    Returns:
        True if validation (2024+), False if train.
    """
    year = int(filename[:4])
    return year >= 2024


def main():
    """Main function to process all files and save CSV."""
    file_list = sorted(glob(f"{DIR_LOAD}/*.h5"))
    print(f"Found {len(file_list)} files")

    train_data = []
    val_data = []
    skipped_count = 0

    for filepath in tqdm(file_list, desc="Processing"):
        try:
            # Skip files with nan/None in OMNI_KEYS columns
            if not has_valid_data(filepath):
                skipped_count += 1
                continue

            info = extract_target_info(filepath)
            filename = info['file_name']

            if is_validation(filename):
                val_data.append(info)
            else:
                train_data.append(info)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"\nSkipped {skipped_count} files with invalid data (nan/None)")

    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Save to CSV
    train_path = os.path.join(ROOT, f'{DATASET_NAME}_train.csv')
    val_path = os.path.join(ROOT, f'{DATASET_NAME}_validation.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nTrain: {len(train_df)} files → {train_path}")
    print(f"Validation: {len(val_df)} files → {val_path}")

    # Print class distribution
    for name, df in [('Train', train_df), ('Validation', val_df)]:
        if len(df) > 0:
            print(f"\n{name} class distribution:")
            for day in [1, 2, 3]:
                col = f'class_day_{day}'
                pos = df[col].sum()
                total = len(df)
                print(f"  Day {day}: {pos}/{total} positive ({100*pos/total:.1f}%)")


if __name__ == "__main__":
    main()
