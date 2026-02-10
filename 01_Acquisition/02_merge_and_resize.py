"""Merge and resize script for solar observation data.

This script merges OMNI and SDO data from multiple timestamps into a single
time-series dataset file. SDO images are resized to 64x64 for efficient storage.
"""

import os
import datetime
from datetime import timedelta
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
BEFORE_DAYS = 7
AFTER_DAYS = 3
OMNI_CADENCE = 3  # hours
SDO_CADENCE = 6   # hours
SDO_SIZE = 64     # resize target

SDO_CHANNELS = ["aia_193", "aia_211", "hmi_magnetogram"]
ROOT = "/Volumes/AP-PROJECT/datasets"
DIR_LOAD = f"{ROOT}/original"
DATASET_NAME = "original_64"
DIR_SAVE = f"{ROOT}/{DATASET_NAME}"


def load_omni_from_h5(filepath: str) -> dict:
    """Loads OMNI data from HDF5 file.

    Args:
        filepath: Path to the HDF5 file.

    Returns:
        Dictionary of OMNI parameters.
    """
    with h5py.File(filepath, 'r') as f:
        omni = {}
        for key in f['omni'].keys():
            omni[key] = f['omni'][key][()]
        return omni


def load_sdo_from_h5(filepath: str, channel: str, size: int = 64) -> np.ndarray:
    """Loads and resizes SDO image from HDF5 file.

    Args:
        filepath: Path to the HDF5 file.
        channel: SDO channel name ('aia_193', 'aia_211', 'hmi_magnetogram').
        size: Target size for resizing.

    Returns:
        Resized image as numpy array.
    """
    with h5py.File(filepath, 'r') as f:
        image = f['sdo'][channel][:]
        # Resize using PIL
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((size, size), Image.BILINEAR)
        return np.array(pil_image)


def process_single_target(t_target: datetime.datetime) -> str:
    """Processes a single target time and saves merged data.

    Args:
        t_target: Target timestamp for prediction.

    Returns:
        Status message string.
    """
    filename = t_target.strftime('%Y%m%d%H.h5')
    filepath_save = os.path.join(DIR_SAVE, filename)

    # Skip if exists
    if os.path.exists(filepath_save):
        return f"Skip (exists): {filename}"

    t_start = t_target - timedelta(days=BEFORE_DAYS)
    t_end = t_target + timedelta(days=AFTER_DAYS)

    # Collect OMNI data (3-hour cadence, BEFORE + AFTER)
    omni_list = []
    t = t_start
    while t < t_end:
        file_name = t.strftime("%Y%m%d%H.h5")
        file_path = os.path.join(DIR_LOAD, file_name)
        if not os.path.exists(file_path):
            return f"Skip (missing): {filename}"
        omni_list.append(load_omni_from_h5(file_path))
        t += timedelta(hours=OMNI_CADENCE)

    # Collect SDO data (6-hour cadence, BEFORE only - up to t_target)
    sdo_data = {ch: [] for ch in SDO_CHANNELS}
    t = t_start
    while t < t_target:  # SDO는 t_target 전까지만 (BEFORE_DAY만)
        file_name = t.strftime("%Y%m%d%H.h5")
        file_path = os.path.join(DIR_LOAD, file_name)
        if not os.path.exists(file_path):
            return f"Skip (missing): {filename}"
        for channel in SDO_CHANNELS:
            image = load_sdo_from_h5(file_path, channel, SDO_SIZE)
            sdo_data[channel].append(image)
        t += timedelta(hours=SDO_CADENCE)

    # Convert to arrays
    # OMNI: dict of lists -> dict of 1D arrays (table format)
    omni_keys = sorted(omni_list[0].keys())
    omni_keys = [k for k in omni_keys if k != 'datetime']  # exclude datetime
    omni_arrays = {
        key: np.array([row.get(key, np.nan) for row in omni_list])
        for key in omni_keys
    }

    # SDO: list of images -> 3D array
    sdo_arrays = {ch: np.stack(images) for ch, images in sdo_data.items()}

    # Save to HDF5
    with h5py.File(filepath_save, 'w') as f:
        f.attrs['t_target'] = t_target.isoformat()
        f.attrs['t_start'] = t_start.isoformat()
        f.attrs['t_end'] = t_end.isoformat()

        # OMNI group (table format: each parameter as a column)
        omni_grp = f.create_group('omni')
        for key, arr in omni_arrays.items():
            omni_grp.create_dataset(key, data=arr, compression='gzip')

        # SDO group
        sdo_grp = f.create_group('sdo')
        for channel, arr in sdo_arrays.items():
            sdo_grp.create_dataset(channel, data=arr, compression='gzip')

    return f"Saved: {filename}"


def verify_merged_file(filepath: str) -> dict:
    """Verifies the structure and content of a merged HDF5 file.

    Args:
        filepath: Path to the HDF5 file to verify.

    Returns:
        Dictionary with verification results:
        {
            'valid': bool,
            'errors': list of error messages,
            'info': dict with file info
        }
    """
    errors = []
    info = {}

    # Expected shapes
    expected_omni_rows = int((BEFORE_DAYS + AFTER_DAYS) * 24 / OMNI_CADENCE)  # 80
    expected_sdo_rows = int(BEFORE_DAYS * 24 / SDO_CADENCE)  # 28

    try:
        with h5py.File(filepath, 'r') as f:
            # Check attributes
            for attr in ['t_target', 't_start', 't_end']:
                if attr not in f.attrs:
                    errors.append(f"Missing attribute: {attr}")
                else:
                    info[attr] = f.attrs[attr]

            # Check OMNI group (table format: each parameter as a column)
            if 'omni' not in f:
                errors.append("Missing group: omni")
            else:
                omni_keys = list(f['omni'].keys())
                info['omni_columns'] = omni_keys
                if len(omni_keys) == 0:
                    errors.append("OMNI group has no datasets")
                else:
                    # Check first column for row count
                    first_key = omni_keys[0]
                    omni_rows = f['omni'][first_key].shape[0]
                    info['omni_rows'] = omni_rows
                    if omni_rows != expected_omni_rows:
                        errors.append(
                            f"OMNI row count mismatch: expected {expected_omni_rows}, "
                            f"got {omni_rows}"
                        )

            # Check SDO group
            if 'sdo' not in f:
                errors.append("Missing group: sdo")
            else:
                for channel in SDO_CHANNELS:
                    if channel not in f['sdo']:
                        errors.append(f"Missing dataset: sdo/{channel}")
                    else:
                        shape = f['sdo'][channel].shape
                        info[f'{channel}_shape'] = shape
                        if shape != (expected_sdo_rows, SDO_SIZE, SDO_SIZE):
                            errors.append(
                                f"SDO {channel} shape mismatch: expected "
                                f"({expected_sdo_rows}, {SDO_SIZE}, {SDO_SIZE}), got {shape}"
                            )

    except Exception as e:
        errors.append(f"Failed to read file: {type(e).__name__}: {e}")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'info': info
    }


def verify_all_files(sample_count: int = 10) -> None:
    """Verifies a sample of merged files.

    Args:
        sample_count: Number of files to verify (0 for all files).
    """
    import glob
    import random

    files = sorted(glob.glob(os.path.join(DIR_SAVE, "*.h5")))
    if not files:
        print("No files found to verify.")
        return

    if sample_count > 0 and len(files) > sample_count:
        files = random.sample(files, sample_count)
        print(f"Verifying {sample_count} random files out of {len(files)} total...")
    else:
        print(f"Verifying all {len(files)} files...")

    valid_count = 0
    for filepath in tqdm(files, desc="Verifying"):
        result = verify_merged_file(filepath)
        if result['valid']:
            valid_count += 1
        else:
            print(f"\n{os.path.basename(filepath)}: INVALID")
            for error in result['errors']:
                print(f"  - {error}")

    print(f"\nVerification complete: {valid_count}/{len(files)} files valid")


def main(num_workers: int = None):
    """Main function with multiprocessing support.

    Args:
        num_workers: Number of worker processes. Defaults to min(cpu_count(), 8).
    """
    os.makedirs(DIR_SAVE, exist_ok=True)

    # Generate timestamp list
    t_target = datetime.datetime(2010, 9, 1, 0, 0, 0)
    t_end = datetime.datetime(2025, 1, 1, 0, 0, 0)
    timestamps = []
    while t_target < t_end:
        timestamps.append(t_target)
        t_target += timedelta(hours=OMNI_CADENCE)

    print(f"Total timestamps: {len(timestamps)}")

    # Set number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 4)
    print(f"Using {num_workers} workers")

    # Multiprocessing
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_target, timestamps),
            total=len(timestamps),
            desc="Processing"
        ))

    # Print summary
    saved = sum(1 for r in results if r.startswith("Saved"))
    skipped_exists = sum(1 for r in results if "exists" in r)
    skipped_missing = sum(1 for r in results if "missing" in r)
    print(f"\nCompleted: {saved} saved, {skipped_exists} existed, {skipped_missing} missing")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # Verify mode: python 02_merge_and_resize.py --verify [sample_count]
        sample_count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        verify_all_files(sample_count)
    else:
        # Normal mode: generate merged dataset
        main()
