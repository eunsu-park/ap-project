"""Convert merged HDF5 files from old format to new table format.

Old format:
    omni/
    ├── data      # (80, num_features) 2D array
    └── columns   # column names

New format:
    omni/
    ├── bx_gse_gsm_nt           # (80,) 1D array
    ├── by_gsm_nt               # (80,) 1D array
    └── ...                     # each parameter as a dataset
"""

import os
import sys
from glob import glob
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from tqdm import tqdm

ROOT = "/Volumes/AP-PROJECT/datasets"
DIR_MERGED = f"{ROOT}/02_merged"


def is_old_format(filepath: str) -> bool:
    """Checks if file uses old format (omni/data + omni/columns).

    Args:
        filepath: Path to the HDF5 file.

    Returns:
        True if old format, False if new format.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if 'omni' not in f:
                return False
            return 'data' in f['omni'] and 'columns' in f['omni']
    except Exception:
        return False


def convert_single_file(filepath: str) -> str:
    """Converts a single file from old format to new format.

    Args:
        filepath: Path to the HDF5 file.

    Returns:
        Status message string.
    """
    filename = os.path.basename(filepath)

    if not is_old_format(filepath):
        return f"Skip (already new format): {filename}"

    try:
        # Read old format data
        with h5py.File(filepath, 'r') as f:
            # Read attributes
            attrs = dict(f.attrs)

            # Read OMNI data (old format)
            omni_data = f['omni/data'][:]
            omni_columns = [
                c.decode() if isinstance(c, bytes) else c
                for c in f['omni/columns'][:]
            ]

            # Read SDO data
            sdo_data = {}
            for channel in f['sdo'].keys():
                sdo_data[channel] = f['sdo'][channel][:]

        # Write new format
        with h5py.File(filepath, 'w') as f:
            # Write attributes
            for key, value in attrs.items():
                f.attrs[key] = value

            # Write OMNI data (new table format)
            omni_grp = f.create_group('omni')
            for i, col_name in enumerate(omni_columns):
                omni_grp.create_dataset(
                    col_name,
                    data=omni_data[:, i],
                    compression='gzip'
                )

            # Write SDO data
            sdo_grp = f.create_group('sdo')
            for channel, arr in sdo_data.items():
                sdo_grp.create_dataset(channel, data=arr, compression='gzip')

        return f"Converted: {filename}"

    except Exception as e:
        return f"Error ({filename}): {type(e).__name__}: {e}"


def main(num_workers: int = None):
    """Main function with multiprocessing support.

    Args:
        num_workers: Number of worker processes.
    """
    files = sorted(glob(os.path.join(DIR_MERGED, "*.h5")))
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files")

    # Count old format files
    old_format_count = sum(1 for f in files if is_old_format(f))
    print(f"Files to convert: {old_format_count}")

    if old_format_count == 0:
        print("All files already in new format.")
        return

    # Set number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    print(f"Using {num_workers} workers")

    # Multiprocessing
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_single_file, files),
            total=len(files),
            desc="Converting"
        ))

    # Print summary
    converted = sum(1 for r in results if r.startswith("Converted"))
    skipped = sum(1 for r in results if r.startswith("Skip"))
    errors = sum(1 for r in results if r.startswith("Error"))
    print(f"\nCompleted: {converted} converted, {skipped} skipped, {errors} errors")

    # Print errors if any
    if errors > 0:
        print("\nErrors:")
        for r in results:
            if r.startswith("Error"):
                print(f"  {r}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Check mode: only count old format files
        files = sorted(glob(os.path.join(DIR_MERGED, "*.h5")))
        old_count = sum(1 for f in tqdm(files, desc="Checking") if is_old_format(f))
        print(f"\nOld format: {old_count}/{len(files)} files")
    else:
        main()
