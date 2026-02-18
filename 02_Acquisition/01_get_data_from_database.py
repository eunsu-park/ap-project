"""SDO image preprocessing script.

This script fetches solar observation data from the database and performs
preprocessing including Level 1.5 conversion and AIA degradation correction.
"""

import datetime
import os
from datetime import timedelta
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.utils import get_correction_table
from sunpy.map import Map
from tqdm import tqdm

from acquisition import get_all_data_at_time
from egghouse.sdo import to_level15

# Load AIA degradation correction table from JSOC
AIA_CORRECTION_TABLE = get_correction_table(source='jsoc')


def process_sdo_channel(file_path: str, channel: str) -> Map:
    """Processes a single SDO channel image.

    Applies Level 1.5 conversion and degradation correction (for AIA channels).

    Args:
        file_path: Path to the FITS file.
        channel: Channel name ('aia_193', 'aia_211', 'hmi_magnetogram').

    Returns:
        Processed SunPy Map object.
    """
    # Level 1.5 conversion
    m_lev15 = to_level15(file_path)

    # Handle NaN values
    data = m_lev15.data.copy()
    data[np.isnan(data)] = 0.0
    m_lev15 = Map(data, m_lev15.meta)

    # Apply degradation correction for AIA channels
    if channel.startswith("aia_"):
        m_lev15 = correct_degradation(m_lev15, correction_table=AIA_CORRECTION_TABLE)

    return m_lev15


def to_image(m: Map, channel: str) -> np.ndarray:
    """Converts a SunPy Map to a uint8 image array.

    Applies appropriate scaling based on channel type:
    - AIA: logarithmic scaling (log2)
    - HMI: linear scaling with off-limb masking

    Args:
        m: SunPy Map object.
        channel: Channel name ('aia_193', 'aia_211', 'hmi_magnetogram').

    Returns:
        uint8 numpy array with values in range [0, 255].
    """
    data = m.data.copy()

    # Calculate r_sun in pixels
    r_sun_pix = m.meta.get('r_sun', m.meta.get('rsun_obs', 960)) / m.meta['cdelt1']
    ny, nx = data.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[:ny, :nx]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if channel.startswith("aia_"):
        # AIA: mask pixels outside 1.2 * r_sun
        mask = dist_from_center > r_sun_pix * 1.2
        data[mask] = 0.0

        # AIA: log2 scaling
        image = np.clip(data + 1.0, 1.0, None)
        image = np.log2(image)
        image = image * (255.0 / 14.0)
    else:
        # HMI: linear scaling
        # Mask pixels outside 0.99 * r_sun
        r_sun_pix = m.meta.get('r_sun', m.meta.get('rsun_obs', 960)) / m.meta['cdelt1']
        ny, nx = data.shape
        cy, cx = ny // 2, nx // 2
        y, x = np.ogrid[:ny, :nx]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist_from_center > r_sun_pix * 0.99
        data[mask] = -5000.0

        # Linear scaling: [-100, 100] -> [0, 255]
        image = (data + 100.0) * (255.0 / 200.0)

    # Clip to [0, 255] and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def save_to_hdf5(
    filepath: str,
    omni: dict,
    sdo: dict,
    timestamp: datetime.datetime
) -> None:
    """Saves observation data to HDF5 file.

    Args:
        filepath: Output file path (.h5).
        omni: Dictionary of OMNI solar wind parameters.
        sdo: Dictionary of SDO images {'aia_193': array, ...}.
        timestamp: Observation timestamp.
    """
    with h5py.File(filepath, 'w') as f:
        # File attributes
        f.attrs['timestamp'] = timestamp.isoformat()

        # OMNI group
        omni_grp = f.create_group('omni')
        for key, value in omni.items():
            # Skip None and NaN values
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue

            # Convert datetime/Timestamp to ISO string
            if hasattr(value, 'isoformat'):
                omni_grp.create_dataset(key, data=value.isoformat())
            # Skip unsupported object types
            elif hasattr(value, 'dtype') and value.dtype == object:
                continue
            else:
                omni_grp.create_dataset(key, data=value)

        # SDO group with gzip compression
        sdo_grp = f.create_group('sdo')
        for channel, image in sdo.items():
            sdo_grp.create_dataset(
                channel,
                data=image,
                compression='gzip',
                compression_opts=4
            )


# Configuration
ROOT = "/Volumes/AP-PROJECT/datasets"
SDO_CHANNELS = ["aia_193", "aia_211", "hmi_magnetogram"]
DIR_SAVE = f"{ROOT}/original"
TIME_RANGE_MINUTES = 12  # SDO data matching range (±minutes)
MAX_QUALITY = 9  # SDO quality filter (0 to MAX_QUALITY inclusive)


def process_single_timestamp(t: datetime.datetime) -> str:
    """Processes a single timestamp and saves to HDF5.

    Args:
        t: Target timestamp.

    Returns:
        Status message string.
    """
    filename = t.strftime('%Y%m%d%H.h5')
    filepath = os.path.join(DIR_SAVE, filename)

    # Skip if exists
    if os.path.exists(filepath):
        return f"Skip (exists): {filename}"

    # Fetch data
    result = get_all_data_at_time(t, time_range_minutes=TIME_RANGE_MINUTES, max_quality=MAX_QUALITY)

    # Check OMNI data
    if result['omni'].empty:
        return f"Skip (missing omni): {t}"

    # Check SDO channels
    missing_channels = [ch for ch in SDO_CHANNELS if result[ch].empty]
    if missing_channels:
        return f"Skip (missing sdo): {t} - {', '.join(missing_channels)}"

    # OMNI to dict
    omni = result['omni'].iloc[0].to_dict()

    # Process SDO images with error handling
    try:
        sdo = {}
        for channel in SDO_CHANNELS:
            file_path = result[channel]["file_path"].iloc[0]
            m_processed = process_sdo_channel(file_path, channel)
            image = to_image(m_processed, channel)
            sdo[channel] = image
    except Exception as e:
        return f"Skip (error): {filename} - {type(e).__name__}"

    # Save
    save_to_hdf5(filepath, omni, sdo, t)
    return f"Saved: {filename}"


def main(num_workers: int = None):
    """Main function with multiprocessing support.

    Args:
        num_workers: Number of worker processes. Defaults to min(cpu_count(), 8).
    """
    # Ensure output directory exists
    os.makedirs(DIR_SAVE, exist_ok=True)

    # Generate timestamp list
    t = datetime.datetime(2010, 9, 1, 0, 0, 0)
    t_end = datetime.datetime(2025, 1, 1, 0, 0, 0)
    timestamps = []
    while t < t_end:
        timestamps.append(t)
        t += timedelta(hours=3)

    print(f"Total timestamps: {len(timestamps)}")

    # Set number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 4)
    print(f"Using {num_workers} workers")

    # Multiprocessing
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_timestamp, timestamps),
            total=len(timestamps),
            desc="Processing"
        ))

    # Print summary
    saved = sum(1 for r in results if r.startswith("Saved"))
    skipped_exists = sum(1 for r in results if r.startswith("Skip (exists)"))
    skipped_missing_omni = sum(1 for r in results if r.startswith("Skip (missing omni)"))
    skipped_missing_sdo = sum(1 for r in results if r.startswith("Skip (missing sdo)"))
    skipped_error = sum(1 for r in results if r.startswith("Skip (error)"))
    total = len(results)
    print(f"\nCompleted ({total} total): {saved} saved, {skipped_exists} existed, "
          f"{skipped_missing_omni} missing omni, {skipped_missing_sdo} missing sdo, "
          f"{skipped_error} errors")


if __name__ == "__main__":
    main()


