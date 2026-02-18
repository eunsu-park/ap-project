# 02_Acquisition Module Documentation

Three-stage data acquisition and preprocessing pipeline that transforms raw database records into ML-ready HDF5 datasets.

## Pipeline Overview

```
Stage 1: DB → Original HDF5 (512×512)
Stage 2: Original → Merged HDF5 (64×64, time-series)
Stage 3: Merged → CSV (train/val split, classification labels)
```

## Directory Structure

```
02_Acquisition/
├── acquisition.py                    # DB query abstraction layer (egghouse)
├── 01_get_data_from_database.py      # Stage 1: DB → Original HDF5
├── 02_merge_and_resize.py            # Stage 2: Merge + resize time-series
├── 03_split_and_class.py             # Stage 3: Classification + train/val split
├── convert_merged_format.py          # Legacy format conversion utility
└── view_h5.py                        # HDF5 file inspector utility
```

## Stage Details

### Stage 1: `01_get_data_from_database.py`

Fetches solar observation data from PostgreSQL and preprocesses into individual HDF5 files.

**Input**: PostgreSQL (`space_weather.omni_low_resolution`, `solar_images.sdo`)

**Processing**:
- 3-hour cadence, 2010-09-01 to 2025-01-01
- Level 1.5 conversion via `egghouse.sdo.to_level15`
- AIA degradation correction via `aiapy`
- Logarithmic scaling (AIA) / Linear scaling (HMI)
- Solar disk masking: 1.2 R_sun (AIA), 0.99 R_sun (HMI)
- uint8 conversion

**Output**: `{ROOT}/original/YYYYMMDDHH.h5`
```
YYYYMMDDHH.h5
├── omni/          # Dict of OMNI parameters
└── sdo/
    ├── aia_193    # (512, 512) uint8
    ├── aia_211    # (512, 512) uint8
    └── hmi_magnetogram  # (512, 512) uint8
```

### Stage 2: `02_merge_and_resize.py`

Creates time-series datasets by merging multiple timesteps and resizing images.

**Input**: Original HDF5 files from Stage 1

**Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| BEFORE_DAYS | 7 | Lookback window (days) |
| AFTER_DAYS | 3 | Forecast window (days) |
| OMNI_CADENCE | 3 hours | OMNI sampling interval |
| SDO_CADENCE | 6 hours | SDO sampling interval |
| SDO_SIZE | 64 | Resize target (64×64) |

**Processing**:
- OMNI: T-7d to T+3d at 3h cadence → 80 timesteps
- SDO: T-7d to T at 6h cadence → 28 timesteps (input only, no future)
- Bilinear resize: 512×512 → 64×64

**Output**: `{ROOT}/original_64/YYYYMMDDHH.h5`
```
YYYYMMDDHH.h5
├── omni/
│   ├── B_Field_Magnitude_Avg_nT    # (80,) float
│   ├── Bz_GSM_nT                   # (80,) float
│   ├── ...                         # (80 individual parameter arrays)
│   └── attrs: t_target, t_start, t_end
└── sdo/
    ├── aia_193          # (28, 64, 64) uint8
    ├── aia_211          # (28, 64, 64) uint8
    └── hmi_magnetogram  # (28, 64, 64) uint8
```

### Stage 3: `03_split_and_class.py`

Extracts ap_index labels and splits dataset into train/validation sets.

**Input**: Merged HDF5 files from Stage 2

**Processing**:
- Validates 12 critical OMNI parameters (rejects samples with NaN)
- Extracts `ap_index_nt` for forecast period (T+0 to T+3 days)
- Calculates max ap_index per day (3 days)
- Binary classification: ap_index >= 48 → positive class
- Year-based split: year < 2024 → train, year >= 2024 → validation

**Output**: CSV files
```csv
file_name, max_day_1, max_day_2, max_day_3, class_day_1, class_day_2, class_day_3
2011010100.h5, 12.0, 7.0, 15.0, 0, 0, 0
```

## Core Module: `acquisition.py`

Database query abstraction layer using `egghouse.database.PostgresManager`.

| Function | Description |
|----------|-------------|
| `get_omni_data(start, end)` | Fetch OMNI time-series for date range |
| `get_sdo_best_match(target_time, channel)` | Find closest SDO image to target time |
| `get_sdo_file_path(target_time, channel)` | Get file path for exact datetime match |

**Database credentials**: Loaded from environment variables `DB_HOST`, `DB_USER`, `DB_PASSWORD` with local defaults.

## Common Commands

```bash
# Stage 1: Fetch data from DB (multiprocessing, ~4 workers)
python 01_get_data_from_database.py

# Stage 2: Merge and resize
python 02_merge_and_resize.py

# Stage 3: Classification and split
python 03_split_and_class.py

# Inspect HDF5 file structure
python view_h5.py /path/to/file.h5

# Convert legacy format (if needed)
python convert_merged_format.py
```

## Data Flow

```
01_Setup_DB (PostgreSQL)
    ↓ SQL queries via acquisition.py
Stage 1 → /datasets/original/YYYYMMDDHH.h5 (512×512)
    ↓
Stage 2 → /datasets/original_64/YYYYMMDDHH.h5 (64×64, time-series)
    ↓
Stage 3 → /datasets/original_64_train.csv + original_64_validation.csv
    ↓
03_Regression (model training)
```

## Dependencies

- `egghouse` (database, sdo.to_level15)
- `h5py` (HDF5 I/O)
- `aiapy` (AIA degradation correction)
- `sunpy` (Map processing)
- `PIL` (image resizing)
- `numpy`, `pandas`, `tqdm`
