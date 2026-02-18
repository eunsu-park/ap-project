# 01_Setup_DB Module Documentation

Solar and space weather data acquisition, validation, and database management system.

## Purpose

Downloads, validates, and registers solar observation data (SDO, LASCO, SECCHI) and space weather time-series data (OMNI) into PostgreSQL databases for downstream processing by `02_Acquisition` and `03_Regression`.

## Directory Structure

```
01_Setup_DB/
├── core/                           # Core library modules
│   ├── __init__.py                # Package exports
│   ├── database.py                # DB creation, table management, insert/upsert
│   ├── query.py                   # Data retrieval (best-match, date-range queries)
│   ├── sdo.py                     # SDO FITS validation, JSOC queries, TAI/UTC conversion
│   ├── lasco.py                   # LASCO VSO queries, metadata extraction
│   ├── secchi.py                  # SECCHI metadata extraction
│   ├── download.py                # HTTP download with retry & parallel support
│   ├── parse.py                   # OMNI Fortran-format parsing, FITS datetime parsing
│   ├── utils.py                   # YAML config loader with env var substitution
│   └── cli.py                     # Shared CLI argument definitions
├── scripts/                        # Executable scripts
│   ├── create_all_tables.py       # Initialize all DB tables
│   ├── download_omni.py           # Download OMNI time-series data
│   ├── download_sdo.py            # Download SDO data via JSOC
│   ├── download_lasco.py          # Download LASCO data via VSO
│   ├── download_secchi.py         # Download SECCHI data via SunPy
│   ├── download_from_urls.py      # Download from pre-generated URL list
│   ├── register_sdo.py            # Scan & register SDO FITS files to DB
│   ├── register_lasco.py          # Scan & register LASCO FITS files to DB
│   ├── register_secchi.py         # Scan & register SECCHI FITS files to DB
│   └── query_sdo.py               # Query JSOC and save URLs to JSON
└── configs/
    ├── solar_images_config.yaml   # SDO, LASCO, SECCHI DB schema & paths
    └── space_weather_config.yaml  # OMNI DB schema & paths
```

Detailed schema documentation: [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)

## Databases

Two PostgreSQL databases managed via `egghouse.database.PostgresManager`:

| Database | Tables | Content |
|----------|--------|---------|
| `solar_images` | `sdo`, `lasco`, `secchi` | FITS file metadata (paths, timestamps, quality) |
| `space_weather` | `omni_low_resolution`, `omni_high_resolution`, `omni_high_resolution_5min` | Solar wind parameters, geomagnetic indices |

### Connection

Configured in YAML files with environment variable substitution:

```yaml
db_config:
  host: localhost
  port: 5432
  database: solar_images
  user: ${DB_USER}
  password: ${DB_PASSWORD}
```

## Data Sources

### SDO (Solar Dynamics Observatory)

- **Source**: JSOC (Joint Science Operations Center) via `drms` library
- **Instruments**: AIA (EUV imaging), HMI (magnetograms)
- **Channels**: AIA 94/131/171/193/211/304/335 Å, HMI m_45s/m_720s
- **Schema PK**: `(telescope, channel, datetime)`

### LASCO (SOHO Coronagraph)

- **Source**: VSO (Virtual Solar Observatory) via SunPy Fido
- **Cameras**: C2 (2-6 R☉), C3 (3.7-30 R☉)
- **Schema PK**: `(camera, datetime)`

### SECCHI (STEREO)

- **Source**: SunPy
- **Instruments**: COR1, COR2, EUVI, HI_1, HI_2
- **Schema PK**: `(datatype, spacecraft, instrument, channel, datetime)`

### OMNI (Space Weather Time-Series)

- **Source**: NOAA/GSFC CDAWeb (Fortran-format text files)
- **Resolutions**: Hourly (55 cols), 1-min (46 cols), 5-min (49 cols)
- **Schema PK**: `datetime`
- **Key columns**: IMF components, plasma parameters, geomagnetic indices (Kp, Dst, AE, Ap)

## Core API Summary

### Database Operations (`core/database.py`)

| Function | Description |
|----------|-------------|
| `initialize_database(db_config, schema_config)` | Create DB + tables |
| `create_database(db_config)` | Create database if not exists |
| `create_tables(db_config, schema_config)` | Create tables from YAML schema |
| `insert(db_config, table, df)` | Batch insert DataFrame |
| `upsert(db_config, table, df)` | Insert with ON CONFLICT DO NOTHING |
| `delete_orphans(db_config, table)` | Remove records for missing files |

### Query Operations (`core/query.py`)

| Function | Description |
|----------|-------------|
| `get_sdo_best_match(db_config, telescope, channel, target_time)` | Find closest SDO image |
| `get_sdo_best_matches(db_config, telescope, channel, target_times)` | Batch best-match |
| `get_lasco_data(db_config, camera, start, end)` | LASCO records by date range |
| `get_secchi_data(db_config, ...)` | SECCHI records with filters |
| `get_hourly_target_times(start, end)` | Generate hourly timestamps |

### SDO Utilities (`core/sdo.py`)

| Function | Description |
|----------|-------------|
| `validate_fits(file_path)` | Validate FITS & extract metadata (returns dict or error string) |
| `query_jsoc_v2(...)` | Query JSOC for download URLs |
| `tai_to_utc(tai_time)` / `utc_to_tai(utc_time)` | TAI ↔ UTC conversion |
| `get_target_path(root, telescope, channel, datetime)` | Generate YYYY/YYYYMMDD path |
| `check_db_exists_in_range(...)` | Check if record exists within time window |

### Download (`core/download.py`)

| Function | Description |
|----------|-------------|
| `download(url)` | Download text with retry |
| `download_file(url, save_path)` | Download binary file with size verification |
| `download_files_parallel(tasks, max_workers)` | Parallel file downloads |
| `list_remote_files(url, extension)` | Parse directory listing for file links |

### Parsing (`core/parse.py`)

| Function/Constant | Description |
|----------|-------------|
| `parse(text, spec)` | Parse Fortran-format text to DataFrame |
| `parse_fits_datetime(file_path)` | Extract datetime from FITS header/filename |
| `LOWRES` / `HIGHRES` / `HIGHRES_5MIN` | OMNI data format specifications |

## Common Commands

```bash
# Initialize all tables
python scripts/create_all_tables.py

# Download SDO data for a specific time
python scripts/download_sdo.py --target-time "2024-01-01 00:00:00" \
    --instrument aia_193 aia_211 hmi_magnetogram

# Download SDO data for a date range
python scripts/download_sdo.py --start-date 2024-01-01 --end-date 2024-01-31 \
    --instrument aia_193

# Download OMNI data
python scripts/download_omni.py --start-date 2024-01-01 --end-date 2024-12-31

# Register existing FITS files to DB
python scripts/register_sdo.py --parallel 8

# Download from pre-generated URL list
python scripts/download_from_urls.py urls.json --parallel 4
```

## File Storage Layout

```
/opt/archive/
├── solar_images/
│   ├── sdo/
│   │   ├── aia/YYYY/YYYYMMDD/*.fits
│   │   ├── hmi/YYYY/YYYYMMDD/*.fits
│   │   ├── downloaded/          # Temp download staging
│   │   ├── invalid_file/        # Corrupted FITS
│   │   ├── invalid_header/      # Missing headers
│   │   └── invalid_data/        # Data validation failed
│   ├── lasco/cX/YYYY/YYYYMM/*.fts
│   └── secchi/datatype/spacecraft/instrument/YYYY/YYYYMMDD/
└── space_weather/omni/
```

## Dependencies

- `egghouse` (database, I/O)
- `astropy` (FITS handling)
- `drms` (JSOC queries)
- `sunpy` (VSO/Fido)
- `pandas`, `numpy`, `requests`
