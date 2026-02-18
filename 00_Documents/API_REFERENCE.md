# API Reference

Complete list of all public functions, classes, and constants in egghouse.

**Detailed Usage Guides:** See [docs/](docs/) folder for module-specific guides.

- [io_guide.md](docs/io_guide.md) - FITS, BMP file I/O
- [image_guide.md](docs/image_guide.md) - Image processing utilities
- [sdo_guide.md](docs/sdo_guide.md) - SDO/AIA/HMI data processing
- [config_guide.md](docs/config_guide.md) - Configuration management
- [database_guide.md](docs/database_guide.md) - PostgreSQL utilities
- [transfer_guide.md](docs/transfer_guide.md) - File transfer utilities (HTTP, FTP, SFTP)

---

## egghouse.io

File I/O utilities for scientific data formats.

### FITS (requires astropy)

| Function | Signature | Description |
|----------|-----------|-------------|
| `read_fits` | `(filepath, hdu_index=0) -> Tuple[ndarray, dict]` | Read FITS data and header |
| `write_fits` | `(filepath, data, header=None, overwrite=False) -> None` | Write numpy array to FITS |
| `read_fits_header` | `(filepath, hdu_index=0) -> dict` | Read header only (no data loading) |
| `append_fits` | `(filepath, data, header=None) -> None` | Append HDU extension to existing FITS |

### FITS (pure numpy, no dependencies)

| Function | Signature | Description |
|----------|-----------|-------------|
| `read_fits_simple` | `(filepath, hdu_index=0, apply_scaling=True) -> Tuple[ndarray, dict]` | Read FITS image HDU (Primary or Extension) |
| `read_fits_header_simple` | `(filepath, hdu_index=0) -> dict` | Read header only (no data loading) |

### BMP (no external dependencies)

| Function | Signature | Description |
|----------|-----------|-------------|
| `read_bmp` | `(filepath) -> Tuple[ndarray, dict]` | Read BMP as (H,W,3) uint8 RGB + info |
| `write_bmp` | `(filepath, data, overwrite=False) -> None` | Write (H,W) or (H,W,3) uint8 to BMP |
| `read_bmp_header` | `(filepath) -> dict` | Read BMP header only |

### Constants

| Name | Description |
|------|-------------|
| `HAS_ASTROPY` | `True` if astropy is available |

---

## egghouse.image

Generic image processing utilities. Organized into submodules:
- `core` - Basic transformations
- `masking` - Circle and annulus masks
- `spatial` - Padding, cropping, flipping
- `filters` - Gaussian, median, edge detection
- `stats` - Normalization, histogram, statistics

### Core

| Function | Signature | Description |
|----------|-----------|-------------|
| `resize_image` | `(image, size, order=1, preserve_range=True) -> ndarray` | Resize image to (height, width), preserves dtype |
| `rotate_image` | `(image, angle, order=1, reshape=False, cval=0, preserve_range=True) -> ndarray` | Rotate image by angle (degrees) |
| `bytescale_image` | `(data, imin=None, imax=None, omin=0, omax=255) -> ndarray` | Scale to uint8 [omin, omax] |

### Masking

| Function | Signature | Description |
|----------|-----------|-------------|
| `circle_mask` | `(image_size, radius, center=None, mask_type='inner') -> ndarray` | Circular boolean mask |
| `annulus_mask` | `(image_size, inner_radius, outer_radius, center=None) -> ndarray` | Ring-shaped boolean mask |

### Spatial

| Function | Signature | Description |
|----------|-----------|-------------|
| `pad_image` | `(data, target_size, pad_value=0, center=True) -> ndarray` | Pad image to target size |
| `crop_or_pad` | `(data, target_size, pad_value=0, center=True) -> ndarray` | Crop or pad to exact target size |
| `flip_image` | `(image, axis='vertical') -> ndarray` | Flip image ('vertical', 'horizontal', 'both') |
| `roll_image` | `(image, shift_y, shift_x) -> ndarray` | Cyclic roll image by (shift_y, shift_x) |

### Filters

| Function | Signature | Description |
|----------|-----------|-------------|
| `gaussian_smooth` | `(image, sigma=1.0, preserve_range=True) -> ndarray` | Gaussian smoothing filter |
| `median_denoise` | `(image, size=3, preserve_range=True) -> ndarray` | Median filter for noise removal |
| `laplacian_edge` | `(image, mode='reflect') -> ndarray` | Laplacian edge detection (2nd derivative) |
| `sobel_edge` | `(image, axis=None, mode='reflect') -> ndarray` | Sobel edge detection (gradient) |
| `unsharp_mask` | `(image, sigma=1.0, amount=1.0, preserve_range=True) -> ndarray` | Sharpen image via unsharp masking |

### Stats

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize_image` | `(image, mean=None, std=None) -> ndarray` | Z-score normalization (mean=0, std=1) |
| `get_image_stats` | `(image, mask=None, percentiles=...) -> dict` | Compute image statistics (mean, std, min, max, percentiles) |
| `histogram_equalization` | `(image, nbins=256) -> ndarray` | Enhance contrast via histogram equalization |
| `percentile_scale` | `(image, low=1.0, high=99.0, omin=0, omax=255) -> ndarray` | Scale using percentile clipping |
| `find_disk_center` | `(image, threshold=None, method='centroid') -> Tuple[float, float]` | Find bright disk center (cy, cx) |
| `adaptive_threshold` | `(image, block_size=35, offset=0.0) -> ndarray` | Adaptive binarization for uneven illumination |

### Aliases

| Alias | Function |
|-------|----------|
| `resize` | `resize_image` |
| `rotate` | `rotate_image` |
| `bytescale` | `bytescale_image` |
| `pad` | `pad_image` |

---

## egghouse.sdo

SDO/AIA and SDO/HMI data processing utilities.

### AIA

| Function | Signature | Description |
|----------|-----------|-------------|
| `aia_intscale` | `(data, wavelength, ...) -> ndarray` | Wavelength-specific intensity scaling (94-6173 Å) |
| `get_aia_calibration` | `(wavelength) -> dict` | Get calibration parameters for wavelength |

### HMI

| Function | Signature | Description |
|----------|-----------|-------------|
| `hmi_intscale` | `(data, vmin=-1000, vmax=1000, ...) -> ndarray` | Magnetogram scaling to uint8 |
| `hmi_field_strength` | `(bx, by, bz) -> ndarray` | Calculate vector field strength |

### Core

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse_fits_header` | `(filepath) -> dict` | Extract SDO FITS header keywords |
| `validate_sdo_image` | `(image, expected_shape=(4096,4096)) -> bool` | Validate SDO image dimensions |
| `get_solar_disk_params` | `(header) -> dict` | Calculate solar disk center/radius from header |

### Level 1.5

| Function | Signature | Description |
|----------|-----------|-------------|
| `to_level15` | `(fits_file, instrument=None, target_plate_scale=None, target_size=4096, order=3, missing=0.0) -> Map` | Convert Level 1.0 to 1.5 (north-up, centered) |
| `batch_to_level15` | `(fits_files, output_dir, instrument=None, overwrite=False, progress_callback=None, **kwargs) -> List[str]` | Batch Level 1.5 conversion |
| `get_level_info` | `(fits_file) -> dict` | Get processing level info from FITS |

### Stacking

| Class/Function | Signature | Description |
|----------------|-----------|-------------|
| `Stacking` | `(nb_stack, method='mean', latitude_deg=0, ...)` | Solar rotation-corrected stacking |
| `StreamingStackAccumulator` | `(method, ...)` | Memory-efficient streaming accumulator |
| `stack_with_rotation_correction` | `(maps, ...) -> ndarray` | Stack with rotation correction |
| `solar_rotation_shift` | `(delta_t, latitude_deg, plate_scale) -> float` | Calculate pixel shift for solar rotation |
| `snodgrass_rotation_rate` | `(latitude_deg) -> float` | Snodgrass (1983) differential rotation rate |
| `detect_cadence_from_maps` | `(maps) -> float` | Detect time cadence from Map sequence |
| `cross_correlate_shift` | `(ref, target, ...) -> Tuple[float, float]` | Sub-pixel shift via phase correlation |

### Quality

| Function | Signature | Description |
|----------|-----------|-------------|
| `decode_quality` | `(quality, instrument='AIA') -> List[dict]` | Decode QUALITY flag into list of issues |
| `format_quality` | `(quality, instrument='AIA', verbose=True) -> str` | Format QUALITY flag as readable string |
| `is_quality_ok` | `(quality, strict=False, ignore_bits=None) -> bool` | Check if data quality is acceptable |
| `get_quality_summary` | `(quality, instrument='AIA') -> dict` | Get structured quality summary |
| `print_all_quality_bits` | `(instrument='AIA') -> None` | Print all quality bit definitions |

### DEM (Differential Emission Measure)

SITES algorithm implementation for DEM inversion from multi-wavelength AIA observations.
Requires aiapy for accurate temperature response functions.

**Response Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_temperature_response` | `(wavelengths=None, temperatures=None, time=None, include_degradation=True) -> ndarray` | Get AIA temperature response K(T) for each channel |
| `get_default_temperatures` | `(logt_min=5.5, logt_max=7.5, n_bins=100) -> ndarray` | Get default log-spaced temperature grid |

**SITES Algorithm:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `dem_sites` | `(intensities, errors, response, temperatures, max_iter=100, tol=1e-3, ...) -> Tuple[ndarray, dict]` | Multi-wavelength DEM inversion using SITES |
| `dem_sites_pixel` | `(intensities, errors, response, temperatures, ...) -> Tuple[ndarray, dict]` | Single-pixel DEM inversion interface |

**Map Processing:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `dem_map` | `(image_cube, error_cube, response, temperatures, mask=None, chunk_size=512, ...) -> Tuple[ndarray, dict]` | Full-map DEM with chunked processing |
| `compute_dem_errors` | `(dem, intensities, errors, response, temperatures, n_monte_carlo=100) -> ndarray` | Monte Carlo DEM error estimation |

**Derived Quantities:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_emission_measure` | `(dem, temperatures, t_min=None, t_max=None) -> float or ndarray` | Compute total emission measure from DEM |
| `get_mean_temperature` | `(dem, temperatures, weight='dem') -> float or ndarray` | Compute DEM-weighted mean temperature |

**Constants:**

| Name | Value | Description |
|------|-------|-------------|
| `HAS_AIAPY` | bool | True if aiapy is available |

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `AIA_PLATE_SCALE` | 0.6 | AIA plate scale (arcsec/px) |
| `HMI_PLATE_SCALE` | 0.6 | HMI plate scale (arcsec/px) |
| `SDO_IMAGE_SIZE` | 4096 | Standard SDO image size |
| `SOLAR_ROTATION_PERIOD` | 25.38 | Sidereal rotation period (days) |
| `SNODGRASS_A` | 14.713 | Snodgrass coefficient A |
| `SNODGRASS_B` | -2.396 | Snodgrass coefficient B |
| `SNODGRASS_C` | -1.787 | Snodgrass coefficient C |
| `HMI_CADENCE_45S` | 45.0 | HMI 45-second cadence |
| `HMI_CADENCE_720S` | 720.0 | HMI 720-second cadence |
| `AIA_CALIBRATION` | dict | AIA wavelength calibration data |
| `AIA_QUALITY_BITS` | dict | AIA QUALITY bit definitions |
| `HMI_QUALITY_BITS` | dict | HMI QUALITY bit definitions |
| `QUALLEV0_BITS` | dict | Level 0 quality bit definitions |
| `HAS_ASTROPY` | bool | True if astropy available |
| `HAS_SUNPY` | bool | True if sunpy available |

---

## egghouse.config

ML/DL configuration management.

### Classes

| Class | Description |
|-------|-------------|
| `BaseConfig` | Dataclass-based configuration with multiple loading methods |

### BaseConfig Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_yaml` | `(path) -> T` | Load from YAML file |
| `from_json` | `(path) -> T` | Load from JSON file |
| `from_env` | `(prefix="") -> T` | Load from environment variables |
| `from_args` | `(args=None) -> T` | Load from CLI arguments |
| `to_yaml` | `(path) -> None` | Save to YAML file |
| `to_json` | `(path, indent=2) -> None` | Save to JSON file |

---

## egghouse.database

PostgreSQL database utilities.

### Classes

| Class | Description |
|-------|-------------|
| `PostgresManager` | Simplified PostgreSQL operations |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_config` | `(path) -> dict` | Load database config from YAML/JSON |
| `from_dict` | `(d) -> dict` | Create config from dictionary |
| `create_example_config` | `(path) -> None` | Create example config file |
| `to_dataframe` | `(results, columns) -> DataFrame` | Convert query results to pandas DataFrame |

### PostgresManager Methods

| Method | Description |
|--------|-------------|
| `insert(table, data)` | Insert row(s) into table |
| `select(table, columns=None, where=None)` | Query table with optional conditions |
| `select_date_range(table, date_column, start, end)` | Query by date range |
| `update(table, data, where)` | Update rows matching condition |
| `delete(table, where)` | Delete rows matching condition |
| `upsert(table, data, conflict_columns, update_columns=None)` | Insert or update on conflict |
| `execute(query, params=None)` | Execute raw SQL |
| `to_dataframe(results, columns)` | Convert results to DataFrame |

---

## egghouse.transfer

File transfer utilities for HTTP, FTP, and SFTP protocols.

### HTTP

| Function | Signature | Description |
|----------|-----------|-------------|
| `download_single_file` | `(source_url, destination, overwrite=False, max_retries=3, timeout=30, verify_ssl=True) -> bool` | Download single file with retry |
| `get_file_list` | `(base_url, extensions, timeout=30, verify_ssl=True) -> List[str]` | Scrape file links from directory listing |
| `download_parallel` | `(download_tasks, overwrite=False, max_retries=3, parallel=1, timeout=30, verify_ssl=True) -> Dict[str, int]` | Parallel download with ThreadPoolExecutor |

### FTP (no external dependencies)

| Function | Signature | Description |
|----------|-----------|-------------|
| `ftp_connection` | `(host, port=21, user='anonymous', password='', timeout=30, passive=True) -> ContextManager[FTP]` | Context manager for FTP connection |
| `ftp_download_file` | `(ftp, remote_path, local_path, overwrite=False) -> bool` | Download single file via FTP |
| `ftp_upload_file` | `(ftp, local_path, remote_path, overwrite=False) -> bool` | Upload single file via FTP |
| `ftp_list_files` | `(ftp, remote_dir='.', extensions=None) -> List[str]` | List files in remote directory |
| `ftp_download_parallel` | `(host, download_tasks, port=21, user='anonymous', password='', overwrite=False, max_retries=3, parallel=1, timeout=30, passive=True) -> Dict[str, int]` | Parallel FTP download |
| `ftp_upload_parallel` | `(host, upload_tasks, port=21, user='anonymous', password='', overwrite=False, max_retries=3, parallel=1, timeout=30, passive=True) -> Dict[str, int]` | Parallel FTP upload |

### SFTP (requires paramiko)

| Function | Signature | Description |
|----------|-----------|-------------|
| `sftp_connection` | `(host, port=22, user=None, password=None, key_file=None, timeout=30) -> ContextManager[SFTPClient]` | Context manager for SFTP connection |
| `sftp_download_file` | `(sftp, remote_path, local_path, overwrite=False) -> bool` | Download single file via SFTP |
| `sftp_upload_file` | `(sftp, local_path, remote_path, overwrite=False) -> bool` | Upload single file via SFTP |
| `sftp_list_files` | `(sftp, remote_dir='.', extensions=None) -> List[str]` | List files in remote directory |
| `sftp_download_parallel` | `(host, download_tasks, port=22, user=None, password=None, key_file=None, overwrite=False, max_retries=3, parallel=1, timeout=30) -> Dict[str, int]` | Parallel SFTP download |
| `sftp_upload_parallel` | `(host, upload_tasks, port=22, user=None, password=None, key_file=None, overwrite=False, max_retries=3, parallel=1, timeout=30) -> Dict[str, int]` | Parallel SFTP upload |

### Constants

| Name | Description |
|------|-------------|
| `HAS_HTTP` | `True` if requests/beautifulsoup4 are available for HTTP |
| `HAS_PARAMIKO` | `True` if paramiko is available for SFTP |
