#!/bin/bash

# OMNI Data Processor Usage Examples
# Make sure you have Python 3.6+ and required packages installed:
# pip install requests beautifulsoup4 pyyaml pandas

echo "=== OMNI Data Processor Usage Examples ==="
echo

# Example 1: Download single year of low resolution data
echo "Example 1: Download 2024 low resolution hourly data"
echo "python get_omni.py --dataset omni_low_res --year 2024"
python get_omni.py --dataset omni_low_res --year 2024
echo

# Example 2: Download single year of high resolution data
echo "Example 2: Download 2024 high resolution 1-minute data"
echo "python get_omni.py --dataset omni_high_res --year 2024"
python get_omni.py --dataset omni_high_res --year 2024
echo

# Example 3: Download multiple years
echo "Example 3: Download low resolution data from 2020-2024"
echo "python get_omni.py --dataset omni_low_res --start-year 2020 --end-year 2024"
python get_omni.py --dataset omni_low_res --start-year 2020 --end-year 2024
echo

# Example 4: Download with custom output directory
echo "Example 4: Download to custom directory"
echo "python get_omni.py --dataset omni_high_res --year 2023 --output-dir /data/omni"
python get_omni.py --dataset omni_high_res --year 2023 --output-dir /data/omni
echo

# Example 5: Download with overwrite and custom timeout
echo "Example 5: Download with overwrite and extended timeout"
echo "python get_omni.py --dataset omni_low_res --year 2024 --overwrite --timeout 120"
python get_omni.py --dataset omni_low_res --year 2024 --overwrite --timeout 120
echo

# Example 6: Download with detailed logging
echo "Example 6: Download with debug logging to file"
echo "python get_omni.py --dataset omni_high_res --year 2024 --log-level DEBUG --log-file omni_debug.log"
python get_omni.py --dataset omni_high_res --year 2024 --log-level DEBUG --log-file omni_debug.log
echo

# Example 7: Download recent years of both datasets
echo "Example 7: Download both low and high resolution for 2023-2024"
echo "Low resolution:"
python get_omni.py --dataset omni_low_res --start-year 2023 --end-year 2024 --output-dir ./omni_data/low_res
echo "High resolution:"
python get_omni.py --dataset omni_high_res --start-year 2023 --end-year 2024 --output-dir ./omni_data/high_res
echo

# Example 8: Download with custom config file
echo "Example 8: Download using custom config file"
echo "python get_omni.py --config omni_config.yaml --dataset omni_low_res --year 2024"
# python get_omni.py --config omni_config.yaml --dataset omni_low_res --year 2024
echo "(Commented out - requires custom config file)"
echo

echo "=== Data Structure Information ==="
echo "Downloaded CSV files will contain:"
echo "- Low resolution: Hourly data from 1963 to present"
echo "- High resolution: 1-minute data from 1995 to present"
echo "- Fill values are automatically converted to NaN/None"
echo "- Output format: CSV with proper column names and units"
echo

echo "=== Output Directory Structure ==="
echo "omni_data/"
echo "├── omni2_2024.csv      (low resolution)"
echo "├── omni_min2024.csv    (high resolution)"
echo "└── ..."
echo

echo "=== All examples completed ==="