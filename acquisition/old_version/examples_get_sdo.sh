#!/bin/bash

# SDO JP2 Downloader Usage Examples
# Make sure you have Python 3.6+ and required packages installed:
# pip install requests beautifulsoup4

echo "=== SDO JP2 Downloader Usage Examples ==="
echo

# Example 1: Download recent 2 days data (default)
echo "Example 1: Download recent 2 days data"
echo "python get_sdo.py"
python get_sdo.py
echo

# Example 2: Download specific date
echo "Example 2: Download specific date (2024-01-15)"
echo "python get_sdo.py --year 2024 --month 1 --day 15"
python get_sdo.py --year 2024 --month 1 --day 15
echo

# Example 3: Download last 5 days
echo "Example 3: Download last 5 days"
echo "python get_sdo.py --days 5"
python get_sdo.py --days 5
echo

# Example 4: Download specific date range
echo "Example 4: Download data from 2024-01-01 to 2024-01-31"
echo "python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-31"
python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-31
echo

# Example 5: Download from start date to today
echo "Example 5: Download from 2024-01-15 to today"
echo "python get_sdo.py --start-date 2024-01-15"
python get_sdo.py --start-date 2024-01-15
echo

# Example 6: Download until end date (using SDO mission start date 2010-09-01)
echo "Example 6: Download from SDO mission start to 2024-01-31"
echo "python get_sdo.py --end-date 2024-01-31"
python get_sdo.py --end-date 2024-01-31
echo

# Example 7: Date range with specific wavelengths
echo "Example 7: Download specific wavelengths for date range"
echo "python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-07 --waves 94 171 304"
python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-07 --waves 94 171 304
echo

# Example 8: Parallel download with date range
echo "Example 8: Parallel download with date range"
echo "python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-31 --parallel 4"
python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-31 --parallel 4
echo

# Example 6: Download with custom destination
echo "Example 9: Download to custom directory with date range"
echo "python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-07 --destination ./sdo_data"
python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-07 --destination ./sdo_data
echo

# Example 7: Download with overwrite and detailed logging
echo "Example 10: Download with overwrite, debug logging and date range"
echo "python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-03 --overwrite --log-level DEBUG --log-file sdo_download.log"
python get_sdo.py --start-date 2024-01-01 --end-date 2024-01-03 --overwrite --log-level DEBUG --log-file sdo_download.log
echo

# Example 8: Comprehensive download command with date range
echo "Example 11: Comprehensive download with date range"
echo "python get_sdo.py --start-date 2024-01-10 --end-date 2024-01-12 --waves 193 211 171 --parallel 3 --destination ./sdo_data --overwrite --max-retries 5"
python get_sdo.py --start-date 2024-01-10 --end-date 2024-01-12 --waves 193 211 171 --parallel 3 --destination ./sdo_data --overwrite --max-retries 5
echo

echo "=== Additional Date Range Examples ==="
echo

# Example 12: Download entire SDO mission data (mission start to today)
echo "Example 12: Download entire SDO mission data (2010-09-01 to today) - WARNING: Very large download!"
echo "# python get_sdo.py --start-date 2010-09-01 --waves 193"
echo "(Commented out - would download 14+ years of data)"
echo

# Example 13: Download one year of data
echo "Example 13: Download one year of SDO data (2023)"
echo "python get_sdo.py --start-date 2023-01-01 --end-date 2023-12-31 --waves 193"
python get_sdo.py --start-date 2023-01-01 --end-date 2023-12-31 --waves 193
echo

# Example 14: Download recent data with mission start as default
echo "Example 14: Download recent data (uses SDO mission start 2010-09-01 as default start)"
echo "python get_sdo.py --end-date 2024-01-07"
python get_sdo.py --end-date 2024-01-07
echo

echo "=== All examples completed ==="