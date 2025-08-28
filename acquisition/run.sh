#!/bin/bash

# SDO JP2 Downloader Usage Examples
# Make sure you have Python 3.6+ and required packages installed:
# pip install requests beautifulsoup4

echo "=== SDO JP2 Downloader Usage Examples ==="
echo

# Example 1: Download recent 2 days data (default)
echo "Example 1: Download recent 2 days data"
echo "python sdo_jp2.py"
python sdo_jp2.py
echo

# Example 2: Download specific date
echo "Example 2: Download specific date (2024-01-15)"
echo "python sdo_jp2.py --year 2024 --month 1 --day 15"
python sdo_jp2.py --year 2024 --month 1 --day 15
echo

# Example 3: Download last 5 days
echo "Example 3: Download last 5 days"
echo "python sdo_jp2.py --days 5"
python sdo_jp2.py --days 5
echo

# Example 4: Download specific wavelengths
echo "Example 4: Download specific wavelengths (94, 171, 304)"
echo "python sdo_jp2.py --waves 94 171 304"
python sdo_jp2.py --waves 94 171 304
echo

# Example 5: Parallel download with 4 threads
echo "Example 5: Parallel download with 4 threads"
echo "python sdo_jp2.py --parallel 4"
python sdo_jp2.py --parallel 4
echo

# Example 6: Download with custom destination
echo "Example 6: Download to custom directory"
echo "python sdo_jp2.py --destination /data/solar/aia"
python sdo_jp2.py --destination /data/solar/aia
echo

# Example 7: Download with overwrite and detailed logging
echo "Example 7: Download with overwrite and debug logging"
echo "python sdo_jp2.py --overwrite --log-level DEBUG --log-file sdo_download.log"
python sdo_jp2.py --overwrite --log-level DEBUG --log-file sdo_download.log
echo

# Example 8: Comprehensive download command
echo "Example 8: Comprehensive download"
echo "python sdo_jp2.py --year 2024 --month 1 --day 10 --waves 193 211 171 --parallel 3 --destination ./sdo_data --overwrite --max-retries 5"
python sdo_jp2.py --year 2024 --month 1 --day 10 --waves 193 211 171 --parallel 3 --destination ./sdo_data --overwrite --max-retries 5
echo

echo "=== All examples completed ==="