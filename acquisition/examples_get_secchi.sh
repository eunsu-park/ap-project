#!/bin/bash

# SECCHI (STEREO) Data Downloader Usage Examples
# Make sure you have Python 3.6+ and required packages installed:
# pip install requests beautifulsoup4

echo "=== SECCHI (STEREO) Data Downloader Examples ==="
echo

# Example 1: Basic usage
echo "Example 1: Download recent 7 days of STEREO-A COR2 science data"
echo "python secchi.py"
python secchi.py
echo

# Example 2: Specific date range
echo "Example 2: Download specific date range"
echo "python secchi.py --start-date 2024-01-01 --end-date 2024-01-07"
python secchi.py --start-date 2024-01-01 --end-date 2024-01-07
echo

# Example 3: Multiple instruments
echo "Example 3: Download COR1 and COR2 coronagraph data"
echo "python secchi.py --instruments cor1 cor2 --days 3"
python secchi.py --instruments cor1 cor2 --days 3
echo

# Example 4: Both spacecraft
echo "Example 4: Download from both STEREO-A and STEREO-B"
echo "python secchi.py --spacecrafts ahead behind --days 2"
python secchi.py --spacecrafts ahead behind --days 2
echo

# Example 5: Science and beacon data
echo "Example 5: Download both science and beacon data"
echo "python secchi.py --data-types science beacon --days 1"
python secchi.py --data-types science beacon --days 1
echo

# Example 6: Heliospheric imagers
echo "Example 6: Download heliospheric imager data"
echo "python secchi.py --instruments hi_1 hi_2 --days 2"
python secchi.py --instruments hi_1 hi_2 --days 2
echo

# Example 7: EUVI telescope data
echo "Example 7: Download EUVI extreme ultraviolet data"
echo "python secchi.py --instruments euvi --categories img cal --days 1"
python secchi.py --instruments euvi --categories img cal --days 1
echo

# Example 8: Parallel download
echo "Example 8: Parallel download with custom destination"
echo "python secchi.py --parallel 4 --destination ./stereo_data --days 3"
python secchi.py --parallel 4 --destination ./stereo_data --days 3
echo

# Example 9: All SECCHI instruments
echo "Example 9: Download all SECCHI instruments"
echo "python secchi.py --instruments cor1 cor2 euvi hi_1 hi_2 --days 1"
python secchi.py --instruments cor1 cor2 euvi hi_1 hi_2 --days 1
echo

# Example 10: Different file extensions
echo "Example 10: Download FTS and SAV files"
echo "python secchi.py --extensions fts sav --days 2"
python secchi.py --extensions fts sav --days 2
echo

# Example 11: Comprehensive download
echo "Example 11: Comprehensive SECCHI download"
echo "python secchi.py --start-date 2023-12-01 --end-date 2023-12-03 --data-types science --spacecrafts ahead --instruments cor1 cor2 euvi --categories img --extensions fts --parallel 2"
python secchi.py --start-date 2023-12-01 --end-date 2023-12-03 --data-types science --spacecrafts ahead --instruments cor1 cor2 euvi --categories img --extensions fts --parallel 2
echo

echo "=== SECCHI Parameter Explanations ==="
echo
echo "Data Types:"
echo "  science : Processed L0 data products"
echo "  beacon  : Near real-time data"
echo
echo "Spacecrafts:"
echo "  ahead   : STEREO-A spacecraft"
echo "  behind  : STEREO-B spacecraft (contact lost in 2014)"
echo
echo "Categories:"
echo "  img : Image data"
echo "  cal : Calibration data"
echo "  seq : Sequence data"
echo
echo "Instruments:"
echo "  cor1 : Inner coronagraph (1.4-4.0 solar radii)"
echo "  cor2 : Outer coronagraph (2.5-15 solar radii)"
echo "  euvi : Extreme Ultraviolet Imager"
echo "  hi_1 : Heliospheric Imager 1 (inner field)"
echo "  hi_2 : Heliospheric Imager 2 (outer field)"
echo
echo "Extensions:"
echo "  fts : FITS files (standard astronomical format)"
echo "  sav : IDL save files"
echo
echo "=== STEREO Mission Information ==="
echo
echo "Mission Details:"
echo "  Launch: 2006-10-27"
echo "  Primary Mission: 2 years (extended multiple times)"
echo "  Orbit: Solar orbit, leading/trailing Earth"
echo
echo "SECCHI Suite:"
echo "  Full name: Sun Earth Connection Coronal and Heliospheric Investigation"
echo "  Purpose: Study coronal mass ejections and solar wind"
echo "  Unique capability: Stereoscopic solar observations"
echo
echo "Data Availability:"
echo "  STEREO-A: 2006-present (active)"
echo "  STEREO-B: 2006-2014 (contact lost)"
echo "  Real-time: Beacon data available within hours"
echo "  Archive: Science data processed and archived"
echo
echo "=== Output Directory Structure ==="
echo
echo "secchi_data/"
echo "├── science/"
echo "│   ├── ahead/"
echo "│   │   └── img/"
echo "│   │       ├── cor1/2024/20240101/"
echo "│   │       ├── cor2/2024/20240101/"
echo "│   │       ├── euvi/2024/20240101/"
echo "│   │       ├── hi_1/2024/20240101/"
echo "│   │       └── hi_2/2024/20240101/"
echo "│   └── behind/"
echo "│       └── img/"
echo "│           └── ..."
echo "└── beacon/"
echo "    ├── ahead/"
echo "    └── behind/"
echo
echo "=== Usage Tips ==="
echo
echo "1. Start with COR2 data - most commonly used coronagraph"
echo "2. Use 'science' data type for research - better quality than beacon"
echo "3. STEREO-B data only available until 2014"
echo "4. Large downloads: use --parallel for faster processing"
echo "5. Check disk space - coronagraph data can be large"
echo "6. For CME studies: combine COR1 and COR2 for full field of view"
echo
echo "=== All SECCHI examples completed ==="