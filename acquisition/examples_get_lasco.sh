#!/bin/bash

# LASCO (SOHO) Data Downloader Usage Examples
# Make sure you have Python 3.6+ and required packages installed:
# pip install requests beautifulsoup4

echo "=== LASCO (SOHO) Data Downloader Examples ==="
echo

# Example 1: Basic usage
echo "Example 1: Download recent 7 days of LASCO C2 data"
echo "python lasco.py"
python lasco.py
echo

# Example 2: Specific date range
echo "Example 2: Download specific date range"
echo "python lasco.py --start-date 2024-01-01 --end-date 2024-01-07"
python lasco.py --start-date 2024-01-01 --end-date 2024-01-07
echo

# Example 3: Multiple cameras
echo "Example 3: Download C2 and C3 coronagraph data"
echo "python lasco.py --cameras c2 c3 --days 3"
python lasco.py --cameras c2 c3 --days 3
echo

# Example 4: All cameras
echo "Example 4: Download from all LASCO cameras"
echo "python lasco.py --cameras c1 c2 c3 c4 --days 1"
python lasco.py --cameras c1 c2 c3 c4 --days 1
echo

# Example 5: Multiple file formats
echo "Example 5: Download FITS and JPEG files"
echo "python lasco.py --extensions fts jpg --days 2"
python lasco.py --extensions fts jpg --days 2
echo

# Example 6: Parallel download
echo "Example 6: Parallel download with custom destination"
echo "python lasco.py --parallel 3 --destination ./soho_data --days 3"
python lasco.py --parallel 3 --destination ./soho_data --days 3
echo

# Example 7: Long time series
echo "Example 7: Download one month of C2 data"
echo "python lasco.py --start-date 2023-12-01 --end-date 2023-12-31 --cameras c2 --parallel 2"
python lasco.py --start-date 2023-12-01 --end-date 2023-12-31 --cameras c2 --parallel 2
echo

# Example 8: C2 only (most common)
echo "Example 8: Download only C2 camera data (most commonly used)"
echo "python lasco.py --cameras c2 --extensions fts --days 5"
python lasco.py --cameras c2 --extensions fts --days 5
echo

# Example 9: Wide field C3 data
echo "Example 9: Download C3 wide-field coronagraph data"
echo "python lasco.py --cameras c3 --start-date 2024-01-01 --end-date 2024-01-10"
python lasco.py --cameras c3 --start-date 2024-01-01 --end-date 2024-01-10
echo

# Example 10: All file types
echo "Example 10: Download all available file types"
echo "python lasco.py --extensions fts jpg txt --days 2"
python lasco.py --extensions fts jpg txt --days 2
echo

# Example 11: Debug mode
echo "Example 11: Debug mode with detailed logging"
echo "python lasco.py --log-level DEBUG --log-file lasco_debug.log --days 1"
python lasco.py --log-level DEBUG --log-file lasco_debug.log --days 1
echo

echo "=== LASCO Parameter Explanations ==="
echo
echo "Cameras:"
echo "  c1 : Inner coronagraph (1.1-3.0 solar radii) - discontinued 1998"
echo "  c2 : Main coronagraph (2.0-6.0 solar radii) - primary CME detector"
echo "  c3 : Wide field coronagraph (3.7-30 solar radii) - outer corona"
echo "  c4 : Backup/calibration camera - rarely used"
echo
echo "Extensions:"
echo "  fts : FITS files (science quality, calibrated data)"
echo "  jpg : JPEG files (quicklook images for browsing)"
echo "  txt : Text files (metadata, pointing information)"
echo
echo "=== SOHO/LASCO Mission Information ==="
echo
echo "Mission Details:"
echo "  Launch: 1995-12-02"
echo "  Operations start: 1996-01-01"
echo "  First light: 1995-12-08"
echo "  Status: Active (29+ years of operations)"
echo "  Orbit: L1 Lagrange point (1.5 million km from Earth)"
echo
echo "LASCO Instrument:"
echo "  Full name: Large Angle and Spectrometric Coronagraph"
echo "  Purpose: Study solar corona and coronal mass ejections"
echo "  Achievement: Discovered >5000 comets"
echo "  Record: Longest operating space-based coronagraph"
echo
echo "Camera Details:"
echo "  C1: 1.1-3.0 Rs (discontinued June 1998)"
echo "  C2: 2.0-6.0 Rs (primary instrument for CME detection)"
echo "  C3: 3.7-30 Rs (wide field for large CME structures)"
echo "  Resolution: 1024x1024 pixels"
echo "  Cadence: Typically 12-96 minutes depending on operations"
echo
echo "Scientific Impact:"
echo "  - Revolutionized understanding of CMEs"
echo "  - Enabled space weather predictions"
echo "  - Discovered majority of known sungrazing comets"
echo "  - Provided context for other solar missions"
echo
echo "=== Output Directory Structure ==="
echo
echo "lasco_data/"
echo "├── c2/"
echo "│   └── 2024/"
echo "│       ├── 20240101/"
echo "│       ├── 20240102/"
echo "│       └── ..."
echo "├── c3/"
echo "│   └── 2024/"
echo "│       ├── 20240101/"
echo "│       ├── 20240102/"
echo "│       └── ..."
echo "└── c4/"
echo "    └── ..."
echo
echo "=== Usage Recommendations ==="
echo
echo "1. Start with C2 data:"
echo "   - Most commonly used for CME studies"
echo "   - Best signal-to-noise ratio"
echo "   - Continuous operations since 1996"
echo
echo "2. File format selection:"
echo "   - Use 'fts' for scientific analysis"
echo "   - Use 'jpg' for quick browsing/movies"
echo "   - Use 'txt' for metadata requirements"
echo
echo "3. Time range considerations:"
echo "   - C1 data only available 1996-1998"
echo "   - Data gaps exist during SOHO emergencies"
echo "   - Some periods have reduced cadence"
echo
echo "4. Performance tips:"
echo "   - Use parallel downloads for large time ranges"
echo "   - C2+C3 combination good for full CME evolution"
echo "   - Consider disk space - FITS files can be large"
echo
echo "=== Common Use Cases ==="
echo
echo "CME Studies:"
echo "  python lasco.py --cameras c2 c3 --start-date YYYY-MM-DD --end-date YYYY-MM-DD"
echo
echo "Comet Hunting:"
echo "  python lasco.py --cameras c2 c3 --extensions fts --days 30"
echo
echo "Space Weather:"
echo "  python lasco.py --cameras c2 --extensions fts jpg --days 7"
echo
echo "Long-term Studies:"
echo "  python lasco.py --cameras c2 --start-date 2023-01-01 --end-date 2023-12-31 --parallel 4"
echo
echo "=== All LASCO examples completed ==="