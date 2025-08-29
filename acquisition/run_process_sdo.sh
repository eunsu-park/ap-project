#!/bin/bash

# SDO Image Processor Usage Examples
# Make sure you have Python 3.6+ and required packages installed:
# pip install opencv-python numpy

echo "=== SDO Image Processor Usage Examples ==="
echo

# Example 1: Basic sequential processing with default settings
echo "Example 1: Basic sequential processing"
echo "python process_sdo.py"
python process_sdo.py
echo

# Example 2: Parallel processing with default settings
echo "Example 2: Parallel processing (default workers)"
echo "python process_sdo.py --parallel"
python process_sdo.py --parallel
echo

# Example 3: Parallel processing with custom worker count
echo "Example 3: Parallel processing with 4 workers"
echo "python process_sdo.py --parallel --max-workers 4"
python process_sdo.py --parallel --max-workers 4
echo

# Example 4: Custom date range
echo "Example 4: Process specific date range (2020-2021)"
echo "python process_sdo.py --start-year 2020 --end-year 2021 --parallel"
python process_sdo.py --start-year 2020 --end-year 2021 --parallel
echo

# Example 5: Custom input/output directories
echo "Example 5: Custom directories"
echo "python process_sdo.py --input-dir /data/sdo_input --output-dir /data/sdo_output --parallel"
python process_sdo.py --input-dir /data/sdo_input --output-dir /data/sdo_output --parallel
echo

# Example 6: Process specific wavelengths
echo "Example 6: Process multiple wavelengths"
echo "python process_sdo.py --waves 94 131 171 193 211 304 --parallel"
python process_sdo.py --waves 94 131 171 193 211 304 --parallel
echo

# Example 7: Custom sequence parameters
echo "Example 7: Custom sequence (12-hour steps, 10 images)"
echo "python process_sdo.py --time-step 12 --num-sequence 10 --parallel"
python process_sdo.py --time-step 12 --num-sequence 10 --parallel
echo

# Example 8: Different processing interval
echo "Example 8: Process every 6 hours instead of 3"
echo "python process_sdo.py --interval 6 --parallel"
python process_sdo.py --interval 6 --parallel
echo

# Example 9: Force reprocessing existing files
echo "Example 9: Force reprocess existing files"
echo "python process_sdo.py --no-skip --parallel"
python process_sdo.py --no-skip --parallel
echo

# Example 10: Short date range for testing
echo "Example 10: Test with short date range (1 week)"
echo "python process_sdo.py --start-year 2010 --start-month 9 --start-day 1 --end-year 2010 --end-month 9 --end-day 7 --parallel"
python process_sdo.py --start-year 2010 --start-month 9 --start-day 1 --end-year 2010 --end-month 9 --end-day 7 --parallel
echo

# Example 11: Comprehensive custom settings
echo "Example 11: Comprehensive custom processing"
echo "python process_sdo.py \\"
echo "  --input-dir /Users/eunsupark/ap_project/data/sdo_jp2/aia \\"
echo "  --output-dir /Users/eunsupark/ap_project/data/processed \\"
echo "  --start-year 2020 --start-month 1 --start-day 1 \\"
echo "  --end-year 2020 --end-month 12 --end-day 31 \\"
echo "  --waves 193 211 304 \\"
echo "  --time-step 6 --num-sequence 20 --interval 3 \\"
echo "  --parallel --max-workers 6"
python process_sdo.py \
  --input-dir /Users/eunsupark/ap_project/data/sdo_jp2/aia \
  --output-dir /Users/eunsupark/ap_project/data/processed \
  --start-year 2020 --start-month 1 --start-day 1 \
  --end-year 2020 --end-month 12 --end-day 31 \
  --waves 193 211 304 \
  --time-step 6 --num-sequence 20 --interval 3 \
  --parallel --max-workers 6
echo

# Example 12: Single wavelength processing for testing
echo "Example 12: Single wavelength (193) for faster testing"
echo "python process_sdo.py --waves 193 --start-year 2010 --start-month 9 --end-year 2010 --end-month 9 --parallel"
python process_sdo.py --waves 193 --start-year 2010 --start-month 9 --end-year 2010 --end-month 9 --parallel
echo

echo "=== Parameter Explanations ==="
echo
echo "Date Parameters:"
echo "  --start-year, --start-month, --start-day, --start-hour : Start date/time"
echo "  --end-year, --end-month, --end-day, --end-hour : End date/time"
echo
echo "Processing Parameters:"
echo "  --waves : List of wavelengths to process (e.g., 94 131 171 193 211 304 335 1600 1700)"
echo "  --time-step : Hours between images in a sequence (default: 6)"
echo "  --num-sequence : Number of images per sequence (default: 20)"
echo "  --interval : Hours between sequence start times (default: 3)"
echo
echo "Processing Options:"
echo "  --parallel : Enable parallel processing"
echo "  --max-workers : Number of parallel workers (default: auto-detect)"
echo "  --skip-existing : Skip already processed sequences (default: True)"
echo "  --no-skip : Force reprocess existing files"
echo
echo "Directory Options:"
echo "  --input-dir : Input directory containing SDO JP2 files"
echo "  --output-dir : Output directory for processed sequences"
echo
echo "=== Sequence Structure ==="
echo "Each sequence contains:"
echo "  - 20 images (default) per wavelength"
echo "  - 6-hour intervals between images (default)"
echo "  - Total sequence duration: 20 × 6 = 120 hours (5 days)"
echo "  - Sequences start every 3 hours (default), creating overlapping sequences"
echo
echo "Output directory structure:"
echo "processed/"
echo "├── 2010090100-2010090523/"  # YYYYMMDDHH-YYYYMMDDHH format
echo "│   ├── 193/"
echo "│   │   ├── image1.jp2"
echo "│   │   └── ..."
echo "│   └── 211/"
echo "│       ├── image1.jp2"
echo "│       └── ..."
echo "├── 2010090103-2010090602/"  # Next sequence (3 hours later)"
echo "└── ..."
echo
echo "=== All examples completed ==="