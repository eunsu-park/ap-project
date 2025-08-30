#!/bin/bash

# OMNI Data Processor Usage Examples
# Make sure you have Python 3.6+ and required packages installed:
# pip install pandas

echo "=== OMNI Data Processor Usage Examples ==="
echo

# Example 1: Basic sequential processing
echo "Example 1: Sequential processing with default settings"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni
echo

# Example 2: Parallel processing
echo "Example 2: Parallel processing with auto-detected workers"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --parallel"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --parallel
echo

# Example 3: Custom parallel workers
echo "Example 3: Parallel processing with 4 workers"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --parallel --max-workers 4"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --parallel --max-workers 4
echo

# Example 4: Custom OMNI columns
echo "Example 4: Extract specific OMNI parameters"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --columns Bx_GSE By_GSM Bz_GSM B_magnitude Flow_speed"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --columns Bx_GSE By_GSM Bz_GSM B_magnitude Flow_speed
echo

# Example 5: Different time settings
echo "Example 5: 6-hour intervals with 2-hour time offset"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --interval 6 --time-offset -2"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --interval 6 --time-offset -2
echo

# Example 6: Longer sequences
echo "Example 6: 10-day sequences instead of default 8 days"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --sequence-days 10"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --sequence-days 10
echo

# Example 7: All magnetic field components
echo "Example 7: Extract all magnetic field components"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --columns Bx_GSE By_GSE Bz_GSE By_GSM Bz_GSM B_magnitude --parallel"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --columns Bx_GSE By_GSE Bz_GSE By_GSM Bz_GSM B_magnitude --parallel
echo

# Example 8: Plasma parameters only
echo "Example 8: Extract plasma parameters only"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --columns Flow_speed Proton_density Temperature Flow_pressure --parallel"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --columns Flow_speed Proton_density Temperature Flow_pressure --parallel
echo

# Example 9: Geomagnetic indices
echo "Example 9: Extract geomagnetic indices"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --columns Kp_index ap_index DST_index AE_index --parallel"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --columns Kp_index ap_index DST_index AE_index --parallel
echo

# Example 10: Real paths example
echo "Example 10: Real directory paths with comprehensive settings"
echo "python process_omni.py \\"
echo "  --dataset-dir /Users/eunsupark/ap_project/data/processed \\"
echo "  --omni-dir /Users/eunsupark/ap_project/data/omni/low_res \\"
echo "  --columns Bx_GSE By_GSM Bz_GSM B_magnitude Flow_speed Proton_density Temperature Kp_index DST_index \\"
echo "  --interval 3 --time-offset -1 --sequence-days 8 \\"
echo "  --parallel --max-workers 6 --log-level INFO"
python process_omni.py \
  --dataset-dir /Users/eunsupark/ap_project/data/processed \
  --omni-dir /Users/eunsupark/ap_project/data/omni/low_res \
  --columns Bx_GSE By_GSM Bz_GSM B_magnitude Flow_speed Proton_density Temperature Kp_index DST_index \
  --interval 3 --time-offset -1 --sequence-days 8 \
  --parallel --max-workers 6 --log-level INFO
echo

# Example 11: Debug mode with single dataset
echo "Example 11: Debug mode with detailed logging"
echo "python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --log-level DEBUG"
python process_omni.py --dataset-dir /data/processed --omni-dir /data/omni --log-level DEBUG
echo

echo "=== Parameter Explanations ==="
echo
echo "Required Parameters:"
echo "  --dataset-dir : Directory containing processed SDO image sequences (e.g., /data/processed)"
echo "  --omni-dir    : Directory containing OMNI CSV files (e.g., /data/omni/low_res)"
echo
echo "OMNI Column Options (--columns):"
echo "  Magnetic Field: Bx_GSE By_GSE Bz_GSE By_GSM Bz_GSM B_magnitude"
echo "  Plasma:        Flow_speed Proton_density Temperature Flow_pressure"
echo "  Derived:       Electric_field Plasma_beta Alfven_mach"
echo "  Indices:       Kp_index ap_index DST_index AE_index f107_index R_sunspot"
echo
echo "Time Parameters:"
echo "  --interval      : Hours between sequence start times (default: 3)"
echo "  --time-offset   : Hours to offset from base time (default: -1, i.e., 1 hour before)"
echo "  --sequence-days : Duration of each sequence in days (default: 8)"
echo
echo "Processing Options:"
echo "  --parallel      : Enable parallel processing"
echo "  --max-workers   : Number of parallel workers (default: auto)"
echo "  --log-level     : Logging detail (DEBUG, INFO, WARNING, ERROR)"
echo
echo "=== Input/Output Structure ==="
echo
echo "Expected input structure:"
echo "dataset-dir/"
echo "├── 2010090100-2010090523/    # SDO sequence directory"
echo "├── 2010090103-2010090602/"
echo "└── ..."
echo
echo "omni-dir/"
echo "├── omni2_2010.csv           # OMNI data files by year"
echo "├── omni2_2011.csv"
echo "└── ..."
echo
echo "Generated output:"
echo "dataset-dir/"
echo "├── 2010090100-2010090523/"
echo "│   ├── hourly_data.csv      # Generated OMNI data"
echo "│   ├── 193/"               # SDO images"
echo "│   └── 211/"
echo "└── ..."
echo
echo "Output CSV format:"
echo "base_time,target_time,Bx_GSE,By_GSM,Bz_GSM,..."
echo "2010-09-01 00,2010-08-31 23,1.234,5.678,..."
echo
echo "=== Processing Logic ==="
echo "1. Extract start time from directory name (e.g., '2010090100' -> 2010-09-01 00:00)"
echo "2. Generate time series with specified interval (default: every 3 hours)"
echo "3. For each time point, find OMNI data at offset time (default: -1 hour)"
echo "4. Match using Year/Day/Hour columns from OMNI CSV files"
echo "5. Save results as hourly_data.csv in each sequence directory"
echo
echo "=== All examples completed ==="