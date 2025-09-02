#!/usr/bin/env python3
"""
Simplified OMNI Data Processor
Extracts time-matched OMNI data for SDO image sequences
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from glob import glob
import argparse

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from utils import TimeSeriesProcessor, DirectoryTimeExtractor, setup_logging

def process_single_dataset(args_tuple):
    """Worker function for parallel processing"""
    dataset_dir, omni_dir, interval_hours, sequence_days, time_offset = args_tuple
    
    # Create processor for this worker (no specific columns = all columns)
    processor = TimeSeriesProcessor(omni_dir, columns=None)
    
    # Extract start time from directory name
    dir_name = os.path.basename(dataset_dir)
    start_time = DirectoryTimeExtractor.extract_from_pattern(dir_name)
    
    if start_time is None:
        print(f"Cannot extract time from directory: {dir_name}")
        return False
    
    # Calculate end time
    end_time = start_time + timedelta(days=sequence_days)
    
    # Output file path
    output_path = os.path.join(dataset_dir, "hourly_data.csv")
    
    # Skip if already exists
    if os.path.exists(output_path):
        print(f"Already processed: {dir_name}")
        return True
    
    print(f"Processing {dir_name}: {start_time} to {end_time}")
    
    # Process time range
    results = processor.process_time_range(start_time, end_time, interval_hours, time_offset)
    
    # Count successful extractions
    success_count = sum(1 for r in results if len(r) > 2)  # More than just base_time and target_time
    
    # Save results
    if processor.save_results(results, output_path):
        print(f"Saved {len(results)} records ({success_count} with data) to {output_path}")
        return True
    else:
        print(f"Failed to save results for {dir_name}")
        return False

def process_dataset(dataset_dir: str, processor: TimeSeriesProcessor, 
                   interval_hours: int = 3, sequence_days: int = 8, 
                   time_offset_hours: int = -1) -> bool:
    """Process single dataset directory"""
    
    # Extract start time from directory name
    dir_name = os.path.basename(dataset_dir)
    start_time = DirectoryTimeExtractor.extract_from_pattern(dir_name)
    
    if start_time is None:
        print(f"Cannot extract time from directory: {dir_name}")
        return False
    
    # Calculate end time
    end_time = start_time + timedelta(days=sequence_days)
    
    # Output file path
    output_path = os.path.join(dataset_dir, "hourly_data.csv")
    
    # Skip if already exists
    if os.path.exists(output_path):
        print(f"Already processed: {dir_name}")
        return True
    
    print(f"Processing {dir_name}: {start_time} to {end_time}")
    
    # Process time range
    results = processor.process_time_range(start_time, end_time, interval_hours, time_offset_hours)
    
    # Count successful extractions (rows with more than just time columns)
    success_count = sum(1 for r in results if len(r) > 2)
    
    # Save results
    if processor.save_results(results, output_path):
        print(f"Saved {len(results)} records ({success_count} with data) to {output_path}")
        return True
    else:
        print(f"Failed to save results for {dir_name}")
        return False

def main():
    parser = argparse.ArgumentParser(description="OMNI Data Processor")
    
    # Input/Output paths
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Directory containing processed SDO datasets')
    parser.add_argument('--omni-dir', type=str, required=True,
                       help='Directory containing OMNI CSV files')
    
    # OMNI columns to extract (optional - if not specified, all columns will be extracted)
    parser.add_argument('--columns', type=str, nargs='+', default=None,
                       help='Specific OMNI columns to extract (default: all columns)')
    
    # Processing parameters
    parser.add_argument('--interval', type=int, default=3,
                       help='Time interval in hours')
    parser.add_argument('--sequence-days', type=int, default=8,
                       help='Sequence length in days')
    parser.add_argument('--time-offset', type=int, default=-1,
                       help='Time offset in hours (negative for past data)')
    
    # Processing options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum parallel workers')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Validate directories
    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset directory not found: {args.dataset_dir}")
        return 1
    
    if not os.path.isdir(args.omni_dir):
        print(f"OMNI directory not found: {args.omni_dir}")
        return 1
    
    # Find dataset directories
    dataset_dirs = sorted([d for d in glob(f"{args.dataset_dir}/*") 
                          if os.path.isdir(d) and os.path.basename(d)[0].isdigit()])
    
    if not dataset_dirs:
        print(f"No valid dataset directories found in {args.dataset_dir}")
        return 1
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    if args.columns:
        print(f"OMNI columns: {args.columns}")
    else:
        print("Extracting all OMNI columns")
    print(f"Time offset: {args.time_offset} hours")
    print(f"Processing interval: {args.interval} hours")
    print(f"Sequence duration: {args.sequence_days} days")
    
    # Create processor
    processor = TimeSeriesProcessor(args.omni_dir, args.columns)
    
    # Process datasets
    if args.parallel:
        # Parallel processing
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import cpu_count
        
        max_workers = args.max_workers or min(cpu_count(), len(dataset_dirs))
        print(f"Using {max_workers} parallel workers")
        
        # Prepare arguments for worker function (removed columns parameter)
        worker_args = [(dataset_dir, args.omni_dir, args.interval, 
                       args.sequence_days, args.time_offset) for dataset_dir in dataset_dirs]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_dataset, worker_args))
        
        successful = sum(results)
        
    else:
        # Sequential processing
        print("Processing sequentially")
        successful = 0
        
        for i, dataset_dir in enumerate(dataset_dirs, 1):
            print(f"[{i}/{len(dataset_dirs)}]", end=" ")
            if process_dataset(dataset_dir, processor, args.interval, args.sequence_days, args.time_offset):
                successful += 1
    
    # Final results
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful}/{len(dataset_dirs)} datasets")
    
    if successful < len(dataset_dirs):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())