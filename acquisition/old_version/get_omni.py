"""
Simplified OMNI Data Processor - Python 3.6+
Downloads and converts NASA OMNI data files to CSV
"""

import sys
from pathlib import Path

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from utils import (
    ConfigLoader, FixedWidthParser, FilePatternSource, 
    TextFileDownloader, setup_logging
)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NASA OMNI Data Processor")
    
    # Basic options
    parser.add_argument("--config", type=str, default="omni_config.yaml", help="Configuration file path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (omni_low_res, omni_high_res)")
    parser.add_argument("--year", type=int, help="Single year to process")
    parser.add_argument("--start-year", type=int, help="Start year")
    parser.add_argument("--end-year", type=int, help="End year")
    parser.add_argument("--output-dir", type=str, default="./omni_data", help="Output directory")
    
    # Download options
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--timeout", type=int, default=60, help="Download timeout")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, help="Log file path")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Load configuration
        config = ConfigLoader.load_config(args.config)
        
        # Get dataset config
        if args.dataset not in config['datasets']:
            available = list(config['datasets'].keys())
            print(f"Dataset '{args.dataset}' not found. Available: {available}")
            return
        
        dataset_config = config['datasets'][args.dataset]
        
        # Create parser from column definitions
        field_specs = []
        for col_def in dataset_config['columns']:
            spec = {
                'name': col_def[0],
                'format': col_def[1],
                'fill_value': col_def[2],
                'description': col_def[3],
                'unit': col_def[4]
            }
            field_specs.append(spec)
        
        parser = FixedWidthParser(field_specs)
        
        # Create data source and downloader
        source = FilePatternSource(dataset_config)
        downloader = TextFileDownloader(
            source,
            parser=parser,
            timeout=args.timeout,
            max_retries=args.max_retries,
            overwrite=args.overwrite
        )
        
        # Determine year range
        if args.year:
            years = [args.year]
        else:
            start_year = args.start_year or config['default_settings']['default_years']['start_year']
            end_year = args.end_year or config['default_settings']['default_years']['end_year']
            years = range(start_year, end_year + 1)
        
        # Process files
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for year in years:
            print(f"Processing year {year}")
            output_path = str(output_dir / dataset_config['output_pattern'].format(year=year))
            
            success = downloader.download_and_convert(output_path, year=year)
            if success:
                print(f"Completed: {output_path}")
            else:
                print(f"Failed: {year}")
        
        print("Processing complete")
        
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()