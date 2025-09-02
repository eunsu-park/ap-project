"""
SDO JP2 Data Downloader - Python 3.6+
Downloads Solar Dynamics Observatory JP2 image files
"""

import sys
import datetime
from pathlib import Path

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from utils import GenericWebSource, GenericDataDownloader, setup_logging

class SDOSource(GenericWebSource):
    """SDO (Solar Dynamics Observatory) data source"""
    
    def __init__(self):
        url_pattern = "https://jsoc1.stanford.edu/data/aia/images/{year:04d}/{month:02d}/{day:02d}/{wave:d}"
        path_pattern = "{wave}/{year:04d}/{year:04d}{month:02d}{day:02d}"
        super().__init__(url_pattern, path_pattern)

def download_date_range(downloader: GenericDataDownloader, start_date: datetime.date, 
                       end_date: datetime.date, **params) -> None:
    """Download data for a date range"""
    current_date = start_date
    while current_date <= end_date:
        date_params = {
            'year': current_date.year,
            'month': current_date.month,
            'day': current_date.day,
            **params
        }
        downloader.run_with_retry(**date_params)
        current_date += datetime.timedelta(days=1)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SDO JP2 Data Downloader")
    
    # Date options
    parser.add_argument("--year", type=int, help="Year")
    parser.add_argument("--month", type=int, help="Month") 
    parser.add_argument("--day", type=int, help="Day")
    parser.add_argument("--days", type=int, default=1, help="Number of recent days to download")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    
    # SDO specific options
    parser.add_argument("--waves", type=int, nargs="+", default=[193, 211], help="Wavelengths")
    parser.add_argument("--extensions", type=str, nargs="+", default=["jp2"], help="File extensions")
    parser.add_argument("--destination", type=str, default="./data/sdo_jp2/aia", help="Destination directory")
    
    # Download options
    parser.add_argument("--parallel", type=int, default=1, help="Parallel downloads")
    parser.add_argument("--use-processes", action="store_true", help="Use processes instead of threads")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, help="Log file path")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Create SDO data source
        config = SDOSource()
        
        # Create downloader
        downloader = GenericDataDownloader(
            config,
            parallel=args.parallel,
            use_threads=not args.use_processes,
            overwrite=args.overwrite,
            max_retries=args.max_retries
        )
        
        # Determine date range
        if args.start_date or args.end_date:
            # Parse start and end dates
            if args.start_date:
                try:
                    start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
                except ValueError:
                    print("Error: Invalid start date format. Use YYYY-MM-DD")
                    sys.exit(1)
            else:
                start_date = datetime.date(2010, 9, 1)  # SDO mission start date
            
            if args.end_date:
                try:
                    end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
                except ValueError:
                    print("Error: Invalid end date format. Use YYYY-MM-DD")
                    sys.exit(1)
            else:
                end_date = datetime.date.today()
                
            # Validate date range
            if start_date > end_date:
                print("Error: Start date must be before or equal to end date")
                sys.exit(1)
                
        elif args.year and args.month and args.day:
            start_date = datetime.date(args.year, args.month, args.day)
            end_date = start_date
        else:
            today = datetime.date.today()
            start_date = today - datetime.timedelta(days=args.days - 1)
            end_date = today
        
        # Download data - changed loop order: date first, then wavelengths
        current_date = start_date
        while current_date <= end_date:
            print(f"Processing date {current_date}")
            
            for wave in args.waves:
                print(f"  Downloading wave {wave}")
                download_date_range(
                    downloader,
                    current_date,
                    current_date,  # single date
                    wave=wave,
                    extensions=args.extensions,
                    destination_root=args.destination
                )
            
            current_date += datetime.timedelta(days=1)
        
        print("Download process completed")
        
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()