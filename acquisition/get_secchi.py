"""
Simplified SECCHI Data Downloader - Python 3.6+
Downloads STEREO/SECCHI coronagraph and heliospheric imager data
"""

import sys
import datetime
from pathlib import Path

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from utils import GenericWebSource, GenericDataDownloader, setup_logging

class SECCHISource(GenericWebSource):
    """SECCHI (STEREO) data source"""
    
    def __init__(self):
        # Base URL pattern will be set dynamically
        super().__init__("", "")
        self.url_root = "https://stereo-ssc.nascom.nasa.gov/data"
        self.start_date = datetime.datetime(2006, 10, 27)  # STEREO launch
    
    def get_url(self, **kwargs) -> str:
        data_type = kwargs.get('data_type', 'science')
        spacecraft = kwargs.get('spacecraft', 'ahead')
        category = kwargs.get('category', 'img')
        instrument = kwargs.get('instrument', 'cor2')
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        
        if data_type == "science":
            base_url = f"{self.url_root}/ins_data/secchi/L0/{spacecraft[0]}/{category}"
        else:  # beacon
            base_url = f"{self.url_root}/beacon/{spacecraft}/secchi/{category}"
        
        return f"{base_url}/{instrument}/{year:04d}{month:02d}{day:02d}"
    
    def get_save_path(self, destination_root: str, **kwargs) -> str:
        data_type = kwargs.get('data_type', 'science')
        spacecraft = kwargs.get('spacecraft', 'ahead')
        category = kwargs.get('category', 'img')
        instrument = kwargs.get('instrument', 'cor2')
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        
        return str(Path(destination_root) / data_type / spacecraft / category / 
                  instrument / f"{year:04d}" / f"{year:04d}{month:02d}{day:02d}")

def download_date_range(downloader: GenericDataDownloader, start_date: datetime.date, 
                       end_date: datetime.date, **params) -> None:
    """Download data for a date range"""
    current_date = start_date
    while current_date <= end_date:
        for data_type in params.get('data_types', ['science']):
            for spacecraft in params.get('spacecrafts', ['ahead']):
                for category in params.get('categories', ['img']):
                    for instrument in params.get('instruments', ['cor2']):
                        print(f"Processing {current_date} {data_type}/{spacecraft}/{category}/{instrument}")
                        downloader.run_with_retry(
                            year=current_date.year,
                            month=current_date.month,
                            day=current_date.day,
                            data_type=data_type,
                            spacecraft=spacecraft,
                            category=category,
                            instrument=instrument,
                            **params
                        )
        current_date += datetime.timedelta(days=1)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SECCHI Data Downloader")
    
    # Date options
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Number of recent days to download")
    
    # SECCHI specific options
    parser.add_argument("--data-types", type=str, nargs="+", default=["science"],
                       choices=["science", "beacon"], help="Data types")
    parser.add_argument("--spacecrafts", type=str, nargs="+", default=["ahead"],
                       choices=["ahead", "behind"], help="Spacecrafts")
    parser.add_argument("--categories", type=str, nargs="+", default=["img"],
                       choices=["img", "cal", "seq"], help="Categories")
    parser.add_argument("--instruments", type=str, nargs="+", default=["cor2"],
                       choices=["cor1", "cor2", "euvi", "hi_1", "hi_2"], help="Instruments")
    parser.add_argument("--extensions", type=str, nargs="+", default=["fts"],
                       help="File extensions")
    parser.add_argument("--destination", type=str, default="./data/secchi", help="Destination directory")
    
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
        # Create data source
        config = SECCHISource()
        
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
            if args.start_date:
                start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
            else:
                start_date = datetime.date.today() - datetime.timedelta(days=args.days - 1)
            
            if args.end_date:
                end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
            else:
                end_date = datetime.date.today()
        else:
            today = datetime.date.today()
            start_date = today - datetime.timedelta(days=args.days - 1)
            end_date = today
        
        # Check mission start date
        mission_start = config.start_date.date()
        if start_date < mission_start:
            print(f"Warning: Start date {start_date} is before STEREO mission start {mission_start}")
            start_date = mission_start
        
        print(f"SECCHI Data Downloader")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Data types: {args.data_types}")
        print(f"Spacecrafts: {args.spacecrafts}")
        print(f"Categories: {args.categories}")
        print(f"Instruments: {args.instruments}")
        
        # Download data
        download_date_range(
            downloader,
            start_date,
            end_date,
            data_types=args.data_types,
            spacecrafts=args.spacecrafts,
            categories=args.categories,
            instruments=args.instruments,
            extensions=args.extensions,
            destination_root=args.destination
        )
        
        print("Download process completed")
        
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()