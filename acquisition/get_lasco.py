"""
Simplified LASCO Data Downloader - Python 3.6+
Downloads SOHO/LASCO coronagraph data
"""

import sys
import datetime
from pathlib import Path

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from utils import GenericWebSource, GenericDataDownloader, setup_logging

class LASCOSource(GenericWebSource):
    """LASCO (SOHO) data source"""
    
    def __init__(self):
        super().__init__("", "")
        self.url_root = "https://lasco-www.nrl.navy.mil/lz/level_05"
        self.start_date = datetime.datetime(1995, 12, 8)  # LASCO mission start
    
    def get_url(self, **kwargs) -> str:
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        camera = kwargs.get('camera', 'c2')
        
        year_short = int(str(year)[2:])  # Convert 2024 to 24
        return f"{self.url_root}/{year_short:02d}{month:02d}{day:02d}/{camera}"
    
    def get_save_path(self, destination_root: str, **kwargs) -> str:
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        camera = kwargs.get('camera', 'c2')
        
        return str(Path(destination_root) / camera / f"{year:04d}" / f"{year:04d}{month:02d}{day:02d}")

def download_date_range(downloader: GenericDataDownloader, start_date: datetime.date, 
                       end_date: datetime.date, **params) -> None:
    """Download data for a date range"""
    current_date = start_date
    while current_date <= end_date:
        for camera in params.get('cameras', ['c2']):
            print(f"Processing {current_date} camera {camera}")
            downloader.run_with_retry(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
                camera=camera,
                **params
            )
        current_date += datetime.timedelta(days=1)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LASCO Data Downloader")
    
    # Date options
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Number of recent days to download")
    
    # LASCO specific options
    parser.add_argument("--cameras", type=str, nargs="+", default=["c2"],
                       choices=["c1", "c2", "c3", "c4"], help="LASCO cameras")
    parser.add_argument("--extensions", type=str, nargs="+", default=["fts"],
                       help="File extensions")
    parser.add_argument("--destination", type=str, default="./data/lasco", help="Destination directory")
    
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
        config = LASCOSource()
        
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
            print(f"Warning: Start date {start_date} is before LASCO mission start {mission_start}")
            start_date = mission_start
        
        print(f"LASCO Data Downloader")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Cameras: {args.cameras}")
        
        # Download data
        download_date_range(
            downloader,
            start_date,
            end_date,
            cameras=args.cameras,
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