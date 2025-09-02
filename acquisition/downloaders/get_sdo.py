"""
SDO JP2 Data Downloader - Python 3.6+
Downloads Solar Dynamics Observatory JP2 image files
파일 위치: downloaders/get_sdo.py
"""

import sys
import datetime
from pathlib import Path

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from core.base_downloader import BaseDataSource, BaseDownloader, BaseCLI, download_date_range

class SDOSource(BaseDataSource):
    """SDO (Solar Dynamics Observatory) data source"""
    
    def __init__(self):
        # SDO 미션 시작일 (2010년 9월 1일)
        super().__init__(datetime.datetime(2010, 9, 1))
    
    def get_url(self, **kwargs) -> str:
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        wave = kwargs.get('wave', 193)
        
        return f"https://jsoc1.stanford.edu/data/aia/images/{year:04d}/{month:02d}/{day:02d}/{wave:d}"
    
    def get_save_path(self, destination_root: str, **kwargs) -> str:
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        wave = kwargs.get('wave', 193)
        
        return str(Path(destination_root) / f"{wave}" / f"{year:04d}" / f"{year:04d}{month:02d}{day:02d}")

def download_sdo_date_range(downloader: BaseDownloader, start_date: datetime.date, 
                           end_date: datetime.date, **params) -> None:
    """SDO 날짜 범위 다운로드 (파장별 처리)"""
    current_date = start_date
    while current_date <= end_date:
        print(f"Processing date {current_date}")
        
        for wave in params.get('waves', [193, 211]):
            print(f"  Downloading wave {wave}")
            downloader.run_with_retry(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
                wave=wave,
                destination_root=params.get('destination_root', './data'),
                extensions=params.get('extensions', ['jp2'])
            )
        
        current_date += datetime.timedelta(days=1)

def main():
    # CLI 설정
    cli = BaseCLI("SDO JP2 Data Downloader", "SDO JP2 Data Downloader")
    parser = cli.setup_common_args()
    
    # SDO 전용 옵션 추가
    parser.add_argument("--year", type=int, help="Year")
    parser.add_argument("--month", type=int, help="Month") 
    parser.add_argument("--day", type=int, help="Day")
    parser.add_argument("--waves", type=int, nargs="+", default=[193, 211], help="Wavelengths")
    
    # 기본값 변경
    parser.set_defaults(
        extensions=["jp2"],
        destination="./data/sdo_jp2/aia"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    cli.setup_logging(args.log_level, args.log_file)
    
    try:
        # SDO 데이터 소스 및 다운로더 생성
        config = SDOSource()
        downloader = BaseDownloader(
            config,
            parallel=args.parallel,
            overwrite=args.overwrite,
            max_retries=args.max_retries
        )
        
        # 날짜 범위 결정
        if args.year and args.month and args.day:
            start_date = datetime.date(args.year, args.month, args.day)
            end_date = start_date
        else:
            start_date, end_date = cli.parse_date_range(args)
        
        # 미션 시작 날짜 검증
        start_date = cli.validate_mission_dates(start_date, config.mission_start_date)
        
        print(f"SDO Data Downloader")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Waves: {args.waves}")
        
        # 데이터 다운로드
        download_sdo_date_range(
            downloader,
            start_date,
            end_date,
            waves=args.waves,
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