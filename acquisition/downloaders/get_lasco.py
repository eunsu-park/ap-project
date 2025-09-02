"""
LASCO Data Downloader - Python 3.6+
Downloads SOHO/LASCO coronagraph data
파일 위치: downloaders/get_lasco.py
"""

import sys
import datetime
from pathlib import Path

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from core.base_downloader import BaseDataSource, BaseDownloader, BaseCLI, download_date_range

class LASCOSource(BaseDataSource):
    """LASCO (SOHO) data source"""
    
    def __init__(self):
        # LASCO 미션 시작일 (1995년 12월 8일)
        super().__init__(datetime.datetime(1995, 12, 8))
    
    def get_url(self, **kwargs) -> str:
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        camera = kwargs.get('camera', 'c2')
        
        year_short = int(str(year)[2:])  # Convert 2024 to 24
        return f"https://lasco-www.nrl.navy.mil/lz/level_05/{year_short:02d}{month:02d}{day:02d}/{camera}"
    
    def get_save_path(self, destination_root: str, **kwargs) -> str:
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        camera = kwargs.get('camera', 'c2')
        
        return str(Path(destination_root) / camera / f"{year:04d}" / f"{year:04d}{month:02d}{day:02d}")

def download_lasco_date_range(downloader: BaseDownloader, start_date: datetime.date, 
                             end_date: datetime.date, **params) -> None:
    """LASCO 날짜 범위 다운로드 (카메라별 처리)"""
    current_date = start_date
    while current_date <= end_date:
        for camera in params.get('cameras', ['c2']):
            print(f"Processing {current_date} camera {camera}")
            downloader.run_with_retry(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
                camera=camera,
                destination_root=params.get('destination_root', './data'),
                extensions=params.get('extensions', ['fts'])
            )
        current_date += datetime.timedelta(days=1)

def main():
    # CLI 설정
    cli = BaseCLI("LASCO Data Downloader", "LASCO Data Downloader")
    parser = cli.setup_common_args()
    
    # LASCO 전용 옵션 추가
    parser.add_argument("--cameras", type=str, nargs="+", default=["c2"],
                       choices=["c1", "c2", "c3", "c4"], help="LASCO cameras")
    
    # 기본값 변경
    parser.set_defaults(
        extensions=["fts"],
        destination="./data/lasco"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    cli.setup_logging(args.log_level, args.log_file)
    
    try:
        # LASCO 데이터 소스 및 다운로더 생성
        config = LASCOSource()
        downloader = BaseDownloader(
            config,
            parallel=args.parallel,
            overwrite=args.overwrite,
            max_retries=args.max_retries
        )
        
        # 날짜 범위 결정
        start_date, end_date = cli.parse_date_range(args)
        
        # 미션 시작 날짜 검증
        start_date = cli.validate_mission_dates(start_date, config.mission_start_date)
        
        print(f"LASCO Data Downloader")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Cameras: {args.cameras}")
        
        # 데이터 다운로드
        download_lasco_date_range(
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