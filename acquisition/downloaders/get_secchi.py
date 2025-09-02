"""
SECCHI Data Downloader - Python 3.6+
Downloads STEREO/SECCHI coronagraph and heliospheric imager data
파일 위치: downloaders/get_secchi.py
"""

import sys
import datetime
from pathlib import Path

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from core.base_downloader import BaseDataSource, BaseDownloader, BaseCLI, download_date_range

class SECCHISource(BaseDataSource):
    """SECCHI (STEREO) data source"""
    
    def __init__(self):
        # STEREO 미션 시작일 (2006년 10월 27일)
        super().__init__(datetime.datetime(2006, 10, 27))
    
    def get_url(self, **kwargs) -> str:
        data_type = kwargs.get('data_type', 'science')
        spacecraft = kwargs.get('spacecraft', 'ahead')
        category = kwargs.get('category', 'img')
        instrument = kwargs.get('instrument', 'cor2')
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        
        if data_type == "science":
            base_url = f"https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/{spacecraft[0]}/{category}"
        else:  # beacon
            base_url = f"https://stereo-ssc.nascom.nasa.gov/data/beacon/{spacecraft}/secchi/{category}"
        
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

def download_secchi_date_range(downloader: BaseDownloader, start_date: datetime.date, 
                              end_date: datetime.date, **params) -> None:
    """SECCHI 날짜 범위 다운로드 (다중 파라미터 처리)"""
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
                            destination_root=params.get('destination_root', './data'),
                            extensions=params.get('extensions', ['fts'])
                        )
        current_date += datetime.timedelta(days=1)

def main():
    # CLI 설정
    cli = BaseCLI("SECCHI Data Downloader", "SECCHI Data Downloader")
    parser = cli.setup_common_args()
    
    # SECCHI 전용 옵션 추가
    parser.add_argument("--data-types", type=str, nargs="+", default=["science"],
                       choices=["science", "beacon"], help="Data types")
    parser.add_argument("--spacecrafts", type=str, nargs="+", default=["ahead"],
                       choices=["ahead", "behind"], help="Spacecrafts")
    parser.add_argument("--categories", type=str, nargs="+", default=["img"],
                       choices=["img", "cal", "seq"], help="Categories")
    parser.add_argument("--instruments", type=str, nargs="+", default=["cor2"],
                       choices=["cor1", "cor2", "euvi", "hi_1", "hi_2"], help="Instruments")
    
    # 기본값 변경
    parser.set_defaults(
        extensions=["fts"],
        destination="./data/secchi"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    cli.setup_logging(args.log_level, args.log_file)
    
    try:
        # SECCHI 데이터 소스 및 다운로더 생성
        config = SECCHISource()
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
        
        print(f"SECCHI Data Downloader")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Data types: {args.data_types}")
        print(f"Spacecrafts: {args.spacecrafts}")
        print(f"Categories: {args.categories}")
        print(f"Instruments: {args.instruments}")
        
        # 데이터 다운로드
        download_secchi_date_range(
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