"""
OMNI 데이터 통합 다운로드 스크립트

Downloads and inserts OMNI Low and High Resolution data into database
"""
import argparse
from datetime import datetime
from config import DB_CONFIG, TABLE_CONFIGS
from parsers.lowres_parser import LowResParser
from parsers.highres_parser import HighResParser
from downloaders.base_downloader import BaseDownloader
from egghouse.database import PostgresManager


def download_lowres(start_year, end_year, save_raw=False):
    """
    Low Resolution 데이터 다운로드
    
    Args:
        start_year: 시작 연도
        end_year: 종료 연도
        save_raw: True면 원본 .dat 파일 저장
    
    Returns:
        dict: 다운로드 결과
    """
    print("\n" + "=" * 60)
    print("LOW RESOLUTION DATA DOWNLOAD")
    print("=" * 60)
    
    # 파서 및 다운로더 초기화
    parser = LowResParser()
    downloader = BaseDownloader(
        db_config=DB_CONFIG,
        table_config=TABLE_CONFIGS['low_resolution'],
        parser=parser
    )
    
    # 다운로드
    results = downloader.download_multiple_years(
        start_year=start_year,
        end_year=end_year,
        save_raw_file=save_raw
    )
    
    return results


def download_highres(start_year, end_year, save_raw=False, is_5min=False):
    """
    High Resolution 데이터 다운로드
    
    Args:
        start_year: 시작 연도
        end_year: 종료 연도
        save_raw: True면 원본 .asc 파일 저장
        is_5min: True면 5분 데이터 (GOES flux 포함)
    
    Returns:
        dict: 다운로드 결과
    """
    print("\n" + "=" * 60)
    print("HIGH RESOLUTION DATA DOWNLOAD")
    print(f"Resolution: {'5-minute' if is_5min else '1-minute'}")
    print("=" * 60)
    
    # 파서 및 다운로더 초기화
    parser = HighResParser(is_5min=is_5min)
    downloader = BaseDownloader(
        db_config=DB_CONFIG,
        table_config=TABLE_CONFIGS['high_resolution'],
        parser=parser
    )
    
    # 다운로드
    results = downloader.download_multiple_years(
        start_year=start_year,
        end_year=end_year,
        save_raw_file=save_raw
    )
    
    return results


def check_database_status():
    """데이터베이스 상태 확인"""
    print("\n" + "=" * 60)
    print("DATABASE STATUS")
    print("=" * 60)
    
    try:
        with PostgresManager(**DB_CONFIG) as db:
            for table_name in ['low_resolution', 'high_resolution']:
                print(f"\n{table_name}:")
                
                # 테이블 존재 확인
                tables = db.list_tables(names_only=True)
                if table_name not in tables:
                    print("  ✗ Table does not exist")
                    continue
                
                # 레코드 수
                count = db.count(table_name)
                print(f"  Total records: {count:,}")
                
                if count == 0:
                    print("  (empty table)")
                    continue
                
                # 연도별 통계
                year_stats = db.execute(f"""
                    SELECT 
                        year,
                        COUNT(*) as count,
                        MIN(datetime) as min_date,
                        MAX(datetime) as max_date
                    FROM {table_name}
                    GROUP BY year
                    ORDER BY year
                """, fetch=True)
                
                print(f"  Years: {len(year_stats)}")
                for stat in year_stats[:5]:  # 처음 5개만 출력
                    print(f"    {stat['year']}: {stat['count']:,} records")
                
                if len(year_stats) > 5:
                    print(f"    ... and {len(year_stats) - 5} more years")
                
    except Exception as e:
        print(f"✗ Error: {e}")


def verify_year(table_name, year):
    """특정 연도 데이터 검증"""
    print(f"\n{'='*60}")
    print(f"Verifying {table_name} - Year {year}")
    print(f"{'='*60}")
    
    try:
        with PostgresManager(**DB_CONFIG) as db:
            # 레코드 수
            count = db.count(table_name, where={'year': year})
            print(f"Total records: {count:,}")
            
            if count == 0:
                print("No data found")
                return
            
            # 날짜 범위
            date_range = db.execute(f"""
                SELECT 
                    MIN(datetime) as min_date,
                    MAX(datetime) as max_date
                FROM {table_name}
                WHERE year = {year}
            """, fetch=True)
            
            if date_range:
                dr = date_range[0]
                print(f"Date range: {dr['min_date']} to {dr['max_date']}")
            
            # 샘플 데이터
            print("\nSample data (first 3 records):")
            samples = db.select(
                table_name,
                where={'year': year},
                limit=3,
                order_by='datetime'
            )
            
            for i, sample in enumerate(samples, 1):
                print(f"\n  Record {i}:")
                print(f"    datetime: {sample.get('datetime')}")
                print(f"    year: {sample.get('year')}")
                
                # 테이블별 샘플 필드
                if table_name == 'low_resolution':
                    print(f"    decimal_day: {sample.get('decimal_day')}")
                    print(f"    hour: {sample.get('hour')}")
                    print(f"    bz_gsm_nt: {sample.get('bz_gsm_nt')}")
                    print(f"    plasma_flow_speed_km_s: {sample.get('plasma_flow_speed_km_s')}")
                else:  # high_resolution
                    print(f"    day: {sample.get('day')}")
                    print(f"    hour: {sample.get('hour')}")
                    print(f"    minute: {sample.get('minute')}")
                    print(f"    bz_gsm_nt: {sample.get('bz_gsm_nt')}")
                    print(f"    flow_speed_km_s: {sample.get('flow_speed_km_s')}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='OMNI Data Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download low resolution data for 2020-2024
  python download_all.py --lowres --start 2020 --end 2024
  
  # Download high resolution data for 2023
  python download_all.py --highres --start 2023 --end 2023
  
  # Download both resolutions for 2010-2024
  python download_all.py --lowres --highres --start 2010 --end 2024
  
  # Check database status
  python download_all.py --status
  
  # Verify specific year
  python download_all.py --verify low_resolution 2023
        """
    )
    
    # 다운로드 옵션
    parser.add_argument('--lowres', action='store_true',
                       help='Download low resolution data')
    parser.add_argument('--highres', action='store_true',
                       help='Download high resolution data')
    parser.add_argument('--5min', action='store_true', dest='five_min',
                       help='Use 5-minute high res data (includes GOES flux)')
    
    # 연도 범위
    parser.add_argument('--start', type=int, default=2010,
                       help='Start year (default: 2010)')
    parser.add_argument('--end', type=int, default=datetime.now().year,
                       help='End year (default: current year)')
    
    # 기타 옵션
    parser.add_argument('--save-raw', action='store_true',
                       help='Save raw .dat/.asc files')
    parser.add_argument('--status', action='store_true',
                       help='Show database status')
    parser.add_argument('--verify', nargs=2, metavar=('TABLE', 'YEAR'),
                       help='Verify data for specific table and year')
    
    args = parser.parse_args()
    
    # 데이터베이스 상태 확인
    if args.status:
        check_database_status()
        return
    
    # 특정 연도 검증
    if args.verify:
        table_name, year = args.verify
        verify_year(table_name, int(year))
        return
    
    # 다운로드 옵션이 없으면 도움말 출력
    if not args.lowres and not args.highres:
        parser.print_help()
        return
    
    # 시작
    print("=" * 60)
    print("OMNI DATA DOWNLOADER")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['database']}")
    print(f"Year range: {args.start} - {args.end}")
    print(f"Save raw files: {args.save_raw}")
    print("=" * 60)
    
    all_results = {}
    
    # Low Resolution 다운로드
    if args.lowres:
        results = download_lowres(
            start_year=args.start,
            end_year=args.end,
            save_raw=args.save_raw
        )
        all_results['low_resolution'] = results
    
    # High Resolution 다운로드
    if args.highres:
        results = download_highres(
            start_year=args.start,
            end_year=args.end,
            save_raw=args.save_raw,
            is_5min=args.five_min
        )
        all_results['high_resolution'] = results
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    for table_name, results in all_results.items():
        print(f"\n{table_name}:")
        print(f"  Success: {len(results['success'])} years")
        print(f"  Failed: {len(results['failed'])} years")
    
    # 데이터베이스 상태 확인
    check_database_status()
    
    print("\n" + "=" * 60)
    print("✓ DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()