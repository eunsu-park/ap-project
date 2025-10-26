#!/usr/bin/env python3
"""
Simplified OMNI Data Processor
Extracts time-matched OMNI data for SDO image sequences
파일 위치: processors/process_omni.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from glob import glob
import argparse
from concurrent.futures import ProcessPoolExecutor

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

# from core.utils import TimeSeriesProcessor, DirectoryTimeExtractor, setup_logging

"""
Simplified Data Processing Utilities - Python 3.6+
Essential utilities for data download and processing
"""

import os
import logging
import yaml
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """로깅 설정"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

class ConfigLoader:
    """YAML 설정 파일 로더"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load config: {e}")
            raise

class FixedWidthParser:
    """고정 너비 텍스트 파일 파서 (OMNI 데이터용)"""
    
    def __init__(self, field_specs: List[Dict[str, Any]]):
        """
        필드 명세로 파서 초기화
        
        Args:
            field_specs: 포맷 정보가 포함된 필드 명세 리스트
        """
        self.field_specs = field_specs
        self.field_widths = self._calculate_field_widths()
    
    def _calculate_field_widths(self) -> List[int]:
        """포맷 문자열로부터 필드 너비 계산"""
        widths = []
        for spec in self.field_specs:
            fmt = spec.get('format', 'F8.2')
            if fmt.startswith('I'):
                width = int(fmt[1:])
            elif fmt.startswith('F'):
                width = int(fmt[1:fmt.index('.')] if '.' in fmt else fmt[1:])
            else:
                width = 8  # 기본값
            widths.append(width)
        return widths
    
    def _clean_value(self, value: str, field_spec: Dict[str, Any]) -> Any:
        """필드 값 정리 및 변환 (에러값 처리 포함)"""
        if not value or not value.strip():
            return None
        
        value = value.strip()
        fmt = field_spec.get('format', 'F8.2')
        fill_value = field_spec.get('fill_value')
        
        try:
            if fmt.startswith('I'):  # 정수
                int_val = int(float(value))
                # OMNI 에러값 처리
                if fill_value is not None and int_val == fill_value:
                    return None
                return int_val
            elif fmt.startswith('F'):  # 실수
                float_val = float(value)
                # OMNI 에러값 처리
                if fill_value is not None and abs(float_val - fill_value) < 1e-3:
                    return None
                return float_val
        except (ValueError, TypeError):
            return None
        
        return value
    
    def parse_line(self, line: str) -> Dict[str, Any]:
        """고정 너비 데이터의 한 줄 파싱"""
        result = {}
        pos = 0
        
        for i, spec in enumerate(self.field_specs):
            width = self.field_widths[i]
            
            if pos + width <= len(line):
                field_value = line[pos:pos + width]
                result[spec['name']] = self._clean_value(field_value, spec)
            else:
                result[spec['name']] = None
            
            pos += width
        
        return result

class SimpleDownloader:
    """단순 HTTP 다운로더"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
    
    def download(self, url: str, destination: str) -> bool:
        """파일 다운로드"""
        import requests
        import time
        
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                with open(destination, 'wb') as f:
                    f.write(response.content)
                return True
                
            except Exception as e:
                if attempt == self.max_retries:
                    self.logger.error(f"Failed to download {url}: {e}")
                    return False
                time.sleep(2 ** attempt)
        
        return False

class FilePatternSource:
    """파일 패턴 기반 데이터 소스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['base_url']
        self.file_pattern = config['file_pattern']
        self.output_pattern = config.get('output_pattern', 
                                       self.file_pattern.replace('.dat', '.csv'))

class TextFileDownloader:
    """텍스트 파일 다운로더 (파싱 기능 포함)"""
    
    def __init__(self, source: FilePatternSource, parser: FixedWidthParser = None,
                 timeout: int = 60, max_retries: int = 3, overwrite: bool = False):
        self.source = source
        self.parser = parser
        self.downloader = SimpleDownloader(timeout, max_retries)
        self.overwrite = overwrite
        self.logger = logging.getLogger(__name__)
    
    def download_and_convert(self, output_path: str, **params) -> bool:
        """파일 다운로드 및 CSV 변환"""
        # 소스 파일 정보 가져오기
        url = f"{self.source.base_url}{self.source.file_pattern.format(**params)}"
        temp_file = Path(output_path).with_suffix('.tmp')
        
        # 파일 다운로드
        if not self.downloader.download(url, str(temp_file)):
            return False
        
        # CSV로 변환 (파서가 있는 경우)
        if self.parser:
            try:
                data_rows = []
                with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parsed = self.parser.parse_line(line)
                            data_rows.append(parsed)
                
                if data_rows:
                    df = pd.DataFrame(data_rows)
                    df.to_csv(output_path, index=False)
                    temp_file.unlink()
                    self.logger.info(f"Converted to CSV: {output_path} ({len(df)} rows)")
                    return True
                
            except Exception as e:
                self.logger.error(f"Failed to convert to CSV: {e}")
        
        # 파서가 없거나 변환 실패시 파일 이동
        Path(temp_file).rename(output_path)
        return True

class TimeSeriesProcessor:
    """시계열 데이터 처리기 (데이터셋 매칭용)"""
    
    def __init__(self, data_dir: str, columns: Optional[List[str]] = None):
        self.data_dir = data_dir
        self.columns = columns  # None = 모든 컬럼
        self.logger = logging.getLogger(__name__)
        self._data_cache = {}
    
    def _load_year_data(self, year: int) -> Optional[pd.DataFrame]:
        """특정 연도의 OMNI 데이터 로드"""
        if year in self._data_cache:
            return self._data_cache[year]
        
        csv_path = Path(self.data_dir) / f"omni2_{year}.csv"
        
        if not csv_path.exists():
            self.logger.warning(f"Data file not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            self._data_cache[year] = df
            return df
        except Exception as e:
            self.logger.error(f"Error loading {csv_path}: {e}")
            return None
    
    def find_matching_data(self, target_datetime: datetime, 
                          time_offset_hours: int = -1) -> Optional[Dict[str, Any]]:
        """시간 오프셋을 적용한 매칭 데이터 찾기"""
        # 시간 오프셋 적용
        search_datetime = target_datetime + timedelta(hours=time_offset_hours)
        year = search_datetime.year
        
        # 연도 데이터 로드
        df = self._load_year_data(year)
        if df is None:
            return None
        
        # Year, Day, Hour 컬럼으로 datetime 생성
        try:
            df['search_datetime'] = pd.to_datetime(df['Year'], format='%Y') + \
                                  pd.to_timedelta(df['Day'] - 1, unit='D') + \
                                  pd.to_timedelta(df['Hour'], unit='h')
        except KeyError as e:
            self.logger.error(f"Required time columns missing: {e}")
            return None
        
        # 정확한 매치 찾기
        matching_rows = df[df['search_datetime'] == search_datetime]
        
        if len(matching_rows) == 0:
            return None
        
        # 컬럼 추출
        result = {}
        row = matching_rows.iloc[0]
        
        # 특정 컬럼이 지정되지 않으면 모든 컬럼 사용 (시간 관련 제외)
        if self.columns is None:
            exclude_cols = {'search_datetime', 'Year', 'Day', 'Hour'}
            data_columns = [col for col in df.columns if col not in exclude_cols]
        else:
            data_columns = self.columns
        
        for col in data_columns:
            if col in df.columns:
                value = row[col]
                result[col] = pd.to_numeric(value, errors='coerce')
            else:
                result[col] = None
        
        return result
    
    def process_time_range(self, start_datetime: datetime, end_datetime: datetime, 
                          interval_hours: int = 3, time_offset_hours: int = -1) -> List[Dict[str, Any]]:
        """시간 범위 처리 및 매칭 데이터 추출"""
        results = []
        current_time = start_datetime
        data_columns = None
        
        while current_time <= end_datetime:
            data = self.find_matching_data(current_time, time_offset_hours)
            
            result_row = {
                'base_time': current_time.strftime('%Y-%m-%d %H')
            }
            
            if data:
                result_row.update(data)
                if data_columns is None:
                    data_columns = list(data.keys())
            else:
                # 누락 데이터에 대해 None으로 채우기
                if data_columns is not None:
                    for col in data_columns:
                        result_row[col] = None
            
            results.append(result_row)
            current_time += timedelta(hours=interval_hours)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> bool:
        """결과를 CSV 파일로 저장"""
        if not results:
            return False
        
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return False

class DirectoryTimeExtractor:
    """디렉토리 이름에서 datetime 추출"""
    
    @staticmethod
    def extract_from_pattern(dirname: str, pattern: str = '%Y%m%d%H') -> Optional[datetime]:
        """패턴을 사용해 디렉토리 이름에서 datetime 추출"""
        try:
            # 디렉토리 이름에서 시간 부분 추출 ('-' 앞부분)
            # time_str = dirname.split('-')[0]
            time_str = dirname
            return datetime.strptime(time_str, pattern)
        except (ValueError, IndexError):
            return None

def process_single_dataset(dataset_dir, omni_dir, interval_hours, days_before, days_after, time_offset):
    """단일 데이터셋 처리 (병렬 처리용 워커 함수)"""
    # 이 워커에 대한 프로세서 생성 (특정 컬럼 지정 안함 = 모든 컬럼)
    processor = TimeSeriesProcessor(omni_dir, columns=None)
    
    # 디렉토리 이름에서 기준 시간 추출
    dir_name = os.path.basename(dataset_dir)
    reference_time = DirectoryTimeExtractor.extract_from_pattern(dir_name)
    
    if reference_time is None:
        print(f"Cannot extract time from directory: {dir_name}")
        return False
    
    # 시작/종료 시간 계산
    # reference_time을 기준으로 이전 days_before일, 이후 days_after일
    start_time = reference_time - timedelta(days=days_before)
    end_time = reference_time + timedelta(days=days_after)
    
    # 출력 파일 경로
    output_path = os.path.join(dataset_dir, "hourly_data.csv")
    
    # 이미 존재하면 스킵
    if os.path.exists(output_path):
        print(f"Already processed: {dir_name}")
        return True
    
    print(f"Processing {dir_name}: {start_time} to {end_time}")
    
    # 시간 범위 처리
    results = processor.process_time_range(start_time, end_time, interval_hours, time_offset)
    
    # 성공적인 추출 개수 계산
    success_count = sum(1 for r in results if len(r) > 2)  # base_time, target_time보다 많은 데이터
    
    # 결과 저장
    if processor.save_results(results, output_path):
        print(f"Saved {len(results)} records ({success_count} with data) to {output_path}")
        return True
    else:
        print(f"Failed to save results for {dir_name}")
        return False

def process_dataset_sequential(dataset_dir, processor, interval_hours, days_before, days_after, time_offset_hours):
    """단일 데이터셋 처리 (순차 처리용)"""
    # 디렉토리 이름에서 기준 시간 추출
    dir_name = os.path.basename(dataset_dir)
    reference_time = DirectoryTimeExtractor.extract_from_pattern(dir_name)
    
    if reference_time is None:
        print(f"Cannot extract time from directory: {dir_name}")
        return False
    
    # 시작/종료 시간 계산
    # reference_time을 기준으로 이전 days_before일, 이후 days_after일
    start_time = reference_time - timedelta(days=days_before)
    end_time = reference_time + timedelta(days=days_after)
    
    # 출력 파일 경로
    output_path = os.path.join(dataset_dir, "hourly_data.csv")
    
    # 이미 존재하면 스킵
    if os.path.exists(output_path):
        print(f"Already processed: {dir_name}")
        return True
    
    print(f"Processing {dir_name}: {start_time} to {end_time}")
    
    # 시간 범위 처리
    results = processor.process_time_range(start_time, end_time, interval_hours, time_offset_hours)
    
    # 성공적인 추출 개수 계산 (시간 컬럼보다 많은 데이터가 있는 행)
    success_count = sum(1 for r in results if len(r) > 2)
    
    # 결과 저장
    if processor.save_results(results, output_path):
        print(f"Saved {len(results)} records ({success_count} with data) to {output_path}")
        return True
    else:
        print(f"Failed to save results for {dir_name}")
        return False

def main():
    parser = argparse.ArgumentParser(description="OMNI Data Processor")
    
    # 입출력 경로
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Directory containing processed SDO datasets')
    parser.add_argument('--omni-dir', type=str, required=True,
                       help='Directory containing OMNI CSV files')
    
    # OMNI 컬럼 추출 (선택사항 - 지정하지 않으면 모든 컬럼 추출)
    parser.add_argument('--columns', type=str, nargs='+', default=None,
                       help='Specific OMNI columns to extract (default: all columns)')
    
    # 처리 파라미터
    parser.add_argument('--interval', type=int, default=3,
                       help='Time interval in hours')
    parser.add_argument('--days-before', type=int, default=10,
                       help='Number of days before reference time')
    parser.add_argument('--days-after', type=int, default=5,
                       help='Number of days after reference time')
    parser.add_argument('--time-offset', type=int, default=-1,
                       help='Time offset in hours (negative for past data)')
    
    # 처리 옵션
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum parallel workers')
    
    # 로깅
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # 디렉토리 검증
    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset directory not found: {args.dataset_dir}")
        return 1
    
    if not os.path.isdir(args.omni_dir):
        print(f"OMNI directory not found: {args.omni_dir}")
        return 1
    
    # 데이터셋 디렉토리 찾기
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
    print(f"Sequence composition: {args.days_before} days before + {args.days_after} days after = {args.days_before + args.days_after} total days")
    
    # 데이터셋 처리
    if args.parallel:
        # 병렬 처리
        from multiprocessing import cpu_count
        
        max_workers = args.max_workers or min(cpu_count(), len(dataset_dirs))
        print(f"Using {max_workers} parallel workers")
        
        # 워커 함수 인자 준비
        worker_args = [(dataset_dir, args.omni_dir, args.interval, 
                       args.days_before, args.days_after, args.time_offset) for dataset_dir in dataset_dirs]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_dataset, *zip(*worker_args)))
        
        successful = sum(results)
        
    else:
        # 순차 처리
        print("Processing sequentially")
        processor = TimeSeriesProcessor(args.omni_dir, args.columns)
        successful = 0
        
        for i, dataset_dir in enumerate(dataset_dirs, 1):
            print(f"[{i}/{len(dataset_dirs)}]", end=" ")
            if process_dataset_sequential(dataset_dir, processor, args.interval, 
                                        args.days_before, args.days_after, args.time_offset):
                successful += 1
    
    # 최종 결과
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful}/{len(dataset_dirs)} datasets")
    
    if successful < len(dataset_dirs):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

# #!/usr/bin/env python3
# """
# Simplified OMNI Data Processor
# Extracts time-matched OMNI data for SDO image sequences
# 파일 위치: processors/process_omni.py
# """

# import os
# import sys
# from pathlib import Path
# from datetime import datetime, timedelta
# from glob import glob
# import argparse
# from concurrent.futures import ProcessPoolExecutor

# # Check Python version
# if sys.version_info < (3, 6):
#     raise RuntimeError("This script requires Python 3.6 or higher")

# from core.utils import TimeSeriesProcessor, DirectoryTimeExtractor, setup_logging

# def process_single_dataset(dataset_dir, omni_dir, interval_hours, sequence_days, time_offset):
#     """단일 데이터셋 처리 (병렬 처리용 워커 함수)"""
#     # 이 워커에 대한 프로세서 생성 (특정 컬럼 지정 안함 = 모든 컬럼)
#     processor = TimeSeriesProcessor(omni_dir, columns=None)
    
#     # 디렉토리 이름에서 시작 시간 추출
#     dir_name = os.path.basename(dataset_dir)
#     start_time = DirectoryTimeExtractor.extract_from_pattern(dir_name)
    
#     if start_time is None:
#         print(f"Cannot extract time from directory: {dir_name}")
#         return False
    
#     # 종료 시간 계산
#     end_time = start_time + timedelta(days=sequence_days)
    
#     # 출력 파일 경로
#     output_path = os.path.join(dataset_dir, "hourly_data.csv")
    
#     # 이미 존재하면 스킵
#     if os.path.exists(output_path):
#         print(f"Already processed: {dir_name}")
#         return True
    
#     print(f"Processing {dir_name}: {start_time} to {end_time}")
    
#     # 시간 범위 처리
#     results = processor.process_time_range(start_time, end_time, interval_hours, time_offset)
    
#     # 성공적인 추출 개수 계산
#     success_count = sum(1 for r in results if len(r) > 2)  # base_time, target_time보다 많은 데이터
    
#     # 결과 저장
#     if processor.save_results(results, output_path):
#         print(f"Saved {len(results)} records ({success_count} with data) to {output_path}")
#         return True
#     else:
#         print(f"Failed to save results for {dir_name}")
#         return False

# def process_dataset_sequential(dataset_dir, processor, interval_hours, sequence_days, time_offset_hours):
#     """단일 데이터셋 처리 (순차 처리용)"""
#     # 디렉토리 이름에서 시작 시간 추출
#     dir_name = os.path.basename(dataset_dir)
#     start_time = DirectoryTimeExtractor.extract_from_pattern(dir_name)
    
#     if start_time is None:
#         print(f"Cannot extract time from directory: {dir_name}")
#         return False
    
#     # 종료 시간 계산
#     end_time = start_time + timedelta(days=sequence_days)
    
#     # 출력 파일 경로
#     output_path = os.path.join(dataset_dir, "hourly_data.csv")
    
#     # 이미 존재하면 스킵
#     if os.path.exists(output_path):
#         print(f"Already processed: {dir_name}")
#         return True
    
#     print(f"Processing {dir_name}: {start_time} to {end_time}")
    
#     # 시간 범위 처리
#     results = processor.process_time_range(start_time, end_time, interval_hours, time_offset_hours)
    
#     # 성공적인 추출 개수 계산 (시간 컬럼보다 많은 데이터가 있는 행)
#     success_count = sum(1 for r in results if len(r) > 2)
    
#     # 결과 저장
#     if processor.save_results(results, output_path):
#         print(f"Saved {len(results)} records ({success_count} with data) to {output_path}")
#         return True
#     else:
#         print(f"Failed to save results for {dir_name}")
#         return False

# def main():
#     parser = argparse.ArgumentParser(description="OMNI Data Processor")
    
#     # 입출력 경로
#     parser.add_argument('--dataset-dir', type=str, required=True,
#                        help='Directory containing processed SDO datasets')
#     parser.add_argument('--omni-dir', type=str, required=True,
#                        help='Directory containing OMNI CSV files')
    
#     # OMNI 컬럼 추출 (선택사항 - 지정하지 않으면 모든 컬럼 추출)
#     parser.add_argument('--columns', type=str, nargs='+', default=None,
#                        help='Specific OMNI columns to extract (default: all columns)')
    
#     # 처리 파라미터
#     parser.add_argument('--interval', type=int, default=3,
#                        help='Time interval in hours')
#     parser.add_argument('--sequence-days', type=int, default=8,
#                        help='Sequence length in days')
#     parser.add_argument('--time-offset', type=int, default=-1,
#                        help='Time offset in hours (negative for past data)')
    
#     # 처리 옵션
#     parser.add_argument('--parallel', action='store_true',
#                        help='Enable parallel processing')
#     parser.add_argument('--max-workers', type=int, default=None,
#                        help='Maximum parallel workers')
    
#     # 로깅
#     parser.add_argument('--log-level', type=str, default='INFO',
#                        help='Logging level')
    
#     args = parser.parse_args()
    
#     setup_logging(args.log_level)
    
#     # 디렉토리 검증
#     if not os.path.isdir(args.dataset_dir):
#         print(f"Dataset directory not found: {args.dataset_dir}")
#         return 1
    
#     if not os.path.isdir(args.omni_dir):
#         print(f"OMNI directory not found: {args.omni_dir}")
#         return 1
    
#     # 데이터셋 디렉토리 찾기
#     dataset_dirs = sorted([d for d in glob(f"{args.dataset_dir}/*") 
#                           if os.path.isdir(d) and os.path.basename(d)[0].isdigit()])
    
#     if not dataset_dirs:
#         print(f"No valid dataset directories found in {args.dataset_dir}")
#         return 1
    
#     print(f"Found {len(dataset_dirs)} dataset directories")
#     if args.columns:
#         print(f"OMNI columns: {args.columns}")
#     else:
#         print("Extracting all OMNI columns")
#     print(f"Time offset: {args.time_offset} hours")
#     print(f"Processing interval: {args.interval} hours")
#     print(f"Sequence duration: {args.sequence_days} days")
    
#     # 데이터셋 처리
#     if args.parallel:
#         # 병렬 처리
#         from multiprocessing import cpu_count
        
#         max_workers = args.max_workers or min(cpu_count(), len(dataset_dirs))
#         print(f"Using {max_workers} parallel workers")
        
#         # 워커 함수 인자 준비
#         worker_args = [(dataset_dir, args.omni_dir, args.interval, 
#                        args.sequence_days, args.time_offset) for dataset_dir in dataset_dirs]
        
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             results = list(executor.map(process_single_dataset, *zip(*worker_args)))
        
#         successful = sum(results)
        
#     else:
#         # 순차 처리
#         print("Processing sequentially")
#         processor = TimeSeriesProcessor(args.omni_dir, args.columns)
#         successful = 0
        
#         for i, dataset_dir in enumerate(dataset_dirs, 1):
#             print(f"[{i}/{len(dataset_dirs)}]", end=" ")
#             if process_dataset_sequential(dataset_dir, processor, args.interval, 
#                                         args.sequence_days, args.time_offset):
#                 successful += 1
    
#     # 최종 결과
#     print(f"\nProcessing complete:")
#     print(f"Successfully processed: {successful}/{len(dataset_dirs)} datasets")
    
#     if successful < len(dataset_dirs):
#         return 1
    
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())