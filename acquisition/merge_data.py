"""
개선된 SDO + OMNI 데이터 병합 스크립트

주요 개선사항:
- DB 연결 안정성 (Context Manager)
- 완벽한 에러 핸들링 및 로깅
- 병렬 처리 (멀티프로세싱)
- 중복 작업 방지 (Resume 기능)
- 재시도 메커니즘
- 진행 상황 표시
"""
import os
import sys
import datetime
import logging
import json
from time import sleep
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from egghouse.database import PostgresManager

# ============================================================================
# 설정
# ============================================================================

SAVE_ROOT = "/Users/eunsupark/projects/ap/data"

SDO_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "sdo",
    "user": "eunsupark",
    "password": "eunsupark",
    "log_queries": False
}

OMNI_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "omni",
    "user": "eunsupark",
    "password": "eunsupark",
    "log_queries": False
}

# 시간 설정
OMNI_CADENCE = 3  # hours
SDO_CADENCE = 6   # hours
DATASET_CADENCE = 3  # hours

# 시퀀스 길이
BEFORE_DAY = 10
AFTER_DAY = 5

SDO_NUM_SEQUENCE = (BEFORE_DAY + AFTER_DAY) * (24 // SDO_CADENCE)
OMNI_NUM_SEQUENCE = (BEFORE_DAY + AFTER_DAY) * (24 // OMNI_CADENCE)

SDO_TABLE_NAMES = ["aia_193", "aia_211", "hmi_magnetogram"]

# 병렬 처리 설정
NUM_WORKERS = max(1, cpu_count() - 1)  # 1개 코어 남겨두기

# 재시도 설정
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# 진행 상황 파일
PROGRESS_FILE = "merge_progress.json"

# ============================================================================
# 로깅 설정
# ============================================================================

def setup_logging(log_file='merge_data.log'):
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# 유틸리티 함수
# ============================================================================

def is_already_processed(target_date: datetime.datetime) -> bool:
    """
    이미 처리된 날짜인지 확인
    
    Args:
        target_date: 확인할 날짜
    
    Returns:
        bool: 이미 처리되었으면 True
    """
    save_dir = f"{SAVE_ROOT}/{target_date:%Y%m%d%H}"
    
    # 모든 필수 파일 존재 확인
    required_files = [
        f"{save_dir}/aia_193.npy",
        f"{save_dir}/aia_211.npy",
        f"{save_dir}/hmi_magnetogram.npy",
        f"{save_dir}/omni.csv"
    ]
    
    return all(os.path.exists(f) for f in required_files)


def save_progress(processed_dates: List[datetime.datetime]):
    """
    처리 완료된 날짜 저장
    
    Args:
        processed_dates: 처리 완료된 날짜 리스트
    """
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                'processed': [d.isoformat() for d in processed_dates],
                'total': len(processed_dates),
                'last_updated': datetime.datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save progress: {e}")


def load_progress() -> set:
    """
    진행 상황 로드
    
    Returns:
        set: 처리 완료된 날짜 세트
    """
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
                processed = set(
                    datetime.datetime.fromisoformat(d) 
                    for d in data['processed']
                )
                logger.info(f"Loaded progress: {len(processed)} dates already processed")
                return processed
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
    
    return set()


def retry_on_failure(func, max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY):
    """
    실패 시 재시도
    
    Args:
        func: 실행할 함수
        max_retries: 최대 재시도 횟수
        delay: 재시도 간 대기 시간 (초)
    
    Returns:
        함수 실행 결과
    
    Raises:
        Exception: 모든 재시도 실패 시
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                logger.info(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise


# ============================================================================
# DB 접근 함수 (Context Manager 사용)
# ============================================================================

def get_sdo_data(table_name: str, date: datetime.datetime) -> List[Dict]:
    """
    SDO 데이터 조회 (안전한 DB 연결)
    
    Args:
        table_name: 테이블 이름
        date: 조회할 날짜
    
    Returns:
        List[Dict]: 조회 결과
    """
    with PostgresManager(**SDO_DB_CONFIG) as db:
        return db.select(table_name, where={"date_rounded": date})


def get_omni_data(date: datetime.datetime) -> List[Dict]:
    """
    OMNI 데이터 조회 (안전한 DB 연결)
    
    Args:
        date: 조회할 날짜
    
    Returns:
        List[Dict]: 조회 결과
    """
    with PostgresManager(**OMNI_DB_CONFIG) as db:
        return db.select("low_resolution", where={"datetime": date})


# ============================================================================
# 이미지 처리 함수
# ============================================================================

def read_png(file_path: str) -> np.ndarray:
    """
    PNG 파일 읽기
    
    Args:
        file_path: 파일 경로
    
    Returns:
        np.ndarray: 이미지 배열
    """
    img = Image.open(file_path)
    arr = np.array(img)
    arr = np.expand_dims(arr, 0)
    arr = np.expand_dims(arr, 0)
    return arr


def read_and_concatenate_images(file_list: List[str]) -> np.ndarray:
    """
    이미지 리스트를 읽어서 연결 (메모리 효율적)
    
    Args:
        file_list: 파일 경로 리스트
    
    Returns:
        np.ndarray: 연결된 이미지 배열
    """
    file_list = sorted(file_list)
    
    # 첫 번째 이미지로 배열 크기 결정
    first_img = read_png(file_list[0].replace("fits", "png"))
    n_images = len(file_list)
    shape = (n_images, *first_img.shape[1:])
    
    # Pre-allocate
    result = np.empty(shape, dtype=first_img.dtype)
    result[0] = first_img[0]
    
    # 나머지 이미지 로드
    for i, file_path in enumerate(file_list[1:], 1):
        file_path_png = file_path.replace("fits", "png")
        result[i] = read_png(file_path_png)[0]
    
    return result


# ============================================================================
# 메인 처리 함수
# ============================================================================

def collect_sdo_files(target_date: datetime.datetime) -> Tuple[bool, Optional[str], Optional[Dict[str, List[str]]]]:
    """
    SDO 파일 경로 수집
    
    Args:
        target_date: 대상 날짜
    
    Returns:
        Tuple[bool, Optional[str], Optional[Dict]]: (성공여부, 에러메시지, 파일리스트)
    """
    date = target_date - datetime.timedelta(days=BEFORE_DAY)
    file_lists = {table_name: [] for table_name in SDO_TABLE_NAMES}
    
    for n in range(SDO_NUM_SEQUENCE):
        for table_name in SDO_TABLE_NAMES:
            try:
                data = get_sdo_data(table_name, date)
                
                if len(data) == 0:
                    error_msg = f"No SDO data: {table_name} at {date}"
                    return False, error_msg, None
                
                file_path = data[0]["file_path"]
                
                if not os.path.exists(file_path):
                    error_msg = f"SDO file not found: {file_path}"
                    return False, error_msg, None
                
                file_lists[table_name].append(file_path)
                
            except Exception as e:
                error_msg = f"SDO data error: {table_name} at {date}: {e}"
                return False, error_msg, None
        
        date += datetime.timedelta(hours=SDO_CADENCE)
    
    # 데이터 개수 확인
    for table_name in SDO_TABLE_NAMES:
        if len(file_lists[table_name]) != SDO_NUM_SEQUENCE:
            error_msg = f"Incomplete SDO data: {table_name} has {len(file_lists[table_name])}/{SDO_NUM_SEQUENCE} files"
            return False, error_msg, None
    
    return True, None, file_lists


def collect_omni_data(target_date: datetime.datetime) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """
    OMNI 데이터 수집
    
    Args:
        target_date: 대상 날짜
    
    Returns:
        Tuple[bool, Optional[str], Optional[DataFrame]]: (성공여부, 에러메시지, 데이터프레임)
    """
    date = target_date - datetime.timedelta(days=BEFORE_DAY)
    omni_datas = []
    
    for n in range(OMNI_NUM_SEQUENCE):
        try:
            data = get_omni_data(date)
            
            if len(data) == 0:
                error_msg = f"No OMNI data at {date}"
                return False, error_msg, None
            
            omni_datas.append(data[0])
            
        except Exception as e:
            error_msg = f"OMNI data error at {date}: {e}"
            return False, error_msg, None
        
        date += datetime.timedelta(hours=OMNI_CADENCE)
    
    # 데이터 개수 확인
    if len(omni_datas) != OMNI_NUM_SEQUENCE:
        error_msg = f"Incomplete OMNI data: {len(omni_datas)}/{OMNI_NUM_SEQUENCE} records"
        return False, error_msg, None
    
    # DataFrame 생성 및 정렬
    df = pd.DataFrame(omni_datas)
    df = df.sort_values(by='datetime', ascending=True)
    
    return True, None, df


def save_merged_data(target_date: datetime.datetime, sdo_files: Dict[str, List[str]], omni_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    병합된 데이터 저장
    
    Args:
        target_date: 대상 날짜
        sdo_files: SDO 파일 경로 딕셔너리
        omni_df: OMNI 데이터프레임
    
    Returns:
        Tuple[bool, Optional[str]]: (성공여부, 에러메시지)
    """
    try:
        # 디렉토리 생성
        save_dir = f"{SAVE_ROOT}/{target_date:%Y%m%d%H}"
        os.makedirs(save_dir, exist_ok=True)
        
        # SDO 데이터 저장
        for table_name in SDO_TABLE_NAMES:
            file_list = sdo_files[table_name]
            data = read_and_concatenate_images(file_list)
            save_path = f"{save_dir}/{table_name}.npy"
            np.save(save_path, data)
        
        # OMNI 데이터 저장
        file_path = f"{save_dir}/omni.csv"
        omni_df.to_csv(file_path, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
        
        return True, None
        
    except Exception as e:
        error_msg = f"Save error: {e}"
        return False, error_msg


def process_single_date(target_date: datetime.datetime) -> Tuple[bool, Optional[str]]:
    """
    단일 날짜 처리 (전체 파이프라인)
    
    Args:
        target_date: 처리할 날짜
    
    Returns:
        Tuple[bool, Optional[str]]: (성공여부, 에러메시지)
    """
    # 1. SDO 파일 수집
    success, error_msg, sdo_files = collect_sdo_files(target_date)
    if not success:
        return False, error_msg
    
    # 2. OMNI 데이터 수집
    success, error_msg, omni_df = collect_omni_data(target_date)
    if not success:
        return False, error_msg
    
    # 3. 저장
    success, error_msg = save_merged_data(target_date, sdo_files, omni_df)
    if not success:
        return False, error_msg
    
    logger.info(f"Successfully processed: {target_date}")
    return True, None


# ============================================================================
# 병렬 처리 래퍼
# ============================================================================

def process_one_date_wrapper(target_date: datetime.datetime) -> Dict:
    """
    병렬 처리를 위한 래퍼 함수
    
    Args:
        target_date: 처리할 날짜
    
    Returns:
        Dict: 처리 결과
    """
    # 이미 처리된 날짜 스킵
    if is_already_processed(target_date):
        return {
            'date': target_date,
            'success': True,
            'error': None,
            'skipped': True
        }
    
    # 재시도 포함 처리
    try:
        success, error_msg = retry_on_failure(
            lambda: process_single_date(target_date),
            max_retries=MAX_RETRIES,
            delay=RETRY_DELAY
        )
        
        return {
            'date': target_date,
            'success': success,
            'error': error_msg,
            'skipped': False
        }
        
    except Exception as e:
        logger.error(f"Failed after {MAX_RETRIES} retries: {target_date}: {e}")
        return {
            'date': target_date,
            'success': False,
            'error': str(e),
            'skipped': False
        }


# ============================================================================
# 메인 실행
# ============================================================================

def generate_date_list(start_date: datetime.datetime, end_date: datetime.datetime) -> List[datetime.datetime]:
    """
    처리할 날짜 리스트 생성
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
    
    Returns:
        List[datetime.datetime]: 날짜 리스트
    """
    dates = []
    current = start_date
    
    while current < end_date:
        dates.append(current)
        current += datetime.timedelta(hours=DATASET_CADENCE)
    
    return dates


def analyze_results(results: List[Dict]):
    """
    처리 결과 분석 및 출력
    
    Args:
        results: 처리 결과 리스트
    """
    success_count = sum(1 for r in results if r['success'])
    skipped_count = sum(1 for r in results if r.get('skipped', False))
    failed_count = len(results) - success_count
    
    logger.info("=" * 60)
    logger.info("PROCESSING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total dates: {len(results)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Skipped (already processed): {skipped_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("=" * 60)
    
    # 실패한 날짜 출력
    if failed_count > 0:
        logger.error("\nFailed dates:")
        for r in results:
            if not r['success']:
                logger.error(f"  {r['date']}: {r['error']}")
    
    # 처리 완료된 날짜 저장
    processed_dates = [r['date'] for r in results if r['success']]
    save_progress(processed_dates)


def main():
    """메인 실행 함수"""
    logger.info("=" * 60)
    logger.info("SDO + OMNI DATA MERGER")
    logger.info("=" * 60)
    logger.info(f"DATASET CADENCE: {DATASET_CADENCE} hours")
    logger.info(f"SDO CADENCE: {SDO_CADENCE} hours")
    logger.info(f"OMNI CADENCE: {OMNI_CADENCE} hours")
    logger.info(f"BEFORE {BEFORE_DAY} days to AFTER {AFTER_DAY} days")
    logger.info(f"SDO_NUM_SEQUENCE: {SDO_NUM_SEQUENCE}")
    logger.info(f"OMNI_NUM_SEQUENCE: {OMNI_NUM_SEQUENCE}")
    logger.info(f"NUM_WORKERS: {NUM_WORKERS}")
    logger.info("=" * 60)
    
    # 날짜 범위 설정
    start_date = datetime.datetime(2011, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2025, 1, 1, 0, 0, 0)
    
    # 날짜 리스트 생성
    all_dates = generate_date_list(start_date, end_date)
    logger.info(f"Total dates to process: {len(all_dates)}")
    
    # 이미 처리된 날짜 제외
    processed_dates = load_progress()
    dates_to_process = [d for d in all_dates if d not in processed_dates]
    
    logger.info(f"Already processed: {len(processed_dates)}")
    logger.info(f"To process: {len(dates_to_process)}")
    
    if len(dates_to_process) == 0:
        logger.info("All dates already processed!")
        return
    
    # 병렬 처리
    logger.info("\nStarting parallel processing...")
    
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_one_date_wrapper, dates_to_process),
            total=len(dates_to_process),
            desc="Processing dates",
            unit="date"
        ))
    
    # 결과 분석
    analyze_results(results)
    
    logger.info("\n✓ Processing complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"\n✗ Fatal error: {e}")
        sys.exit(1)

# import os
# import datetime

# from PIL import Image
# import numpy as np
# import pandas as pd

# from egghouse.database import PostgresManager

# SAVE_ROOT = "/Users/eunsupark/projects/ap/data"
# SDO_DB_CONFIG = {
#     "host": "localhost",
#     "port": 5432,
#     "database": "sdo",
#     "user": "eunsupark",
#     "password": "eunsupark",
#     "log_queries": False
# }
# OMNI_DB_CONFIG = {
#     "host": "localhost",
#     "port": 5432,
#     "database": "omni",
#     "user": "eunsupark",
#     "password": "eunsupark",
#     "log_queries": False
# }

# OMNI_CADENCE = 3
# SDO_CADENCE = 6
# DATASET_CADENCE = 3

# BEFORE_DAY = 10
# AFTER_DAY = 5

# SDO_NUM_SEQUENCE = (BEFORE_DAY + AFTER_DAY) * (24//SDO_CADENCE)
# OMNI_NUM_SEQUENCE = (BEFORE_DAY + AFTER_DAY) * (24//OMNI_CADENCE)

# SDO_TABLE_NAMES = ["aia_193", "aia_211", "hmi_magnetogram"]

# sdo_db = PostgresManager(**SDO_DB_CONFIG)
# omni_db = PostgresManager(**OMNI_DB_CONFIG)

# print(f"DATASET CADENCE: {DATASET_CADENCE} hours")
# print(f"SDO CADENCE: {SDO_CADENCE} hours")
# print(f"OMNI CADENCE: {OMNI_CADENCE} hours")

# print(f"BEFORE {BEFORE_DAY} days to AFTER {AFTER_DAY} days")
# print(f"SDO_NUM_SEQUENCE: {SDO_NUM_SEQUENCE}")
# print(f"OMNI_NUM_SEQUENCE: {OMNI_NUM_SEQUENCE}")


# def read_png(file_path):
#     img = Image.open(file_path)
#     arr = np.array(img)
#     arr = np.expand_dims(arr, 0)
#     arr = np.expand_dims(arr, 0)
#     return arr


# def main(target_date):

#     # Find SDO data from database
#     date = target_date - datetime.timedelta(days=BEFORE_DAY)
#     file_lists = {}
#     for table_name in SDO_TABLE_NAMES:
#         file_lists[table_name]=[]
#     for n in range(SDO_NUM_SEQUENCE) :
#         for table_name in SDO_TABLE_NAMES:
#             data = sdo_db.select(table_name, where={"date_rounded": date})
#             if len(data) > 0 :
#                 file_path = data[0]["file_path"]
#                 if os.path.exists(file_path):
#                     file_lists[table_name].append(file_path)
#                 else :
#                     return False, None
#             else :
#                 return False, None
#         date += datetime.timedelta(hours=SDO_CADENCE)

#     # Check number of SDO data
#     for table_name in SDO_TABLE_NAMES:
#         if len(file_lists[table_name]) != SDO_NUM_SEQUENCE :
#             return False, None

#     # Concatenate SDO data
#     sdo_datas = {}
#     for table_name in SDO_TABLE_NAMES:
#         file_list = sorted(file_lists[table_name])
#         data_list = []
#         for file_path in file_list :
#             file_path_png = file_path.replace("fits", "png")
#             data = read_png(file_path_png)
#             data_list.append(data)
#         data_list = np.concatenate(data_list, 0)
#         sdo_datas[table_name] = data_list

#     # Find OMNI data from database
#     date = target_date - datetime.timedelta(days=BEFORE_DAY)
#     omni_datas = []
#     for n in range(OMNI_NUM_SEQUENCE) :
#         data = omni_db.select("low_resolution", where={"datetime": date})
#         if len(data) > 0 :
#             omni_datas.append(data[0])
#         else :
#             return False, None
#         date += datetime.timedelta(hours=OMNI_CADENCE)
        
#     # Check number of OMNI data
#     if len(omni_datas) != OMNI_NUM_SEQUENCE:
#         return False, None
    
#     # Concatenate OMNI data
#     df = pd.DataFrame(omni_datas)
#     df = df.sort_values(by='datetime', ascending=True)

#     # Create directory
#     save_dir = f"{SAVE_ROOT}/{target_date:%Y%m%d%H}"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)

#     # Save SDO data
#     for table_name in SDO_TABLE_NAMES:
#         data = sdo_datas[table_name]
#         save_path = f"{save_dir}/{table_name}.npy"
#         np.save(save_path, data)

#     # Save OMNI data
#     file_path = f"{save_dir}/omni.csv"
#     df.to_csv(file_path, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')

#     print(f"{target_date}: Successfully saved")        
#     return True, None

# if __name__ == "__main__" :

#     target_date = datetime.datetime(2011, 1, 1, 0, 0, 0)
#     end_date = datetime.datetime(2025, 1, 1, 0, 0, 0)

#     while target_date < end_date :
#         result = main(target_date)
#         target_date += datetime.timedelta(hours=DATASET_CADENCE)