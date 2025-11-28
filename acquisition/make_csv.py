import os
import datetime
import logging
from glob import glob
from multiprocessing import Pool, cpu_count
from shutil import move

from sunpy.map import Map
import pandas as pd
from tqdm import tqdm


DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
CSV_PATH = f"{DATA_ROOT}/sdo_fits.csv"
LOG_PATH = f"{DATA_ROOT}/sdo_fits.log"
COLUMNS = ["datetime", "aia_193", "aia_211", "hmi_magnetogram"]
INVALID_HEADER_DIR = f"{DATA_ROOT}/invalid_header"
INVALID_FILE_DIR = f"{DATA_ROOT}/invalid_file"
NON_ZERO_QUALITY_DIR = f"{DATA_ROOT}/non_zero_quality"
AIA_DIR = f"{DATA_ROOT}/aia"
HMI_DIR = f"{DATA_ROOT}/hmi"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)


def parsing_date(t_rec):
    """T_REC 문자열을 datetime 객체로 변환"""
    year = int(t_rec[0:4])
    month = int(t_rec[5:7])
    day = int(t_rec[8:10])
    hour = int(t_rec[11:13])
    minute = int(t_rec[14:16])
    second = int(t_rec[17:19])         
    date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    return date


def extract_metadata(file_path):
    """
    파일에서 메타데이터만 추출 (CSV 접근 없음)
    병렬 처리용 함수
    
    Returns:
        dict or str: 성공 시 메타데이터 딕셔너리, 실패 시 에러 타입 문자열
    """
    file_name = os.path.basename(file_path)
    
    try:
        sdo_map = Map(file_path)
        meta = sdo_map.meta
        
        # 필수 메타데이터 확인
        required_keys = ["T_REC", "QUALITY", "TELESCOP"]
        for key in required_keys:
            if key not in meta:
                # move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
                return "invalid_header"
        
        # 메타데이터 추출
        t_rec = meta["T_REC"]
        quality = meta["QUALITY"]
        instrument = meta["TELESCOP"].split('/')[1].lower()
        
        # 날짜 파싱 (한 번만)
        try:
            date_file = parsing_date(t_rec)
            # 시간 반올림
            if date_file.minute >= 30:
                date_rounded = date_file.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
            else:
                date_rounded = date_file.replace(minute=0, second=0, microsecond=0)
        except (ValueError, IndexError) as e:
            # move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
            return "invalid_header"
        
        # wavelength 추출
        if instrument == "aia":
            if "WAVELNTH" not in meta:
                # move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
                return "invalid_header"
            wavelength = meta["WAVELNTH"]
        elif instrument == "hmi":
            if "CONTENT" not in meta:
                # move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
                return "invalid_header"
            wavelength = meta["CONTENT"].lower()
        else:
            # move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
            return "invalid_header"
        
        # quality가 0인 것만 처리
        if quality != 0:
            # move(file_path, f"{NON_ZERO_QUALITY_DIR}/{file_name}")
            return "non_zero_quality"
        
        # 메타데이터 반환
        return {
            'file_path': file_path,
            'file_name': file_name,
            'instrument': instrument,
            'wavelength': wavelength,
            'date_file': date_file,
            'date_rounded': date_rounded,
            'csv_key': f"{instrument}_{wavelength}"
        }
        
    except Exception as e:
        try:
            pass
            # move(file_path, f"{INVALID_FILE_DIR}/{file_name}")
        except:
            pass
        # 병렬 처리에서는 logging 대신 에러 메시지 반환
        return f"invalid_file: {str(e)}"


def update_csv_with_metadata(df, metadata_list):
    """
    수집된 메타데이터로 CSV 업데이트
    
    Args:
        df: pandas DataFrame
        metadata_list: extract_metadata에서 반환된 딕셔너리 리스트
    
    Returns:
        tuple: (updated_df, success_count, moved_files)
    """
    success_count = 0
    moved_files = []
    
    for metadata in tqdm(metadata_list, desc="Updating CSV"):
        date_rounded = metadata['date_rounded']
        csv_key = metadata['csv_key']
        csv_value = metadata['file_name']
        instrument = metadata['instrument']
        date_file = metadata['date_file']
        file_path = metadata['file_path']
        
        mask = df['datetime'] == date_rounded
        
        if mask.any():
            # 기존 행이 있는 경우
            current_value = df.loc[mask, csv_key].iloc[0]
            
            if pd.isna(current_value):
                # 값이 없으면 추가
                df.loc[mask, csv_key] = csv_value
                success_count += 1
                moved_files.append((file_path, instrument, csv_value))
                
            else:
                # 값이 있으면 시간 차이 비교
                if current_value != csv_value:
                    try:
                        current_file_path = f"{DATA_ROOT}/{instrument}/{current_value}"
                        M_current = Map(current_file_path)
                        date_current = parsing_date(M_current.meta["T_REC"])
                        
                        dif_current = abs(date_rounded - date_current)
                        dif_new = abs(date_rounded - date_file)
                        
                        if dif_new < dif_current:
                            df.loc[mask, csv_key] = csv_value
                            success_count += 1
                            moved_files.append((file_path, instrument, csv_value))
                            logging.info(f"Replaced {current_value} with {csv_value} (closer to rounded time)")
                        else:
                            # 새 파일이 더 멀면 무시 (파일은 그대로 원본 위치에 둠)
                            logging.info(f"Kept {current_value}, skipped {csv_value} (further from rounded time)")
                    except Exception as e:
                        # 기존 파일 로드 실패 시 새 파일로 교체
                        logging.warning(f"Failed to load existing file {current_value}: {e}. Replacing with new file.")
                        df.loc[mask, csv_key] = csv_value
                        success_count += 1
                        moved_files.append((file_path, instrument, csv_value))
                else:
                    # 같은 파일명이면 스킵
                    success_count += 1
                    moved_files.append((file_path, instrument, csv_value))
        else:
            # 새 행 추가
            new_row = {'datetime': date_rounded, csv_key: csv_value}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            success_count += 1
            moved_files.append((file_path, instrument, csv_value))
    
    # 날짜순 정렬
    df = df.sort_values(by='datetime', ascending=True)
    
    return df, success_count, moved_files


def move_processed_files(moved_files):
    """처리 완료된 파일들을 최종 디렉토리로 이동"""
    for file_path, instrument, file_name in tqdm(moved_files, desc="Moving files"):
        try:
            if instrument == 'aia':
                save_path = f"{AIA_DIR}/{file_name}"
            elif instrument == 'hmi':
                save_path = f"{HMI_DIR}/{file_name}"
            else:
                continue
            
            # 중복 파일명 처리
            if os.path.exists(save_path):
                logging.warning(f"File already exists: {save_path}. Overwriting.")
            
            # move(file_path, save_path)
        except Exception as e:
            logging.error(f"Failed to move {file_path}: {e}")


if __name__ == "__main__":
    
    # 디렉토리 생성
    for dir_path in [INVALID_HEADER_DIR, INVALID_FILE_DIR, 
                     NON_ZERO_QUALITY_DIR, AIA_DIR, HMI_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # CSV 파일 초기화
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(CSV_PATH, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
        logging.info(f"Created new CSV: {CSV_PATH}")
    else:
        logging.info(f"Using existing CSV: {CSV_PATH}")
    
    # 파일 목록 수집
    file_list = []
    file_list += glob(f"{DATA_ROOT}/aia/*.fits")
    # file_list += glob(f"{DATA_ROOT}/aia_tmp2/*.fits")
    file_list += glob(f"{DATA_ROOT}/hmi/*.fits")
    # file_list += glob(f"{DATA_ROOT}/hmi_tmp2/*.fits")
    
    logging.info(f"Total files found: {len(file_list)}")
    
    # 테스트용: 처음 100개만 처리 (전체 처리 시 [:100] 제거)
    file_list = file_list
    
    # 병렬 처리 설정
    num_processes = 8 # cpu_count()  # 또는 원하는 프로세스 수 지정 (예: num_processes = 4)
    logging.info(f"Using {num_processes} processes for parallel processing")
    
    # ==========================================================
    # 1단계: 메타데이터 추출 (병렬 처리)
    # ==========================================================
    logging.info("Step 1: Extracting metadata from files (parallel)...")
    metadata_list = []
    num_invalid_file = 0
    num_invalid_header = 0
    num_non_zero_quality = 0
    
    # 병렬 처리로 메타데이터 추출
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(extract_metadata, file_list),
            total=len(file_list),
            desc="Extracting metadata"
        ))
    
    # 결과 분류
    for result in results:
        if isinstance(result, dict):
            metadata_list.append(result)
        elif result == "invalid_header":
            num_invalid_header += 1
        elif result == "non_zero_quality":
            num_non_zero_quality += 1
        elif isinstance(result, str) and result.startswith("invalid_file"):
            num_invalid_file += 1
    
    logging.info(f"Valid files: {len(metadata_list)}")
    logging.info(f"Invalid header: {num_invalid_header}")
    logging.info(f"Invalid file: {num_invalid_file}")
    logging.info(f"Non-zero quality: {num_non_zero_quality}")
    
    # ==========================================================
    # 2단계: CSV 읽기 (한 번만)
    # ==========================================================
    if len(metadata_list) > 0:
        logging.info("Step 2: Loading CSV...")
        df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
        logging.info(f"Loaded CSV with {len(df)} rows")
        
        # ==========================================================
        # 3단계: CSV 업데이트 (메모리에서만)
        # ==========================================================
        logging.info("Step 3: Updating CSV in memory...")
        df, num_success, moved_files = update_csv_with_metadata(df, metadata_list)
        
        # ==========================================================
        # 4단계: CSV 저장 (한 번만)
        # ==========================================================
        logging.info("Step 4: Saving CSV...")
        df.to_csv(CSV_PATH, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
        logging.info(f"CSV saved with {len(df)} rows")
        
        # ==========================================================
        # 5단계: 파일 이동
        # ==========================================================
        logging.info("Step 5: Moving processed files...")
        move_processed_files(moved_files)
        
        # ==========================================================
        # 최종 통계
        # ==========================================================
        logging.info("="*60)
        logging.info(f"Success                 : {num_success:6d}")
        logging.info(f"Invalid file            : {num_invalid_file:6d}")
        logging.info(f"Invalid header          : {num_invalid_header:6d}")
        logging.info(f"Non-zero quality        : {num_non_zero_quality:6d}")
        logging.info(f"Total processed         : {len(file_list):6d}")
        logging.info("="*60)
    else:
        logging.warning("No valid files to process")

# import os
# import datetime
# import logging
# from glob import glob
# from multiprocessing import Pool, cpu_count
# from shutil import move

# from sunpy.map import Map
# import pandas as pd
# from tqdm import tqdm


# DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
# CSV_PATH = f"{DATA_ROOT}/sdo_fits.csv"
# LOG_PATH = f"{DATA_ROOT}/sdo_fits.log"
# COLUMNS = ["datetime", "aia_193", "aia_211", "hmi_magnetogram"]
# INVALID_HEADER_DIR = f"{DATA_ROOT}/invalid_header"
# INVALID_FILE_DIR = f"{DATA_ROOT}/invalid_file"
# NON_ZERO_QUALITY_DIR = f"{DATA_ROOT}/non_zero_quality"
# AIA_DIR = f"{DATA_ROOT}/aia"
# HMI_DIR = f"{DATA_ROOT}/hmi"


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(LOG_PATH),
#         logging.StreamHandler()
#     ]
# )


# def parsing_date(t_rec):
#     year = int(t_rec[0:4])
#     month = int(t_rec[5:7])
#     day = int(t_rec[8:10])
#     hour = int(t_rec[11:13])
#     minute = int(t_rec[14:16])
#     second = int(t_rec[17:19])         
#     date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
#     return date


# def parse_file(file_path):

#     try:
#         file_name = os.path.basename(file_path)
#         sdo_map = Map(file_path)
#         meta = sdo_map.meta
        
#         # 필수 메타데이터 확인
#         required_keys = ["T_REC", "QUALITY", "TELESCOP"]
#         for key in required_keys:
#             if key not in meta:
#                 move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
#                 return "invalid_header"
        
#         # 메타데이터 추출
#         t_rec = meta["T_REC"]
#         quality = meta["QUALITY"]
#         instrument = meta["TELESCOP"].split('/')[1].lower()
        
#         # 날짜 파싱
#         try:
#             date_file = parsing_date(t_rec)
#             date = parsing_date(t_rec)
#             if date.minute >= 30 :
#                 date = date.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
#             else :
#                 date = date.replace(minute=0, second=0, microsecond=0)
#         except (ValueError, IndexError) as e:
#             move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
#             return "invalid_header"
        
#         # wavelength 추출
#         if instrument == "aia":
#             if "WAVELNTH" not in meta:
#                 move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
#                 return "invalid_header"
#             wavelength = meta["WAVELNTH"]
#         elif instrument == "hmi":
#             if "CONTENT" not in meta:
#                 move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
#                 return "invalid_header"
#             wavelength = meta["CONTENT"].lower()
#         else:
#             move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
#             return "invalid_header"
        
#         # quality가 0인 것만 처리
#         if quality != 0:
#             move(file_path, f"{NON_ZERO_QUALITY_DIR}/{file_name}")
#             return "non_zero_quality"

#         csv_key = f"{instrument}_{wavelength}"
#         csv_value = file_name

#         df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
#         logging.info(f"Loaded existing CSV with {len(df)} rows")

#         mask = df['datetime'] == date

#         if mask.any() :
#             current_value = df.loc[mask, csv_key].iloc[0]
#             if pd.isna(current_value) :
#                 df.loc[mask, csv_key] = csv_value

#             else :
#                 if current_value != csv_value :
#                     M_current = Map(f"{DATA_ROOT}/{instrument}/{current_value}")
#                     date_current = parsing_date(M_current.meta["T_REC"])

#                     dif_current = abs(date - date_current)
#                     dif_new = abs(date - date_file)

#                     if dif_new < dif_current :
#                         df.loc[mask, csv_key] = csv_value

#         else:
#             # 새 행 추가
#             data_dict = {
#                 "datetime": date,
#                 f"{instrument}_{wavelength}": file_name
#             }
#             df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
#         df = df.sort_values(by='datetime', ascending=True)
        
#         # 저장
#         df.to_csv(CSV_PATH, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
#         logging.info(f"CSV updated successfully: {csv_key}:{csv_value}")

#         if instrument == 'aia' :
#             save_path = f"{AIA_DIR}/{file_name}"
#         elif instrument == 'hmi' :
#             save_path = f"{HMI_DIR}/{file_name}"
#         move(file_path, save_path)
#         return "success"
        
#     except Exception as e:
#         move(file_path, f"{INVALID_FILE_DIR}/{file_name}")
#         print(e)
#         return "invalid_file"


# if __name__ == "__main__" :

#     if not os.path.exists(CSV_PATH):
#         df = pd.DataFrame(columns=COLUMNS)
#         df.to_csv(CSV_PATH, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
#         logging.info(f"Created new CSV: {CSV_PATH}")
#     else:
#         logging.info(f"Using existing CSV: {CSV_PATH}")

#     file_list = []
#     file_list += glob(f"{DATA_ROOT}/aia_tmp/*.fits")
#     file_list += glob(f"{DATA_ROOT}/aia_tmp2/*.fits")
#     file_list += glob(f"{DATA_ROOT}/hmi_tmp/*.fits")
#     file_list += glob(f"{DATA_ROOT}/hmi_tmp2/*.fits")

#     print(len(file_list))

#     num_success = 0
#     num_invalid_file = 0
#     num_invalid_header = 0
#     num_non_zero_quality = 0

#     for file_path in file_list[:100] :
#         result = parse_file(file_path)
#         if result == "success":
#             num_success += 1
#         elif result == "invalid_header":
#             num_invalid_header += 1
#         elif result == "invalid_file":
#             num_invalid_file += 1
#         elif result == "non_zero_quality":
#             num_non_zero_quality += 1

#     print(f"Success                 : {num_success:6d}")
#     print(f"Invalid file            : {num_invalid_file:6d}")
#     print(f"Invalid header          : {num_invalid_header:6d}")
#     print(f"Invalid non zero quality: {num_non_zero_quality:6d}")

