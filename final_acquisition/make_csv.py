import os
import datetime
import logging
from glob import glob
from multiprocessing import Pool, cpu_count

from sunpy.map import Map
import pandas as pd
from tqdm import tqdm



DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
CSV_PATH = f"{DATA_ROOT}/sdo_fits.csv"
LOG_PATH = f"{DATA_ROOT}/sdo_fits.log"
COLUMNS = ["datetime", "aia_193", "aia_211", "hmi_magnetogram"]

# 병렬 처리 설정
NUM_WORKERS = 8 # cpu_count() - 1  # CPU 코어 수 - 1 (시스템 여유 확보)
USE_PARALLEL = True  # 병렬 처리 사용 여부


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)


def parse_file(file_path):
    """
    단일 FITS 파일을 파싱하여 데이터 딕셔너리 반환
    
    Returns:
        tuple: (data_dict or None, status_message)
        status: 'success', 'skipped_quality', 'error'
    """
    try:
        file_name = os.path.basename(file_path)
        sdo_map = Map(file_path)
        meta = sdo_map.meta
        
        # 필수 메타데이터 확인
        required_keys = ["T_REC", "QUALITY", "TELESCOP"]
        for key in required_keys:
            if key not in meta:
                return None, f"Missing metadata: {key}"
        
        # 메타데이터 추출
        t_rec = meta["T_REC"]
        quality = meta["QUALITY"]
        instrument = meta["TELESCOP"].split('/')[1].lower()
        
        # 날짜 파싱
        try:
            year = int(t_rec[0:4])
            month = int(t_rec[5:7])
            day = int(t_rec[8:10])
            hour = int(t_rec[11:13])
            minute = int(t_rec[14:16])
            second = int(t_rec[17:19])               
            date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
            if datetime.minute >= 30 :
                date = date.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
            else :
                date = date.replace(minute=0, second=0, microsecond=0)
        except (ValueError, IndexError) as e:
            return None, f"Invalid T_REC format: {t_rec}"
        
        # wavelength 추출
        if instrument == "aia":
            if "WAVELNTH" not in meta:
                return None, "Missing WAVELNTH for AIA"
            wavelength = meta["WAVELNTH"]
        elif instrument == "hmi":
            if "CONTENT" not in meta:
                return None, "Missing CONTENT for HMI"
            wavelength = meta["CONTENT"].lower()
        else:
            return None, f"Unknown instrument: {instrument}"
        
        # quality가 0인 것만 처리
        if quality != 0:
            return None, f"skipped_quality"
        
        data_dict = {
            "datetime": date,
            f"{instrument}_{wavelength}": file_name
        }

        save_path = f"{DATA_ROOT}/{instrument}/{file_name}"
        command = f"mv {file_path} {save_path}"
        os.system(command)
        
        return data_dict, "success"
        
    except Exception as e:
        return None, f"error: {str(e)}"


def update_csv_batch(csv_path, data_list):
    """
    수집된 데이터 리스트로 CSV를 배치 업데이트
    
    Args:
        csv_path: CSV 파일 경로
        data_list: parse_file()에서 반환된 딕셔너리 리스트
    """
    if not data_list:
        logging.warning("No valid data to update")
        return
    
    # 기존 CSV 읽기
    try:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        logging.info(f"Loaded existing CSV with {len(df)} rows")
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        raise
    
    updated_count = 0
    added_count = 0
    
    # 각 데이터 처리
    for data_dict in data_list:
        date = data_dict["datetime"]
        mask = df['datetime'] == date
        
        if mask.any():
            # 기존 행 업데이트 (값이 다른 경우만)
            changed = False
            for key, new_value in data_dict.items():
                if key == "datetime":
                    continue
                
                if key in df.columns:
                    current_value = df.loc[mask, key].iloc[0]
                    # NaN 또는 값이 다른 경우 업데이트
                    if pd.isna(current_value) or current_value != new_value:
                        df.loc[mask, key] = new_value
                        changed = True
                else:
                    # 새 컬럼 추가
                    df[key] = None
                    df.loc[mask, key] = new_value
                    changed = True
                    logging.info(f"Added new column: {key}")
            
            if changed:
                updated_count += 1
        else:
            # 새 행 추가
            df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
            added_count += 1
    
    # 정렬
    df = df.sort_values(by='datetime', ascending=True)
    
    # 저장
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
        logging.info(f"CSV updated successfully: {added_count} added, {updated_count} updated, total {len(df)} rows")
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        raise


if __name__ == "__main__":
    
    logging.info("=== Starting FITS to CSV processing ===")
    
    # CSV 파일이 없으면 생성
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(CSV_PATH, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
        logging.info(f"Created new CSV: {CSV_PATH}")
    else:
        logging.info(f"Using existing CSV: {CSV_PATH}")

    num_exp = 5236 * 24 * 3
    aia_expected = 5236 * 24 * 2
    hmi_expected = 5236 * 24
    num_aia = len(glob(os.path.join(DATA_ROOT, "aia", "*.fits")))
    num_hmi = len(glob(os.path.join(DATA_ROOT, "hmi", "*.fits")))
    num_downloaded = len(glob(os.path.join(DATA_ROOT, "downloaded", "*.fits")))
    # num_spike = len(glob(os.path.join(DATA_ROOT, "aia_spike", "*.fits")))
    num_invalid = len(glob(os.path.join(DATA_ROOT, "invalid", "*.fits")))
    num_total = num_aia + num_hmi + num_downloaded + num_invalid
    percentage = 100.*(float(num_total)/float(num_exp))

    print(f"classified aia                : {num_aia:6d}")
    print(f"classified hmi                : {num_hmi:6d}")
    print(f"not classified                : {num_downloaded:6d}")
    # print(f"aia_spike                     : {num_spike:6d}")
    print(f"classified as invalid quality : {num_invalid:6d}")
    print(f"total downloaded              : {num_total:6d}")
    print(f"total expected                : {num_exp:6d}")
    print(f"progress rate                 : {percentage:.2f}%")

    # import sys
    # sys.exit()

    for file_path in glob(os.path.join(DATA_ROOT, "downloaded", "*spike*.fits")) :
        file_name = os.path.basename(file_path)
        os.system(f"mv {file_path} {DATA_ROOT}/aia_spike/")
    
    # 파일 리스트 수집
    file_list = []
    file_list += glob(os.path.join(DATA_ROOT, "downloaded", "*.magnetogram.fits"))
    file_list += glob(os.path.join(DATA_ROOT, "downloaded", "*.193.image_lev1.fits"))
    file_list += glob(os.path.join(DATA_ROOT, "downloaded", "*.211.image_lev1.fits"))

    import random
    random.shuffle(file_list)
    # file_list = file_list[:100000]
    
    logging.info(f"Found {len(file_list)} FITS files")
    
    if not file_list:
        logging.warning("No FITS files found. Exiting.")
        exit(0)
    
    # 1단계: 모든 파일 파싱 (병렬 처리)
    logging.info(f"Parsing FITS files with {NUM_WORKERS if USE_PARALLEL else 1} workers...")
    
    if USE_PARALLEL and len(file_list) > 10:
        # 병렬 처리
        with Pool(processes=NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(parse_file, file_list),
                total=len(file_list),
                desc="Parsing (parallel)"
            ))
    else:
        # 순차 처리 (파일이 적거나 병렬 비활성화 시)
        results = [parse_file(f) for f in tqdm(file_list, desc="Parsing (sequential)")]
    
    # 결과 분류
    data_list = []
    stats = {
        'success': 0,
        'skipped_quality': 0,
        'error': 0
    }
    error_files = []
    
    for i, (data, status) in enumerate(results):
        if data:
            data_list.append(data)
            stats['success'] += 1
        elif 'skipped_quality' in status:
            stats['skipped_quality'] += 1
        else:
            stats['error'] += 1
            error_files.append((file_list[i], status))
            logging.debug(f"Failed: {os.path.basename(file_list[i])} - {status}")
    
    # 통계 출력
    logging.info(f"\nParsing completed:")
    logging.info(f"  - Success: {stats['success']}")
    logging.info(f"  - Skipped (quality): {stats['skipped_quality']}")
    logging.info(f"  - Errors: {stats['error']}")
    
    # 에러 파일 상세 로그
    if error_files:
        logging.warning(f"\nError details ({len(error_files)} files):")
        for file_path, status in error_files[:10]:  # 처음 10개만
            logging.warning(f"  - {os.path.basename(file_path)}: {status}")
        if len(error_files) > 10:
            logging.warning(f"  ... and {len(error_files) - 10} more errors")
    
    # 2단계: CSV 업데이트 (한 번만)
    if data_list:
        logging.info("\nUpdating CSV...")
        update_csv_batch(CSV_PATH, data_list)
    else:
        logging.warning("No valid data to add to CSV")
    
    logging.info("=== Processing completed ===")

    for file_path in file_list :
        if os.path.exists(file_path) :
            os.system(f"mv {file_path} {DATA_ROOT}/invalid/")


# import os
# import datetime
# from glob import glob

# from sunpy.map import Map
# import pandas as pd


# DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
# CSV_PATH = f"{DATA_ROOT}/sdo_fits.csv"
# COLUMNS = ["datetime", "aia_193", "aia_211", "hmi_magnetogram"]


# def parse(file_path):
#     file_name = os.path.basename(file_path)
#     sdo_map = Map(file_path)
#     meta = sdo_map.meta
#     t_rec = meta["T_REC"]
#     quality = meta["QUALITY"]
#     instrument = meta["TELESCOP"].split('/')[1].lower()
#     year = int(t_rec[0:4])
#     month = int(t_rec[5:7])
#     day = int(t_rec[8:10])
#     hour = int(t_rec[11:13])
#     date = datetime.datetime(
#         year = year,
#         month = month,
#         day = day,
#         hour = hour
#     )

#     if instrument == "aia" :
#         wavelength = meta["WAVELNTH"]
#     elif instrument == "hmi" :
#         wavelength = meta["CONTENT"].lower()

#     if quality == 0 :

#         data_dict = {
#         "datetime" : date,
#         f"{instrument}_{wavelength}" : file_name
#         }

#         df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
#         mask = df['datetime'] == date
#         if mask.any() :
#             for key, value in data_dict.items():
#                 current_value = df.loc[mask, key].iloc[0]
#                 if current_value != value :
#                     df.loc[mask, key] = value
#         else :
#             df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

#         df.sort_values(by='datetime', ascending=True)
#         df.to_csv(CSV_PATH, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')


# if __name__ == "__main__" :

#     if os.path.exists(CSV_PATH) is False :
#         df = pd.DataFrame(columns=COLUMNS)
#         df.to_csv(CSV_PATH, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')

#     file_list = []
#     file_list += glob(os.path.join(DATA_ROOT, "hmi", "*.magnetogram.fits"))
#     file_list += glob(os.path.join(DATA_ROOT, "aia", f"*.193.image_lev1.fits"))
#     file_list += glob(os.path.join(DATA_ROOT, "aia", f"*.211.image_lev1.fits"))

#     for file_path in file_list :
#         parse(file_path)
