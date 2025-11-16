import os
import datetime
import psycopg2
from sunpy.map import Map
from shutil import move
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 데이터베이스 연결 설정
DB_CONFIG = {
    'dbname': 'sdo_data',
    'user': 'eunsupark',
    'password': 'eunsupark',
    'host': 'localhost',
    'port': '5432'
}
DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
INVALID_HEADER_DIR = f"{DATA_ROOT}/invalid_header"
INVALID_FILE_DIR = f"{DATA_ROOT}/invalid_file"
NON_ZERO_QUALITY_DIR = f"{DATA_ROOT}/non_zero_quality"
AIA_DIR = f"{DATA_ROOT}/aia"
HMI_DIR = f"{DATA_ROOT}/hmi"


def parsing_date(t_rec, instrument):
    """T_REC 문자열을 datetime 객체로 변환"""
    if instrument == "aia" :
        date = datetime.datetime.strptime(t_rec, "%Y-%m-%dT%H:%M:%S.%f")
    elif instrument == "hmi" :
        date = datetime.datetime.strptime(t_rec, "%Y.%m.%d_%H:%M:%S.%f_TAI")
    return date


def extract_metadata_only(file_path):
    """
    1단계: 메타데이터만 추출 (DB 접근 없음, 병렬 처리 안전)
    
    Returns:
        dict or str: 성공 시 메타데이터, 실패 시 에러 타입
    """
    file_name = os.path.basename(file_path)
    
    try:
        sdo_map = Map(file_path)
        meta = sdo_map.meta
        
        # 필수 메타데이터 확인
        required_keys = ["T_REC", "QUALITY", "TELESCOP"]
        for key in required_keys:
            if key not in meta:
                move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
                return "invalid_header"
        
        t_rec = meta["T_REC"]
        quality = meta["QUALITY"]
        instrument = meta["TELESCOP"].split('/')[1].lower()
        
        # 날짜 파싱
        try:
            date = parsing_date(t_rec, instrument)
            if date.minute >= 30:
                date_rounded = date.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
            else:
                date_rounded = date.replace(minute=0, second=0, microsecond=0)
        except Exception as e:
            move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
            return "invalid_header"
        
        # wavelength 추출
        if instrument == "aia":
            if "WAVELNTH" not in meta:
                move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
                return "invalid_header"
            wavelength = meta["WAVELNTH"]
        elif instrument == "hmi":
            if "CONTENT" not in meta:
                move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
                return "invalid_header"
            wavelength = meta["CONTENT"].lower()
        else:
            move(file_path, f"{INVALID_HEADER_DIR}/{file_name}")
            return "invalid_header"
        
        table_name = f"{instrument}_{wavelength}"
        
        # quality 확인
        if quality != 0:
            move(file_path, f"{NON_ZERO_QUALITY_DIR}/{file_name}")
            return "non_zero_quality"
        
        # 메타데이터 반환
        return {
            'file_path': file_path,
            'file_name': file_name,
            'instrument': instrument,
            'wavelength': wavelength,
            'date': date,
            'date_rounded': date_rounded,
            'table_name': table_name,
            'quality': quality
        }
        
    except Exception as e:
        try:
            move(file_path, f"{INVALID_FILE_DIR}/{file_name}")
        except:
            pass
        return f"invalid_file: {str(e)}"


def batch_insert_to_db(metadata_list):
    """
    2단계: 메타데이터를 DB에 일괄 삽입 (순차 처리, DB 충돌 방지)
    
    Args:
        metadata_list: extract_metadata_only에서 반환된 딕셔너리 리스트
    
    Returns:
        dict: 처리 결과 통계
    """
    stats = {
        "success": 0,
        "duplicate": 0,
        "updated": 0,
        "db_error": 0,
        "move_error": 0
    }
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 테이블별로 그룹화
        by_table = {}
        for meta in metadata_list:
            table_name = meta['table_name']
            if table_name not in by_table:
                by_table[table_name] = []
            by_table[table_name].append(meta)
        
        # 테이블별로 처리
        for table_name, metas in by_table.items():
            print(f"\nProcessing table: {table_name} ({len(metas)} files)")
            
            # 기존 레코드 조회
            dates = [m['date'] for m in metas]
            check_query = f"""
                SELECT date, file_name
                FROM {table_name}
                WHERE date = ANY(%s)
            """
            cursor.execute(check_query, (dates,))
            existing = {row[0]: row[1] for row in cursor.fetchall()}
            
            # INSERT/UPDATE 분류
            to_insert = []
            to_update = []
            to_move = []
            
            for meta in metas:
                date = meta['date']
                date_rounded = meta['date_rounded']
                file_name = meta['file_name']
                quality = meta['quality']
                file_path = meta['file_path']
                instrument = meta['instrument']
                
                if date in existing:
                    if existing[date] == file_name:
                        stats["duplicate"] += 1
                    else:
                        to_update.append((date_rounded, file_name, quality, date))
                        to_move.append((file_path, instrument, file_name))
                        stats["updated"] += 1
                else:
                    to_insert.append((date_rounded, date, file_name, quality))
                    to_move.append((file_path, instrument, file_name))
                    stats["success"] += 1
            
            # 배치 INSERT
            if to_insert:
                from psycopg2.extras import execute_batch
                insert_query = f"""
                    INSERT INTO {table_name} (date_rounded, date, file_name, quality)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (date) DO NOTHING
                """
                execute_batch(cursor, insert_query, to_insert, page_size=1000)
                print(f"  Inserted: {len(to_insert)}")
            
            # 배치 UPDATE
            if to_update:
                from psycopg2.extras import execute_batch
                update_query = f"""
                    UPDATE {table_name}
                    SET date_rounded = %s, file_name = %s, quality = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE date = %s
                """
                execute_batch(cursor, update_query, to_update, page_size=1000)
                print(f"  Updated: {len(to_update)}")
            
            # 커밋
            conn.commit()
            
            # 파일 이동
            for file_path, instrument, file_name in tqdm(to_move, desc=f"Moving {table_name}"):
                try:
                    if instrument == 'aia':
                        target_path = f"{AIA_DIR}/{file_name}"
                    elif instrument == 'hmi':
                        target_path = f"{HMI_DIR}/{file_name}"
                    else:
                        continue
                    
                    move(file_path, target_path)
                except Exception as e:
                    print(f"⚠ Move failed for {file_name}: {e}")
                    stats["move_error"] += 1
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"✗ Database error: {e}")
        stats["db_error"] += 1
        if conn:
            conn.rollback()
            conn.close()
    
    return stats


if __name__ == "__main__":
    from glob import glob
    
    # 디렉토리 생성
    for dir_path in [INVALID_HEADER_DIR, INVALID_FILE_DIR,
                     NON_ZERO_QUALITY_DIR, AIA_DIR, HMI_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 파일 목록 수집
    file_list = []
    file_list += glob(f"{DATA_ROOT}/downloaded/*.fits")
    print(f"Total files: {len(file_list)}")

    while len(file_list) > 0 :
    
        file_list = file_list[:1000]
    
        # 병렬 처리 설정
        num_processes = 8 #cpu_count()
        print(f"Using {num_processes} processes\n")
        
        # ==========================================================
        # 1단계: 메타데이터 추출 (병렬 처리)
        # ==========================================================
        print("Step 1: Extracting metadata (parallel)...")
        metadata_list = []
        error_counts = {
            "invalid_file": 0,
            "invalid_header": 0,
            "non_zero_quality": 0
        }
        
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(extract_metadata_only, file_list),
                total=len(file_list),
                desc="Extracting"
            ))
        
        # 결과 분류
        for result in results:
            if isinstance(result, dict):
                metadata_list.append(result)
            elif result == "invalid_header":
                error_counts["invalid_header"] += 1
            elif result == "non_zero_quality":
                error_counts["non_zero_quality"] += 1
            elif isinstance(result, str) and result.startswith("invalid_file"):
                error_counts["invalid_file"] += 1
        
        print(f"\nValid files: {len(metadata_list)}")
        for key, value in error_counts.items():
            print(f"{key}: {value}")
        
        # ==========================================================
        # 2단계: DB 업데이트 (순차 처리)
        # ==========================================================
        if len(metadata_list) > 0:
            print("\nStep 2: Updating database (sequential)...")
            stats = batch_insert_to_db(metadata_list)
            
            # ==========================================================
            # 최종 통계
            # ==========================================================
            print("\n" + "="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            print(f"Total processed         : {len(file_list):6d}")
            print(f"Valid metadata          : {len(metadata_list):6d}")
            print(f"Success (inserted)      : {stats['success']:6d}")
            print(f"Updated                 : {stats['updated']:6d}")
            print(f"Duplicate (skipped)     : {stats['duplicate']:6d}")
            print(f"Invalid file            : {error_counts['invalid_file']:6d}")
            print(f"Invalid header          : {error_counts['invalid_header']:6d}")
            print(f"Non-zero quality        : {error_counts['non_zero_quality']:6d}")
            print(f"DB errors               : {stats['db_error']:6d}")
            print(f"Move errors             : {stats['move_error']:6d}")
            print("="*60)
        else:
            print("No valid files to process")

        file_list = []
        # file_list += glob(f"{DATA_ROOT}/aia_tmp/*.fits")
        # file_list += glob(f"{DATA_ROOT}/hmi_tmp/*.fits")
        file_list += glob(f"{DATA_ROOT}/invalid_header/*.fits")
        print(f"Total files: {len(file_list)}")
