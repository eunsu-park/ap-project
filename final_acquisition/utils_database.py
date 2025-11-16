"""
update_database.py - utils_database.py 활용 버전
"""

import os
import datetime
from glob import glob
from multiprocessing import Pool
from shutil import move

from sunpy.map import Map
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

from utils_database import (
    get_engine,
    get_session,
    DB_CONFIG,
    TABLE_MODELS
)


DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
INVALID_HEADER_DIR = f"{DATA_ROOT}/invalid_header"
INVALID_FILE_DIR = f"{DATA_ROOT}/invalid_file"
NON_ZERO_QUALITY_DIR = f"{DATA_ROOT}/non_zero_quality"
AIA_DIR = f"{DATA_ROOT}/aia"
HMI_DIR = f"{DATA_ROOT}/hmi"


def parsing_date(t_rec, instrument):
    """T_REC 문자열을 datetime 객체로 변환"""
    if instrument == "aia":
        date = datetime.datetime.strptime(t_rec, "%Y-%m-%dT%H:%M:%S.%f")
    elif instrument == "hmi":
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
        # utils_database의 DB_CONFIG 사용
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
                insert_query = f"""
                    INSERT INTO {table_name} (date_rounded, date, file_name, quality)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (date) DO NOTHING
                """
                execute_batch(cursor, insert_query, to_insert, page_size=1000)
                print(f"  Inserted: {len(to_insert)}")
            
            # 배치 UPDATE
            if to_update:
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
    
    # 디렉토리 생성
    for dir_path in [INVALID_HEADER_DIR, INVALID_FILE_DIR,
                     NON_ZERO_QUALITY_DIR, AIA_DIR, HMI_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 파일 목록 수집
    file_list = []
    file_list += glob(f"{DATA_ROOT}/downloaded/*.fits")
    
    print(f"Total files: {len(file_list)}")
    
    while len(file_list) > 0:
        
        # 배치 크기 설정
        batch_size = 1000
        file_batch = file_list[:batch_size]
        
        # 병렬 처리 설정
        num_processes = 8
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
                pool.imap(extract_metadata_only, file_batch),
                total=len(file_batch),
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
            print(f"Total processed         : {len(file_batch):6d}")
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
        
        # 다음 배치를 위해 파일 목록 갱신
        file_list = []
        file_list += glob(f"{DATA_ROOT}/downloaded/*.fits")
        
        print(f"\nRemaining files: {len(file_list)}")
        
        if len(file_list) == 0:
            print("\n모든 파일 처리 완료!")
            break



# """
# utils_database.py

# SDO 데이터베이스 작업을 위한 유틸리티 함수 모음
# SQLAlchemy 기반
# """

# from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, func
# from sqlalchemy.orm import sessionmaker, declarative_base
# from sqlalchemy.exc import SQLAlchemyError
# import pandas as pd
# from datetime import datetime, timedelta
# from contextlib import contextmanager
# import logging

# # 로깅 설정
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ==========================================================
# # 설정
# # ==========================================================

# DB_CONFIG = {
#     'dbname': 'sdo_data',
#     'user': 'eunsupark',
#     'password': 'eunsupark',
#     'host': 'localhost',
#     'port': '5432'
# }

# # SQLAlchemy 엔진 생성
# DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
# engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)

# # ORM Base
# Base = declarative_base()
# Session = sessionmaker(bind=engine)

# # ==========================================================
# # ORM 모델 정의
# # ==========================================================

# class AIA193(Base):
#     __tablename__ = 'aia_193'
    
#     id = Column(Integer, primary_key=True)
#     date_rounded = Column(DateTime, nullable=False)
#     date = Column(DateTime, unique=True, nullable=False)
#     file_name = Column(String(255), nullable=False)
#     quality = Column(Integer, default=0)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
#     def __repr__(self):
#         return f"<AIA193(date={self.date}, file={self.file_name})>"


# class AIA211(Base):
#     __tablename__ = 'aia_211'
    
#     id = Column(Integer, primary_key=True)
#     date_rounded = Column(DateTime, nullable=False)
#     date = Column(DateTime, unique=True, nullable=False)
#     file_name = Column(String(255), nullable=False)
#     quality = Column(Integer, default=0)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
#     def __repr__(self):
#         return f"<AIA211(date={self.date}, file={self.file_name})>"


# class HMIMagnetogram(Base):
#     __tablename__ = 'hmi_magnetogram'
    
#     id = Column(Integer, primary_key=True)
#     date_rounded = Column(DateTime, nullable=False)
#     date = Column(DateTime, unique=True, nullable=False)
#     file_name = Column(String(255), nullable=False)
#     quality = Column(Integer, default=0)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
#     def __repr__(self):
#         return f"<HMIMagnetogram(date={self.date}, file={self.file_name})>"


# # 테이블명 → ORM 클래스 매핑
# TABLE_MODELS = {
#     'aia_193': AIA193,
#     'aia_211': AIA211,
#     'hmi_magnetogram': HMIMagnetogram
# }

# # ==========================================================
# # 세션 관리
# # ==========================================================

# @contextmanager
# def get_session():
#     """
#     안전한 세션 관리 (Context Manager)
    
#     사용법:
#         with get_session() as session:
#             result = session.query(...).all()
#     """
#     session = Session()
#     try:
#         yield session
#         session.commit()
#     except Exception as e:
#         session.rollback()
#         logger.error(f"Session error: {e}")
#         raise
#     finally:
#         session.close()


# def get_engine():
#     """SQLAlchemy 엔진 반환"""
#     return engine


# # ==========================================================
# # 데이터베이스 생성 및 관리
# # ==========================================================

# def create_database(db_name='sdo_data'):
#     """
#     새 데이터베이스 생성
    
#     Args:
#         db_name: 데이터베이스 이름
#     """
#     try:
#         # postgres 데이터베이스에 연결
#         temp_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres"
#         temp_engine = create_engine(temp_url, isolation_level='AUTOCOMMIT')
        
#         with temp_engine.connect() as conn:
#             # 데이터베이스 존재 확인
#             result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
#             exists = result.fetchone() is not None
            
#             if not exists:
#                 conn.execute(text(f"CREATE DATABASE {db_name}"))
#                 logger.info(f"Database '{db_name}' created successfully")
#             else:
#                 logger.info(f"Database '{db_name}' already exists")
        
#         temp_engine.dispose()
        
#     except SQLAlchemyError as e:
#         logger.error(f"Error creating database: {e}")
#         raise


# def create_tables():
#     """
#     모든 테이블 생성
#     """
#     try:
#         Base.metadata.create_all(engine)
#         logger.info("All tables created successfully")
#     except SQLAlchemyError as e:
#         logger.error(f"Error creating tables: {e}")
#         raise


# def drop_tables():
#     """
#     모든 테이블 삭제 (주의!)
#     """
#     try:
#         Base.metadata.drop_all(engine)
#         logger.info("All tables dropped successfully")
#     except SQLAlchemyError as e:
#         logger.error(f"Error dropping tables: {e}")
#         raise


# def get_table_info(table_name):
#     """
#     테이블 정보 조회
    
#     Args:
#         table_name: 테이블명
    
#     Returns:
#         dict: 테이블 정보
#     """
#     try:
#         with get_session() as session:
#             model = TABLE_MODELS.get(table_name)
#             if not model:
#                 raise ValueError(f"Unknown table: {table_name}")
            
#             # 레코드 수
#             count = session.query(model).count()
            
#             # 시간 범위
#             min_max = session.query(
#                 func.min(model.date_rounded),
#                 func.max(model.date_rounded)
#             ).first()
            
#             return {
#                 'table_name': table_name,
#                 'record_count': count,
#                 'min_date': min_max[0],
#                 'max_date': min_max[1],
#                 'days': (min_max[1] - min_max[0]).days if min_max[0] and min_max[1] else 0
#             }
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error getting table info: {e}")
#         raise


# # ==========================================================
# # 데이터 조회 (Core 방식)
# # ==========================================================

# def query_by_exact_time(table_name, date_rounded, return_type='dataframe'):
#     """
#     특정 시간의 레코드 조회
    
#     Args:
#         table_name: 테이블명 ('aia_193', 'aia_211', 'hmi_magnetogram')
#         date_rounded: 조회할 시간 (datetime 또는 문자열)
#         return_type: 'dataframe' or 'dict' or 'list'
    
#     Returns:
#         DataFrame, dict, 또는 list
#     """
#     if isinstance(date_rounded, str):
#         date_rounded = datetime.fromisoformat(date_rounded)
    
#     try:
#         query = text(f"""
#             SELECT id, date_rounded, date, file_name, quality, created_at, updated_at
#             FROM {table_name}
#             WHERE date_rounded = :date_rounded
#             ORDER BY date
#         """)
        
#         if return_type == 'dataframe':
#             df = pd.read_sql_query(query, engine, params={'date_rounded': date_rounded})
#             return df
        
#         else:
#             with engine.connect() as conn:
#                 result = conn.execute(query, {'date_rounded': date_rounded})
                
#                 if return_type == 'dict':
#                     return [dict(row._mapping) for row in result]
#                 elif return_type == 'list':
#                     return [tuple(row) for row in result]
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error querying by exact time: {e}")
#         raise


# def query_by_time_range(table_name, start_time, end_time, return_type='dataframe'):
#     """
#     시간 범위로 레코드 조회
    
#     Args:
#         table_name: 테이블명
#         start_time: 시작 시간 (datetime 또는 문자열)
#         end_time: 종료 시간 (datetime 또는 문자열)
#         return_type: 'dataframe' or 'dict' or 'list'
    
#     Returns:
#         DataFrame, dict, 또는 list
#     """
#     if isinstance(start_time, str):
#         start_time = datetime.fromisoformat(start_time)
#     if isinstance(end_time, str):
#         end_time = datetime.fromisoformat(end_time)
    
#     try:
#         query = text(f"""
#             SELECT id, date_rounded, date, file_name, quality, created_at, updated_at
#             FROM {table_name}
#             WHERE date_rounded >= :start_time AND date_rounded <= :end_time
#             ORDER BY date_rounded, date
#         """)
        
#         if return_type == 'dataframe':
#             df = pd.read_sql_query(
#                 query, 
#                 engine, 
#                 params={'start_time': start_time, 'end_time': end_time}
#             )
#             return df
        
#         else:
#             with engine.connect() as conn:
#                 result = conn.execute(query, {'start_time': start_time, 'end_time': end_time})
                
#                 if return_type == 'dict':
#                     return [dict(row._mapping) for row in result]
#                 elif return_type == 'list':
#                     return [tuple(row) for row in result]
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error querying by time range: {e}")
#         raise


# def query_synchronized_records(start_time, end_time, return_type='dataframe'):
#     """
#     모든 파장이 있는 레코드만 조회 (INNER JOIN)
    
#     Args:
#         start_time: 시작 시간
#         end_time: 종료 시간
#         return_type: 'dataframe' or 'dict' or 'list'
    
#     Returns:
#         DataFrame, dict, 또는 list
#     """
#     if isinstance(start_time, str):
#         start_time = datetime.fromisoformat(start_time)
#     if isinstance(end_time, str):
#         end_time = datetime.fromisoformat(end_time)
    
#     try:
#         query = text("""
#             SELECT 
#                 a193.date_rounded,
#                 a193.date as aia_193_date,
#                 a193.file_name as aia_193_file,
#                 a211.date as aia_211_date,
#                 a211.file_name as aia_211_file,
#                 hmi.date as hmi_date,
#                 hmi.file_name as hmi_file
#             FROM aia_193 a193
#             INNER JOIN aia_211 a211 ON a193.date_rounded = a211.date_rounded
#             INNER JOIN hmi_magnetogram hmi ON a193.date_rounded = hmi.date_rounded
#             WHERE a193.date_rounded >= :start_time AND a193.date_rounded <= :end_time
#             ORDER BY a193.date_rounded
#         """)
        
#         if return_type == 'dataframe':
#             df = pd.read_sql_query(
#                 query, 
#                 engine, 
#                 params={'start_time': start_time, 'end_time': end_time}
#             )
#             return df
        
#         else:
#             with engine.connect() as conn:
#                 result = conn.execute(query, {'start_time': start_time, 'end_time': end_time})
                
#                 if return_type == 'dict':
#                     return [dict(row._mapping) for row in result]
#                 elif return_type == 'list':
#                     return [tuple(row) for row in result]
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error querying synchronized records: {e}")
#         raise


# # ==========================================================
# # 데이터 조회 (ORM 방식)
# # ==========================================================

# def query_by_orm(table_name, filters=None, limit=None, offset=None):
#     """
#     ORM으로 유연한 조회
    
#     Args:
#         table_name: 테이블명
#         filters: 필터 딕셔너리 예) {'quality': 0, 'date_rounded__gte': datetime(...)}
#         limit: 제한 개수
#         offset: 오프셋
    
#     Returns:
#         list of ORM objects
    
#     Example:
#         results = query_by_orm('aia_193', 
#                                filters={'quality': 0, 'date_rounded__gte': datetime(2014, 4, 11)},
#                                limit=10)
#     """
#     try:
#         with get_session() as session:
#             model = TABLE_MODELS.get(table_name)
#             if not model:
#                 raise ValueError(f"Unknown table: {table_name}")
            
#             query = session.query(model)
            
#             # 필터 적용
#             if filters:
#                 for key, value in filters.items():
#                     if '__gte' in key:  # Greater Than or Equal
#                         field = key.replace('__gte', '')
#                         query = query.filter(getattr(model, field) >= value)
#                     elif '__lte' in key:  # Less Than or Equal
#                         field = key.replace('__lte', '')
#                         query = query.filter(getattr(model, field) <= value)
#                     elif '__gt' in key:  # Greater Than
#                         field = key.replace('__gt', '')
#                         query = query.filter(getattr(model, field) > value)
#                     elif '__lt' in key:  # Less Than
#                         field = key.replace('__lt', '')
#                         query = query.filter(getattr(model, field) < value)
#                     else:
#                         query = query.filter(getattr(model, key) == value)
            
#             # Limit, Offset
#             if limit:
#                 query = query.limit(limit)
#             if offset:
#                 query = query.offset(offset)
            
#             return query.all()
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error in ORM query: {e}")
#         raise


# # ==========================================================
# # 파일 경로 관련
# # ==========================================================

# def get_file_path(table_name, date_value, data_root="/Users/eunsupark/Data/sdo/fits"):
#     """
#     date로 파일 경로 반환
    
#     Args:
#         table_name: 테이블명
#         date_value: date 값 (datetime 또는 문자열)
#         data_root: 데이터 루트 경로
    
#     Returns:
#         str: 파일 전체 경로 또는 None
#     """
#     if isinstance(date_value, str):
#         date_value = datetime.fromisoformat(date_value)
    
#     try:
#         query = text(f"""
#             SELECT file_name
#             FROM {table_name}
#             WHERE date = :date_value
#         """)
        
#         with engine.connect() as conn:
#             result = conn.execute(query, {'date_value': date_value})
#             row = result.fetchone()
            
#             if row:
#                 file_name = row[0]
#                 instrument = 'aia' if table_name.startswith('aia') else 'hmi'
#                 return f"{data_root}/{instrument}/{file_name}"
#             else:
#                 return None
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error getting file path: {e}")
#         raise


# def get_file_paths_bulk(table_name, start_time, end_time, data_root="/Users/eunsupark/Data/sdo/fits"):
#     """
#     시간 범위의 파일 경로 리스트 반환
    
#     Args:
#         table_name: 테이블명
#         start_time: 시작 시간
#         end_time: 종료 시간
#         data_root: 데이터 루트 경로
    
#     Returns:
#         list of str: 파일 경로 리스트
#     """
#     if isinstance(start_time, str):
#         start_time = datetime.fromisoformat(start_time)
#     if isinstance(end_time, str):
#         end_time = datetime.fromisoformat(end_time)
    
#     try:
#         query = text(f"""
#             SELECT file_name
#             FROM {table_name}
#             WHERE date_rounded >= :start_time AND date_rounded <= :end_time
#             ORDER BY date_rounded
#         """)
        
#         with engine.connect() as conn:
#             result = conn.execute(query, {'start_time': start_time, 'end_time': end_time})
#             file_names = [row[0] for row in result]
        
#         instrument = 'aia' if table_name.startswith('aia') else 'hmi'
#         return [f"{data_root}/{instrument}/{fname}" for fname in file_names]
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error getting file paths: {e}")
#         raise


# # ==========================================================
# # 통계 및 분석
# # ==========================================================

# def count_records(table_name, start_time=None, end_time=None):
#     """
#     레코드 개수 세기
    
#     Args:
#         table_name: 테이블명
#         start_time: 시작 시간 (선택)
#         end_time: 종료 시간 (선택)
    
#     Returns:
#         int: 레코드 개수
#     """
#     try:
#         with get_session() as session:
#             model = TABLE_MODELS.get(table_name)
#             if not model:
#                 raise ValueError(f"Unknown table: {table_name}")
            
#             query = session.query(func.count(model.id))
            
#             if start_time and end_time:
#                 if isinstance(start_time, str):
#                     start_time = datetime.fromisoformat(start_time)
#                 if isinstance(end_time, str):
#                     end_time = datetime.fromisoformat(end_time)
                
#                 query = query.filter(
#                     model.date_rounded >= start_time,
#                     model.date_rounded <= end_time
#                 )
            
#             return query.scalar()
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error counting records: {e}")
#         raise


# def get_time_range(table_name):
#     """
#     테이블의 시간 범위 조회
    
#     Args:
#         table_name: 테이블명
    
#     Returns:
#         tuple: (min_date, max_date)
#     """
#     try:
#         with get_session() as session:
#             model = TABLE_MODELS.get(table_name)
#             if not model:
#                 raise ValueError(f"Unknown table: {table_name}")
            
#             result = session.query(
#                 func.min(model.date_rounded),
#                 func.max(model.date_rounded)
#             ).first()
            
#             return result
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error getting time range: {e}")
#         raise


# def find_missing_hours(table_name, start_time, end_time):
#     """
#     누락된 시간대 찾기 (1시간 간격 기준)
    
#     Args:
#         table_name: 테이블명
#         start_time: 시작 시간
#         end_time: 종료 시간
    
#     Returns:
#         list of datetime: 누락된 시간대
#     """
#     if isinstance(start_time, str):
#         start_time = datetime.fromisoformat(start_time)
#     if isinstance(end_time, str):
#         end_time = datetime.fromisoformat(end_time)
    
#     try:
#         with get_session() as session:
#             model = TABLE_MODELS.get(table_name)
#             if not model:
#                 raise ValueError(f"Unknown table: {table_name}")
            
#             # 실제 있는 시간대
#             actual = session.query(model.date_rounded).filter(
#                 model.date_rounded >= start_time,
#                 model.date_rounded <= end_time
#             ).distinct().all()
            
#             actual_times = {row[0] for row in actual}
        
#         # 기대하는 시간대
#         expected_times = []
#         current = start_time
#         while current <= end_time:
#             expected_times.append(current)
#             current += timedelta(hours=1)
        
#         # 누락된 시간대
#         missing = [t for t in expected_times if t not in actual_times]
#         return missing
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error finding missing hours: {e}")
#         raise


# # ==========================================================
# # 데이터 삽입 및 업데이트
# # ==========================================================

# def insert_record(table_name, data):
#     """
#     단일 레코드 삽입
    
#     Args:
#         table_name: 테이블명
#         data: 딕셔너리 {'date_rounded': ..., 'date': ..., 'file_name': ..., 'quality': ...}
    
#     Returns:
#         int: 삽입된 레코드의 id
#     """
#     try:
#         with get_session() as session:
#             model = TABLE_MODELS.get(table_name)
#             if not model:
#                 raise ValueError(f"Unknown table: {table_name}")
            
#             record = model(**data)
#             session.add(record)
#             session.flush()
            
#             return record.id
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error inserting record: {e}")
#         raise


# def bulk_insert(table_name, data_list):
#     """
#     대량 레코드 삽입
    
#     Args:
#         table_name: 테이블명
#         data_list: 딕셔너리 리스트
    
#     Returns:
#         int: 삽입된 레코드 수
#     """
#     try:
#         with get_session() as session:
#             model = TABLE_MODELS.get(table_name)
#             if not model:
#                 raise ValueError(f"Unknown table: {table_name}")
            
#             records = [model(**data) for data in data_list]
#             session.bulk_save_objects(records)
            
#             return len(records)
    
#     except SQLAlchemyError as e:
#         logger.error(f"Error bulk inserting: {e}")
#         raise


# # ==========================================================
# # 사용 예시 (테스트용)
# # ==========================================================

# if __name__ == "__main__":
    
#     print("="*60)
#     print("utils_database.py 테스트")
#     print("="*60)
    
#     # 테이블 정보
#     print("\n테이블 정보:")
#     for table in ['aia_193', 'aia_211', 'hmi_magnetogram']:
#         info = get_table_info(table)
#         print(f"\n{table}:")
#         print(f"  레코드 수: {info['record_count']:,}")
#         print(f"  시작: {info['min_date']}")
#         print(f"  종료: {info['max_date']}")
#         print(f"  기간: {info['days']} 일")
    
#     # 특정 시간 조회
#     print("\n특정 시간 조회:")
#     df = query_by_exact_time('aia_193', '2014-04-11 11:00:00')
#     print(df)
    
#     # 시간 범위 조회
#     print("\n시간 범위 조회:")
#     df = query_by_time_range('aia_193', '2014-04-11 10:00:00', '2014-04-11 12:00:00')
#     print(f"조회 결과: {len(df)} 레코드")
    
#     # 동기화된 레코드
#     print("\n동기화된 레코드:")
#     df = query_synchronized_records('2014-04-11 10:00:00', '2014-04-11 12:00:00')
#     print(f"동기화된 레코드: {len(df)} 개")
    
#     print("\n테스트 완료!")
