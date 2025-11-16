"""
utils_database.py

SDO 데이터베이스 작업을 위한 유틸리티 함수 모음
SQLAlchemy 기반
"""

from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# 설정
# ==========================================================

DB_CONFIG = {
    'dbname': 'sdo_data',
    'user': 'eunsupark',
    'password': 'eunsupark',
    'host': 'localhost',
    'port': '5432'
}

# SQLAlchemy 엔진 생성
DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)

# ORM Base
Base = declarative_base()
Session = sessionmaker(bind=engine)

# ==========================================================
# ORM 모델 정의
# ==========================================================

class AIA193(Base):
    __tablename__ = 'aia_193'
    
    id = Column(Integer, primary_key=True)
    date_rounded = Column(DateTime, nullable=False)
    date = Column(DateTime, unique=True, nullable=False)
    file_name = Column(String(255), nullable=False)
    quality = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AIA193(date={self.date}, file={self.file_name})>"


class AIA211(Base):
    __tablename__ = 'aia_211'
    
    id = Column(Integer, primary_key=True)
    date_rounded = Column(DateTime, nullable=False)
    date = Column(DateTime, unique=True, nullable=False)
    file_name = Column(String(255), nullable=False)
    quality = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AIA211(date={self.date}, file={self.file_name})>"


class HMIMagnetogram(Base):
    __tablename__ = 'hmi_magnetogram'
    
    id = Column(Integer, primary_key=True)
    date_rounded = Column(DateTime, nullable=False)
    date = Column(DateTime, unique=True, nullable=False)
    file_name = Column(String(255), nullable=False)
    quality = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<HMIMagnetogram(date={self.date}, file={self.file_name})>"


# 테이블명 → ORM 클래스 매핑
TABLE_MODELS = {
    'aia_193': AIA193,
    'aia_211': AIA211,
    'hmi_magnetogram': HMIMagnetogram
}

# ==========================================================
# 세션 관리
# ==========================================================

@contextmanager
def get_session():
    """
    안전한 세션 관리 (Context Manager)
    
    사용법:
        with get_session() as session:
            result = session.query(...).all()
    """
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Session error: {e}")
        raise
    finally:
        session.close()


def get_engine():
    """SQLAlchemy 엔진 반환"""
    return engine


# ==========================================================
# 데이터베이스 생성 및 관리
# ==========================================================

def create_database(db_name='sdo_data'):
    """
    새 데이터베이스 생성
    
    Args:
        db_name: 데이터베이스 이름
    """
    try:
        # postgres 데이터베이스에 연결
        temp_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres"
        temp_engine = create_engine(temp_url, isolation_level='AUTOCOMMIT')
        
        with temp_engine.connect() as conn:
            # 데이터베이스 존재 확인
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
            exists = result.fetchone() is not None
            
            if not exists:
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                logger.info(f"Database '{db_name}' created successfully")
            else:
                logger.info(f"Database '{db_name}' already exists")
        
        temp_engine.dispose()
        
    except SQLAlchemyError as e:
        logger.error(f"Error creating database: {e}")
        raise


def create_tables():
    """
    모든 테이블 생성
    """
    try:
        Base.metadata.create_all(engine)
        logger.info("All tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating tables: {e}")
        raise


def drop_tables():
    """
    모든 테이블 삭제 (주의!)
    """
    try:
        Base.metadata.drop_all(engine)
        logger.info("All tables dropped successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error dropping tables: {e}")
        raise


def get_table_info(table_name):
    """
    테이블 정보 조회
    
    Args:
        table_name: 테이블명
    
    Returns:
        dict: 테이블 정보
    """
    try:
        with get_session() as session:
            model = TABLE_MODELS.get(table_name)
            if not model:
                raise ValueError(f"Unknown table: {table_name}")
            
            # 레코드 수
            count = session.query(model).count()
            
            # 시간 범위
            min_max = session.query(
                func.min(model.date_rounded),
                func.max(model.date_rounded)
            ).first()
            
            return {
                'table_name': table_name,
                'record_count': count,
                'min_date': min_max[0],
                'max_date': min_max[1],
                'days': (min_max[1] - min_max[0]).days if min_max[0] and min_max[1] else 0
            }
    
    except SQLAlchemyError as e:
        logger.error(f"Error getting table info: {e}")
        raise


# ==========================================================
# 데이터 조회 (Core 방식)
# ==========================================================

def query_by_exact_time(table_name, date_rounded, return_type='dataframe'):
    """
    특정 시간의 레코드 조회
    
    Args:
        table_name: 테이블명 ('aia_193', 'aia_211', 'hmi_magnetogram')
        date_rounded: 조회할 시간 (datetime 또는 문자열)
        return_type: 'dataframe' or 'dict' or 'list'
    
    Returns:
        DataFrame, dict, 또는 list
    """
    if isinstance(date_rounded, str):
        date_rounded = datetime.fromisoformat(date_rounded)
    
    try:
        query = text(f"""
            SELECT id, date_rounded, date, file_name, quality, created_at, updated_at
            FROM {table_name}
            WHERE date_rounded = :date_rounded
            ORDER BY date
        """)
        
        if return_type == 'dataframe':
            df = pd.read_sql_query(query, engine, params={'date_rounded': date_rounded})
            return df
        
        else:
            with engine.connect() as conn:
                result = conn.execute(query, {'date_rounded': date_rounded})
                
                if return_type == 'dict':
                    return [dict(row._mapping) for row in result]
                elif return_type == 'list':
                    return [tuple(row) for row in result]
    
    except SQLAlchemyError as e:
        logger.error(f"Error querying by exact time: {e}")
        raise


def query_by_time_range(table_name, start_time, end_time, return_type='dataframe'):
    """
    시간 범위로 레코드 조회
    
    Args:
        table_name: 테이블명
        start_time: 시작 시간 (datetime 또는 문자열)
        end_time: 종료 시간 (datetime 또는 문자열)
        return_type: 'dataframe' or 'dict' or 'list'
    
    Returns:
        DataFrame, dict, 또는 list
    """
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time)
    
    try:
        query = text(f"""
            SELECT id, date_rounded, date, file_name, quality, created_at, updated_at
            FROM {table_name}
            WHERE date_rounded >= :start_time AND date_rounded <= :end_time
            ORDER BY date_rounded, date
        """)
        
        if return_type == 'dataframe':
            df = pd.read_sql_query(
                query, 
                engine, 
                params={'start_time': start_time, 'end_time': end_time}
            )
            return df
        
        else:
            with engine.connect() as conn:
                result = conn.execute(query, {'start_time': start_time, 'end_time': end_time})
                
                if return_type == 'dict':
                    return [dict(row._mapping) for row in result]
                elif return_type == 'list':
                    return [tuple(row) for row in result]
    
    except SQLAlchemyError as e:
        logger.error(f"Error querying by time range: {e}")
        raise


def query_synchronized_records(start_time, end_time, return_type='dataframe'):
    """
    모든 파장이 있는 레코드만 조회 (INNER JOIN)
    
    Args:
        start_time: 시작 시간
        end_time: 종료 시간
        return_type: 'dataframe' or 'dict' or 'list'
    
    Returns:
        DataFrame, dict, 또는 list
    """
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time)
    
    try:
        query = text("""
            SELECT 
                a193.date_rounded,
                a193.date as aia_193_date,
                a193.file_name as aia_193_file,
                a211.date as aia_211_date,
                a211.file_name as aia_211_file,
                hmi.date as hmi_date,
                hmi.file_name as hmi_file
            FROM aia_193 a193
            INNER JOIN aia_211 a211 ON a193.date_rounded = a211.date_rounded
            INNER JOIN hmi_magnetogram hmi ON a193.date_rounded = hmi.date_rounded
            WHERE a193.date_rounded >= :start_time AND a193.date_rounded <= :end_time
            ORDER BY a193.date_rounded
        """)
        
        if return_type == 'dataframe':
            df = pd.read_sql_query(
                query, 
                engine, 
                params={'start_time': start_time, 'end_time': end_time}
            )
            return df
        
        else:
            with engine.connect() as conn:
                result = conn.execute(query, {'start_time': start_time, 'end_time': end_time})
                
                if return_type == 'dict':
                    return [dict(row._mapping) for row in result]
                elif return_type == 'list':
                    return [tuple(row) for row in result]
    
    except SQLAlchemyError as e:
        logger.error(f"Error querying synchronized records: {e}")
        raise


# ==========================================================
# 데이터 조회 (ORM 방식)
# ==========================================================

def query_by_orm(table_name, filters=None, limit=None, offset=None):
    """
    ORM으로 유연한 조회
    
    Args:
        table_name: 테이블명
        filters: 필터 딕셔너리 예) {'quality': 0, 'date_rounded__gte': datetime(...)}
        limit: 제한 개수
        offset: 오프셋
    
    Returns:
        list of ORM objects
    
    Example:
        results = query_by_orm('aia_193', 
                               filters={'quality': 0, 'date_rounded__gte': datetime(2014, 4, 11)},
                               limit=10)
    """
    try:
        with get_session() as session:
            model = TABLE_MODELS.get(table_name)
            if not model:
                raise ValueError(f"Unknown table: {table_name}")
            
            query = session.query(model)
            
            # 필터 적용
            if filters:
                for key, value in filters.items():
                    if '__gte' in key:  # Greater Than or Equal
                        field = key.replace('__gte', '')
                        query = query.filter(getattr(model, field) >= value)
                    elif '__lte' in key:  # Less Than or Equal
                        field = key.replace('__lte', '')
                        query = query.filter(getattr(model, field) <= value)
                    elif '__gt' in key:  # Greater Than
                        field = key.replace('__gt', '')
                        query = query.filter(getattr(model, field) > value)
                    elif '__lt' in key:  # Less Than
                        field = key.replace('__lt', '')
                        query = query.filter(getattr(model, field) < value)
                    else:
                        query = query.filter(getattr(model, key) == value)
            
            # Limit, Offset
            if limit:
                query = query.limit(limit)
            if offset:
                query = query.offset(offset)
            
            return query.all()
    
    except SQLAlchemyError as e:
        logger.error(f"Error in ORM query: {e}")
        raise


# ==========================================================
# 파일 경로 관련
# ==========================================================

def get_file_path(table_name, date_value, data_root="/Users/eunsupark/Data/sdo/fits"):
    """
    date로 파일 경로 반환
    
    Args:
        table_name: 테이블명
        date_value: date 값 (datetime 또는 문자열)
        data_root: 데이터 루트 경로
    
    Returns:
        str: 파일 전체 경로 또는 None
    """
    if isinstance(date_value, str):
        date_value = datetime.fromisoformat(date_value)
    
    try:
        query = text(f"""
            SELECT file_name
            FROM {table_name}
            WHERE date = :date_value
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {'date_value': date_value})
            row = result.fetchone()
            
            if row:
                file_name = row[0]
                instrument = 'aia' if table_name.startswith('aia') else 'hmi'
                return f"{data_root}/{instrument}/{file_name}"
            else:
                return None
    
    except SQLAlchemyError as e:
        logger.error(f"Error getting file path: {e}")
        raise


def get_file_paths_bulk(table_name, start_time, end_time, data_root="/Users/eunsupark/Data/sdo/fits"):
    """
    시간 범위의 파일 경로 리스트 반환
    
    Args:
        table_name: 테이블명
        start_time: 시작 시간
        end_time: 종료 시간
        data_root: 데이터 루트 경로
    
    Returns:
        list of str: 파일 경로 리스트
    """
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time)
    
    try:
        query = text(f"""
            SELECT file_name
            FROM {table_name}
            WHERE date_rounded >= :start_time AND date_rounded <= :end_time
            ORDER BY date_rounded
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {'start_time': start_time, 'end_time': end_time})
            file_names = [row[0] for row in result]
        
        instrument = 'aia' if table_name.startswith('aia') else 'hmi'
        return [f"{data_root}/{instrument}/{fname}" for fname in file_names]
    
    except SQLAlchemyError as e:
        logger.error(f"Error getting file paths: {e}")
        raise


# ==========================================================
# 통계 및 분석
# ==========================================================

def count_records(table_name, start_time=None, end_time=None):
    """
    레코드 개수 세기
    
    Args:
        table_name: 테이블명
        start_time: 시작 시간 (선택)
        end_time: 종료 시간 (선택)
    
    Returns:
        int: 레코드 개수
    """
    try:
        with get_session() as session:
            model = TABLE_MODELS.get(table_name)
            if not model:
                raise ValueError(f"Unknown table: {table_name}")
            
            query = session.query(func.count(model.id))
            
            if start_time and end_time:
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time)
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time)
                
                query = query.filter(
                    model.date_rounded >= start_time,
                    model.date_rounded <= end_time
                )
            
            return query.scalar()
    
    except SQLAlchemyError as e:
        logger.error(f"Error counting records: {e}")
        raise


def get_time_range(table_name):
    """
    테이블의 시간 범위 조회
    
    Args:
        table_name: 테이블명
    
    Returns:
        tuple: (min_date, max_date)
    """
    try:
        with get_session() as session:
            model = TABLE_MODELS.get(table_name)
            if not model:
                raise ValueError(f"Unknown table: {table_name}")
            
            result = session.query(
                func.min(model.date_rounded),
                func.max(model.date_rounded)
            ).first()
            
            return result
    
    except SQLAlchemyError as e:
        logger.error(f"Error getting time range: {e}")
        raise


def find_missing_hours(table_name, start_time, end_time):
    """
    누락된 시간대 찾기 (1시간 간격 기준)
    
    Args:
        table_name: 테이블명
        start_time: 시작 시간
        end_time: 종료 시간
    
    Returns:
        list of datetime: 누락된 시간대
    """
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time)
    
    try:
        with get_session() as session:
            model = TABLE_MODELS.get(table_name)
            if not model:
                raise ValueError(f"Unknown table: {table_name}")
            
            # 실제 있는 시간대
            actual = session.query(model.date_rounded).filter(
                model.date_rounded >= start_time,
                model.date_rounded <= end_time
            ).distinct().all()
            
            actual_times = {row[0] for row in actual}
        
        # 기대하는 시간대
        expected_times = []
        current = start_time
        while current <= end_time:
            expected_times.append(current)
            current += timedelta(hours=1)
        
        # 누락된 시간대
        missing = [t for t in expected_times if t not in actual_times]
        return missing
    
    except SQLAlchemyError as e:
        logger.error(f"Error finding missing hours: {e}")
        raise


# ==========================================================
# 데이터 삽입 및 업데이트
# ==========================================================

def insert_record(table_name, data):
    """
    단일 레코드 삽입
    
    Args:
        table_name: 테이블명
        data: 딕셔너리 {'date_rounded': ..., 'date': ..., 'file_name': ..., 'quality': ...}
    
    Returns:
        int: 삽입된 레코드의 id
    """
    try:
        with get_session() as session:
            model = TABLE_MODELS.get(table_name)
            if not model:
                raise ValueError(f"Unknown table: {table_name}")
            
            record = model(**data)
            session.add(record)
            session.flush()
            
            return record.id
    
    except SQLAlchemyError as e:
        logger.error(f"Error inserting record: {e}")
        raise


def bulk_insert(table_name, data_list):
    """
    대량 레코드 삽입
    
    Args:
        table_name: 테이블명
        data_list: 딕셔너리 리스트
    
    Returns:
        int: 삽입된 레코드 수
    """
    try:
        with get_session() as session:
            model = TABLE_MODELS.get(table_name)
            if not model:
                raise ValueError(f"Unknown table: {table_name}")
            
            records = [model(**data) for data in data_list]
            session.bulk_save_objects(records)
            
            return len(records)
    
    except SQLAlchemyError as e:
        logger.error(f"Error bulk inserting: {e}")
        raise


# ==========================================================
# 사용 예시 (테스트용)
# ==========================================================

if __name__ == "__main__":
    
    print("="*60)
    print("utils_database.py 테스트")
    print("="*60)
    
    # 테이블 정보
    print("\n테이블 정보:")
    for table in ['aia_193', 'aia_211', 'hmi_magnetogram']:
        info = get_table_info(table)
        print(f"\n{table}:")
        print(f"  레코드 수: {info['record_count']:,}")
        print(f"  시작: {info['min_date']}")
        print(f"  종료: {info['max_date']}")
        print(f"  기간: {info['days']} 일")
    
    # 특정 시간 조회
    print("\n특정 시간 조회:")
    df = query_by_exact_time('aia_193', '2014-04-11 11:00:00')
    print(df)
    
    # 시간 범위 조회
    print("\n시간 범위 조회:")
    df = query_by_time_range('aia_193', '2014-04-11 10:00:00', '2014-04-11 12:00:00')
    print(f"조회 결과: {len(df)} 레코드")
    
    # 동기화된 레코드
    print("\n동기화된 레코드:")
    df = query_synchronized_records('2014-04-11 10:00:00', '2014-04-11 12:00:00')
    print(f"동기화된 레코드: {len(df)} 개")
    
    print("\n테스트 완료!")
