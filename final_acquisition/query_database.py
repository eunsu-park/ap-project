import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# 데이터베이스 연결 설정
DB_CONFIG = {
    'dbname': 'sdo_data',
    'user': 'eunsupark',
    'password': 'eunsupark',
    'host': 'localhost',
    'port': '5432'
}

# SQLAlchemy 엔진 생성
def get_engine():
    """SQLAlchemy 엔진 생성"""
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    return create_engine(connection_string)


def get_records_by_exact_time(table_name, date_rounded):
    """
    (1) date_rounded가 특정 값인 행을 추출
    
    Args:
        table_name: 테이블명 ('aia_193', 'aia_211', 'hmi_magnetogram')
        date_rounded: 찾을 시간 (datetime 객체 또는 문자열)
    
    Returns:
        pandas DataFrame
    """
    
    # 문자열을 datetime으로 변환
    if isinstance(date_rounded, str):
        date_rounded = datetime.fromisoformat(date_rounded)
    
    try:
        # SQLAlchemy 엔진 생성
        engine = get_engine()
        
        # 쿼리 실행 (text()로 감싸서 파라미터 바인딩)
        query = text(f"""
            SELECT id, date_rounded, date, file_name, quality, created_at, updated_at
            FROM {table_name}
            WHERE date_rounded = :date_rounded
            ORDER BY date
        """)
        
        # pandas로 읽기
        df = pd.read_sql_query(query, engine, params={'date_rounded': date_rounded})
        
        print(f"✓ Found {len(df)} record(s) for {date_rounded}")
        return df
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        return None


def get_records_by_time_range(table_name, start_time, end_time):
    """
    (2) date_rounded의 범위를 정해서 조건을 만족하는 행들을 추출
    
    Args:
        table_name: 테이블명 ('aia_193', 'aia_211', 'hmi_magnetogram')
        start_time: 시작 시간 (datetime 객체 또는 문자열)
        end_time: 종료 시간 (datetime 객체 또는 문자열)
    
    Returns:
        pandas DataFrame
    """
    
    # 문자열을 datetime으로 변환
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time)
    
    try:
        # SQLAlchemy 엔진 생성
        engine = get_engine()
        
        # 쿼리 실행
        query = text(f"""
            SELECT id, date_rounded, date, file_name, quality, created_at, updated_at
            FROM {table_name}
            WHERE date_rounded >= :start_time AND date_rounded <= :end_time
            ORDER BY date_rounded, date
        """)
        
        # pandas로 읽기
        df = pd.read_sql_query(query, engine, params={'start_time': start_time, 'end_time': end_time})
        
        print(f"✓ Found {len(df)} record(s) from {start_time} to {end_time}")
        return df
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        return None


def get_records_from_all_tables(start_time, end_time):
    """
    모든 테이블에서 시간 범위로 데이터 추출
    
    Args:
        start_time: 시작 시간
        end_time: 종료 시간
    
    Returns:
        dict: {'aia_193': df1, 'aia_211': df2, 'hmi_magnetogram': df3}
    """
    
    tables = ['aia_193', 'aia_211', 'hmi_magnetogram']
    results = {}
    
    for table in tables:
        print(f"\nQuerying {table}...")
        df = get_records_by_time_range(table, start_time, end_time)
        if df is not None:
            results[table] = df
    
    return results


def get_synchronized_records(start_time, end_time):
    """
    여러 테이블에서 동일한 date_rounded를 가진 레코드만 추출
    (모든 파장이 있는 시간대만 선택)
    
    Args:
        start_time: 시작 시간
        end_time: 종료 시간
    
    Returns:
        pandas DataFrame (merged)
    """
    
    # 문자열을 datetime으로 변환
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time)
    
    try:
        # SQLAlchemy 엔진 생성
        engine = get_engine()
        
        # 모든 테이블에서 공통 date_rounded 찾기
        query = text("""
            SELECT 
                a193.date_rounded,
                a193.file_name as aia_193_file,
                a193.date as aia_193_date,
                a211.file_name as aia_211_file,
                a211.date as aia_211_date,
                hmi.file_name as hmi_magnetogram_file,
                hmi.date as hmi_magnetogram_date
            FROM aia_193 a193
            INNER JOIN aia_211 a211 ON a193.date_rounded = a211.date_rounded
            INNER JOIN hmi_magnetogram hmi ON a193.date_rounded = hmi.date_rounded
            WHERE a193.date_rounded >= :start_time AND a193.date_rounded <= :end_time
            ORDER BY a193.date_rounded
        """)
        
        df = pd.read_sql_query(query, engine, params={'start_time': start_time, 'end_time': end_time})
        
        print(f"✓ Found {len(df)} synchronized record(s)")
        return df
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        return None


# ==========================================================
# 사용 예시
# ==========================================================

if __name__ == "__main__":


    date = datetime(year=2010, month=9, day=1)
    date_end = datetime(year=2025, month=1, day=1)

    table_names = ["aia_193", "aia_211", "hmi_magnetogram"]

    num = {"num_total":0, "num_aia_193":0, "num_aia_211":0, "num_hmi_magnetogram":0}

    while date < date_end :
        num["num_total"] += 1
        for table_name in table_names :
            df = get_records_by_exact_time(table_name, date)
            if df is not None and len(df) > 0:
                num[f"num_{table_name}"] += 1
                
                # print("\n결과:")
                # print(df)
                # print(f"\n파일명: {df.iloc[0]['file_name']}")



        date += timedelta(hours=1)

    print(num)


    # print("="*60)
    # print("예시 1: 특정 시간의 레코드 조회")
    # print("="*60)
    
    # # 특정 시간 조회
    # target_time = "2014-04-11 11:00:00"
    # df = get_records_by_exact_time('aia_211', target_time)
    
    # if df is not None and len(df) > 0:
    #     print("\n결과:")
    #     print(df)
    #     print(f"\n파일명: {df.iloc[0]['file_name']}")
    
    
    # print("\n" + "="*60)
    # print("예시 2: 시간 범위로 레코드 조회")
    # print("="*60)
    
    # # 시간 범위 조회
    # start = "2014-04-11 10:00:00"
    # end = "2014-04-11 12:00:00"
    # df = get_records_by_time_range('aia_193', start, end)
    
    # if df is not None and len(df) > 0:
    #     print("\n결과:")
    #     print(df[['date_rounded', 'date', 'file_name']])
    #     print(f"\n총 {len(df)}개 레코드")
    
    
    # print("\n" + "="*60)
    # print("예시 3: 모든 테이블에서 범위 조회")
    # print("="*60)
    
    # # 모든 테이블에서 조회
    # all_data = get_records_from_all_tables(start, end)
    
    # for table_name, df in all_data.items():
    #     if len(df) > 0:
    #         print(f"\n{table_name}: {len(df)} records")
    #         print(df[['date_rounded', 'file_name']].head())
    
    
    # print("\n" + "="*60)
    # print("예시 4: 동기화된 레코드만 조회 (모든 파장 존재)")
    # print("="*60)
    
    # # 모든 파장이 있는 시간대만 조회
    # df_sync = get_synchronized_records(start, end)
    
    # if df_sync is not None and len(df_sync) > 0:
    #     print("\n결과:")
    #     print(df_sync)
    #     print(f"\n총 {len(df_sync)}개의 완전한 세트")