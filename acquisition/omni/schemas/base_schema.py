"""
공통 스키마 유틸리티

Common schema utilities for OMNI tables
"""

def get_time_columns_schema(include_minute=False):
    """
    시간 컬럼 스키마 반환
    
    Args:
        include_minute: True면 Minute 컬럼 포함 (High Res)
    
    Returns:
        dict: 시간 컬럼 스키마
    """
    schema = {
        'datetime': 'TIMESTAMP PRIMARY KEY',
        'year': 'INTEGER NOT NULL',
        'day': 'INTEGER NOT NULL',  # Low: Decimal_Day, High: Day
        'hour': 'INTEGER NOT NULL',
    }
    
    if include_minute:
        schema['minute'] = 'INTEGER NOT NULL'
    
    return schema


def get_data_type(value_range='normal', is_float=False):
    """
    데이터 타입 헬퍼 함수
    
    Args:
        value_range: 'small', 'normal', 'large'
        is_float: True면 실수형 반환
    
    Returns:
        str: PostgreSQL 데이터 타입
    """
    if is_float:
        return 'REAL'  # Float (4 bytes, ~6-7 significant digits)
    
    if value_range == 'small':
        return 'SMALLINT'  # -32768 ~ 32767
    elif value_range == 'normal':
        return 'INTEGER'   # -2,147,483,648 ~ 2,147,483,647
    elif value_range == 'large':
        return 'BIGINT'    # Very large values
    else:
        return 'REAL'


def create_index_query(table_name, index_name, columns):
    """
    인덱스 생성 쿼리 생성
    
    Args:
        table_name: 테이블 이름
        index_name: 인덱스 이름
        columns: 컬럼 이름 리스트 또는 문자열
    
    Returns:
        str: CREATE INDEX 쿼리
    """
    if isinstance(columns, list):
        columns_str = ', '.join(columns)
    else:
        columns_str = columns
    
    return f"CREATE INDEX {index_name} ON {table_name}({columns_str})"