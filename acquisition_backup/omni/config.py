"""
OMNI 데이터베이스 설정 파일

Database and table configurations for OMNI data
"""

# ============================================================================
# 데이터베이스 연결 설정
# ============================================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'omni',
    'user': 'eunsupark',
    'password': 'eunsupark',
    'log_queries': False
}

# ============================================================================
# 테이블별 설정
# ============================================================================

TABLE_CONFIGS = {
    'low_resolution': {
        'table_name': 'low_resolution',
        'url_pattern': 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat',
        'file_extension': '.dat',
        'time_resolution': 'hourly',
        'columns_count': 55,
        'description': 'OMNI2 Low Resolution (Hourly) data',
    },
    'high_resolution': {
        'table_name': 'high_resolution',
        'url_pattern': 'https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_min{year}.asc',
        'file_extension': '.asc',
        'time_resolution': 'minutely',
        'columns_count': 46,  # 1-min 기준 (5-min은 49개)
        'description': 'OMNI High Resolution (1-minute) data',
    }
}

# ============================================================================
# 공통 설정
# ============================================================================

# DB 배치 삽입 크기
BATCH_SIZE = 1000

# 다운로드 타임아웃 (초)
DOWNLOAD_TIMEOUT = 30

# 재시도 설정
MAX_RETRIES = 3
RETRY_DELAY = 5  # 초

# 로깅 설정
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'