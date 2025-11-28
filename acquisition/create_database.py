"""
create_database.py - utils_database.py 활용 버전
"""

from utils_database import (
    create_database,
    create_tables,
    drop_tables,
    get_table_info,
    TABLE_MODELS
)
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def show_table_info():
    """테이블 정보 출력"""
    
    table_names = ['aia_193', 'aia_211', 'hmi_magnetogram']
    
    logging.info("\n테이블 정보:")
    for table_name in table_names:
        try:
            info = get_table_info(table_name)
            
            logging.info(f"\n{table_name}:")
            logging.info(f"  총 레코드:  {info['record_count']:,}")
            logging.info(f"  시작 날짜:  {info['min_date']}")
            logging.info(f"  종료 날짜:  {info['max_date']}")
            logging.info(f"  총 기간:    {info['days']} 일")
            
            if info['days'] > 0:
                avg_per_day = info['record_count'] / info['days']
                logging.info(f"  일평균:     {avg_per_day:.1f}")
        
        except Exception as e:
            logging.warning(f"{table_name}: 테이블이 존재하지 않거나 비어있음")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--drop':
        # 테이블 삭제
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == 'yes':
            try:
                drop_tables()
                logging.info("All tables dropped successfully!")
            except Exception as e:
                logging.error(f"Error dropping tables: {e}")
        else:
            logging.info("Drop operation cancelled")
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--info':
        # 테이블 정보만 출력
        show_table_info()
    
    else:
        # 데이터베이스 및 테이블 생성
        try:
            # 데이터베이스 생성 (이미 있으면 스킵)
            logging.info("Checking/Creating database...")
            create_database('sdo_data')
            
            # 테이블 생성 (이미 있으면 스킵)
            logging.info("Creating tables...")
            create_tables()
            
            # 생성된 테이블 정보 출력
            show_table_info()
            
        except Exception as e:
            logging.error(f"Error: {e}")
            sys.exit(1)

# import psycopg2
# from psycopg2 import sql
# import logging

# # 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # 데이터베이스 연결 설정
# DB_CONFIG = {
#     'dbname': 'sdo_data',
#     'user': 'eunsupark',  # 본인의 macOS 사용자명으로 변경
#     'password': 'eunsupark',  # 비밀번호가 있으면 입력
#     'host': 'localhost',
#     'port': '5432'
# }

# def create_tables():
#     """PostgreSQL 테이블 생성"""
    
#     # 테이블 정의
#     tables = {
#         'aia_193': """
#             CREATE TABLE IF NOT EXISTS aia_193 (
#                 id SERIAL PRIMARY KEY,
#                 date TIMESTAMP NOT NULL UNIQUE,
#                 date_rounded TIMESTAMP NOT NULL,
#                 file_name VARCHAR(255) NOT NULL,
#                 quality INTEGER DEFAULT 0,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#         """,
#         'aia_211': """
#             CREATE TABLE IF NOT EXISTS aia_211 (
#                 id SERIAL PRIMARY KEY,
#                 date TIMESTAMP NOT NULL UNIQUE,
#                 date_rounded TIMESTAMP NOT NULL,
#                 file_name VARCHAR(255) NOT NULL,
#                 quality INTEGER DEFAULT 0,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#         """,
#         'hmi_magnetogram': """
#             CREATE TABLE IF NOT EXISTS hmi_magnetogram (
#                 id SERIAL PRIMARY KEY,
#                 date TIMESTAMP NOT NULL UNIQUE,
#                 date_rounded TIMESTAMP NOT NULL,
#                 file_name VARCHAR(255) NOT NULL,
#                 quality INTEGER DEFAULT 0,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#         """
#     }
    
#     # 인덱스 정의
#     indexes = {
#         'aia_193': "CREATE INDEX IF NOT EXISTS idx_aia_193_datetime ON aia_193(date);",
#         'aia_211': "CREATE INDEX IF NOT EXISTS idx_aia_211_datetime ON aia_211(date);",
#         'hmi_magnetogram': "CREATE INDEX IF NOT EXISTS idx_hmi_magnetogram_datetime ON hmi_magnetogram(date);"
#     }
    
#     try:
#         # 데이터베이스 연결
#         logging.info("Connecting to PostgreSQL database...")
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor()
        
#         # 테이블 생성
#         for table_name, create_sql in tables.items():
#             logging.info(f"Creating table: {table_name}")
#             cursor.execute(create_sql)
            
#             # 인덱스 생성
#             logging.info(f"Creating index for: {table_name}")
#             cursor.execute(indexes[table_name])
        
#         # 변경사항 커밋
#         conn.commit()
#         logging.info("All tables and indexes created successfully!")
        
#         # 생성된 테이블 확인
#         cursor.execute("""
#             SELECT tablename 
#             FROM pg_tables 
#             WHERE schemaname = 'public'
#             ORDER BY tablename;
#         """)
#         tables_list = cursor.fetchall()
        
#         logging.info("Existing tables:")
#         for table in tables_list:
#             logging.info(f"  - {table[0]}")
        
#         # 각 테이블의 구조 확인
#         for table_name in tables.keys():
#             cursor.execute(f"""
#                 SELECT column_name, data_type, character_maximum_length
#                 FROM information_schema.columns
#                 WHERE table_name = '{table_name}'
#                 ORDER BY ordinal_position;
#             """)
#             columns = cursor.fetchall()
            
#             logging.info(f"\nTable structure for {table_name}:")
#             for col in columns:
#                 col_name, data_type, max_length = col
#                 if max_length:
#                     logging.info(f"  - {col_name}: {data_type}({max_length})")
#                 else:
#                     logging.info(f"  - {col_name}: {data_type}")
        
#     except psycopg2.Error as e:
#         logging.error(f"Database error: {e}")
#         if conn:
#             conn.rollback()
#         raise
    
#     finally:
#         # 연결 종료
#         if cursor:
#             cursor.close()
#         if conn:
#             conn.close()
#         logging.info("Database connection closed")


# def drop_tables():
#     """테이블 삭제 (필요시 사용)"""
    
#     try:
#         logging.info("Connecting to PostgreSQL database...")
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor()
        
#         tables = ['aia_193', 'aia_211', 'hmi_magnetogram']
        
#         for table_name in tables:
#             logging.info(f"Dropping table: {table_name}")
#             cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        
#         conn.commit()
#         logging.info("All tables dropped successfully!")
        
#     except psycopg2.Error as e:
#         logging.error(f"Database error: {e}")
#         if conn:
#             conn.rollback()
#         raise
    
#     finally:
#         if cursor:
#             cursor.close()
#         if conn:
#             conn.close()
#         logging.info("Database connection closed")


# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1 and sys.argv[1] == '--drop':
#         # 테이블 삭제
#         confirm = input("Are you sure you want to drop all tables? (yes/no): ")
#         if confirm.lower() == 'yes':
#             drop_tables()
#         else:
#             logging.info("Drop operation cancelled")
#     else:
#         # 테이블 생성
#         create_tables()