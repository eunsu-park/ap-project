"""
OMNI 데이터베이스 테이블 생성 스크립트

Creates database and tables for OMNI Low and High Resolution data
"""
from egghouse.database import PostgresManager
from config import DB_CONFIG
from schemas.lowres_schema import get_lowres_schema, get_lowres_indices
from schemas.highres_schema import get_highres_schema, get_highres_indices


def create_omni_database():
    """
    omni 데이터베이스 생성
    
    기존 데이터베이스에 연결하여 omni 데이터베이스를 생성합니다.
    """
    print("\n" + "=" * 60)
    print("OMNI 데이터베이스 생성")
    print("=" * 60)
    
    # 먼저 omni DB에 직접 연결 시도
    try:
        with PostgresManager(**DB_CONFIG) as db:
            print("✓ omni 데이터베이스 이미 존재")
            return
    except:
        # omni DB가 없으면 생성 시도
        pass
    
    # 관리자 DB 연결 시도 (여러 기본 DB 시도)
    admin_databases = ['template1', 'sdo']
    
    for admin_db in admin_databases:
        try:
            admin_config = DB_CONFIG.copy()
            admin_config['database'] = admin_db
            
            print(f"관리자 DB 연결 시도: {admin_db}")
            
            with PostgresManager(**admin_config) as db:
                # 데이터베이스 존재 확인
                result = db.execute(
                    "SELECT 1 FROM pg_database WHERE datname = 'omni'",
                    fetch=True
                )
                
                if not result:
                    # 데이터베이스 생성
                    db.execute("CREATE DATABASE omni")
                    print("✓ omni 데이터베이스 생성 완료")
                else:
                    print("✓ omni 데이터베이스 이미 존재")
                
                return
                
        except Exception as e:
            print(f"  {admin_db} 연결 실패: {e}")
            continue
    
    # 모든 시도 실패
    print("\n⚠ 경고: omni 데이터베이스를 생성할 수 없습니다.")
    print("다음 방법으로 수동 생성해주세요:")
    print("  psql -U eunsupark -d sdo_data -c 'CREATE DATABASE omni;'")
    print("\n또는 기존 데이터베이스에 연결하여:")
    print("  CREATE DATABASE omni;")
    print("\n테이블 생성은 계속 진행됩니다...")
    print("(omni 데이터베이스가 이미 존재하면 무시하세요)")
    print("=" * 60)


def create_low_resolution_table(drop_if_exists=False):
    """
    Low Resolution 테이블 생성
    
    Args:
        drop_if_exists: True면 기존 테이블 삭제 후 재생성
    """
    print("\n" + "=" * 60)
    print("Low Resolution 테이블 생성")
    print("=" * 60)
    
    table_name = 'low_resolution'
    
    try:
        with PostgresManager(**DB_CONFIG) as db:
            # 기존 테이블 확인
            tables = db.list_tables(names_only=True)
            table_exists = table_name in tables
            
            if table_exists:
                if drop_if_exists:
                    print(f"⚠ 기존 테이블 삭제 중: {table_name}")
                    db.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                    print(f"✓ 기존 테이블 삭제 완료")
                else:
                    print(f"✓ 테이블 이미 존재: {table_name}")
                    print("  (drop_if_exists=True로 재생성 가능)")
                    return
            
            # 테이블 스키마 가져오기
            schema = get_lowres_schema()
            
            # 테이블 생성
            print(f"테이블 생성 중: {table_name} ({len(schema)} 컬럼)")
            db.create_table(table_name, schema)
            print(f"✓ 테이블 생성 완료")
            
            # 인덱스 생성
            print("인덱스 생성 중...")
            index_queries = get_lowres_indices()
            for i, query in enumerate(index_queries, 1):
                db.execute(query)
                print(f"  {i}. {query.split('INDEX')[1].split('ON')[0].strip()}")
            
            print(f"✓ {len(index_queries)}개 인덱스 생성 완료")
            
            # 테이블 정보 확인
            print("\n테이블 정보:")
            columns = db.describe_table(table_name)
            print(f"  총 컬럼 수: {len(columns)}")
            print(f"  Primary Key: datetime")
            
    except Exception as e:
        print(f"✗ Low Resolution 테이블 생성 실패: {e}")
        raise


def create_high_resolution_table(drop_if_exists=False, include_5min=False):
    """
    High Resolution 테이블 생성
    
    Args:
        drop_if_exists: True면 기존 테이블 삭제 후 재생성
        include_5min: True면 5분 데이터용 GOES flux 컬럼 포함
    """
    print("\n" + "=" * 60)
    print("High Resolution 테이블 생성")
    print("=" * 60)
    
    table_name = 'high_resolution'
    
    try:
        with PostgresManager(**DB_CONFIG) as db:
            # 기존 테이블 확인
            tables = db.list_tables(names_only=True)
            table_exists = table_name in tables
            
            if table_exists:
                if drop_if_exists:
                    print(f"⚠ 기존 테이블 삭제 중: {table_name}")
                    db.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                    print(f"✓ 기존 테이블 삭제 완료")
                else:
                    print(f"✓ 테이블 이미 존재: {table_name}")
                    print("  (drop_if_exists=True로 재생성 가능)")
                    return
            
            # 테이블 스키마 가져오기
            schema = get_highres_schema(include_5min_fields=include_5min)
            
            # 테이블 생성
            data_type = "5-min" if include_5min else "1-min"
            print(f"테이블 생성 중: {table_name} ({len(schema)} 컬럼, {data_type})")
            db.create_table(table_name, schema)
            print(f"✓ 테이블 생성 완료")
            
            # 인덱스 생성
            print("인덱스 생성 중...")
            index_queries = get_highres_indices()
            for i, query in enumerate(index_queries, 1):
                db.execute(query)
                print(f"  {i}. {query.split('INDEX')[1].split('ON')[0].strip()}")
            
            print(f"✓ {len(index_queries)}개 인덱스 생성 완료")
            
            # 테이블 정보 확인
            print("\n테이블 정보:")
            columns = db.describe_table(table_name)
            print(f"  총 컬럼 수: {len(columns)}")
            print(f"  Primary Key: datetime")
            print(f"  데이터 타입: {data_type}")
            
    except Exception as e:
        print(f"✗ High Resolution 테이블 생성 실패: {e}")
        raise


def show_table_info():
    """생성된 테이블 정보 출력"""
    print("\n" + "=" * 60)
    print("생성된 테이블 정보")
    print("=" * 60)
    
    try:
        with PostgresManager(**DB_CONFIG) as db:
            # 테이블 리스트
            tables = db.list_tables()
            
            if not tables:
                print("생성된 테이블이 없습니다.")
                return
            
            for table_info in tables:
                table_name = table_info['name']
                table_size = table_info['size']
                
                print(f"\n테이블: {table_name}")
                print(f"  크기: {table_size}")
                
                # 레코드 수
                count = db.count(table_name)
                print(f"  레코드 수: {count:,}")
                
                # 컬럼 수
                columns = db.describe_table(table_name)
                print(f"  컬럼 수: {len(columns)}")
                
    except Exception as e:
        print(f"✗ 테이블 정보 조회 실패: {e}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("OMNI 데이터베이스 및 테이블 생성")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['database']}")
    print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print("=" * 60)
    
    try:
        # 1. 데이터베이스 생성 (실패해도 계속 진행)
        try:
            create_omni_database()
        except Exception as e:
            print(f"⚠ 데이터베이스 생성 실패 (계속 진행): {e}")
        
        # 2. Low Resolution 테이블 생성
        create_low_resolution_table(drop_if_exists=False)
        
        # 3. High Resolution 테이블 생성
        # 1-min 데이터용 (46 컬럼)
        create_high_resolution_table(drop_if_exists=False, include_5min=False)
        
        # 5-min 데이터도 필요하면 아래 주석 해제
        # create_high_resolution_table(drop_if_exists=False, include_5min=True)
        
        # 4. 테이블 정보 확인
        show_table_info()
        
        print("\n" + "=" * 60)
        print("✓ 모든 테이블 생성 완료!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 에러 발생: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()