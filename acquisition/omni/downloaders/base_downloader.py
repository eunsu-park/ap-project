"""
공통 다운로드 로직

Base downloader class with common download and database insertion methods
"""
import requests
from egghouse.database import PostgresManager
from config import BATCH_SIZE, DOWNLOAD_TIMEOUT


class BaseDownloader:
    """
    OMNI 다운로더 베이스 클래스
    
    공통 다운로드 및 DB 삽입 로직을 제공
    """
    
    def __init__(self, db_config, table_config, parser):
        """
        초기화
        
        Args:
            db_config: 데이터베이스 설정 딕셔너리
            table_config: 테이블 설정 딕셔너리
            parser: 파서 인스턴스 (LowResParser 또는 HighResParser)
        """
        self.db_config = db_config
        self.table_config = table_config
        self.parser = parser
    
    def download_from_url(self, url, timeout=None):
        """
        URL에서 파일 다운로드 (공통)
        
        Args:
            url: 다운로드 URL
            timeout: 타임아웃 (초), None이면 기본값 사용
        
        Returns:
            str: 다운로드한 텍스트 데이터
            None: 다운로드 실패 시
        """
        if timeout is None:
            timeout = DOWNLOAD_TIMEOUT
        
        try:
            print(f"  Downloading: {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            print(f"  ✓ Downloaded: {len(response.text):,} bytes")
            return response.text
            
        except requests.exceptions.Timeout:
            print(f"  ✗ Timeout error (>{timeout}s)")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Download error: {e}")
            return None
    
    def insert_to_db(self, df, year):
        """
        DataFrame을 DB에 삽입 (공통)
        
        Args:
            df: pandas DataFrame
            year: 연도 (중복 제거용)
        
        Returns:
            int: 삽입된 레코드 수
        """
        # DataFrame → 레코드 리스트 변환
        print("  Converting to records...")
        records = self.parser.convert_to_records(df)
        
        if not records:
            print("  ✗ No records to insert")
            return 0
        
        table_name = self.table_config['table_name']
        
        with PostgresManager(**self.db_config) as db:
            # 기존 연도 데이터 확인
            print(f"  Checking existing data for year {year}...")
            existing_count = db.count(table_name, where={'year': year})
            
            if existing_count > 0:
                print(f"  ⚠ Found {existing_count} existing records for {year}")
                print(f"  Deleting existing records...")
                db.delete(table_name, where={'year': year})
                print(f"  ✓ Deleted {existing_count} records")
            
            # 배치 삽입
            print(f"  Inserting {len(records)} records in batches...")
            batch_size = BATCH_SIZE
            inserted = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                try:
                    db.insert(table_name, batch)
                    inserted += len(batch)
                    
                    # 진행 상황 출력
                    if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= len(records):
                        print(f"    Progress: {min(i + batch_size, len(records)):,}/{len(records):,}")
                        
                except Exception as e:
                    print(f"  ✗ Batch insert error at position {i}: {e}")
                    # 에러 발생해도 계속 진행
                    continue
            
            print(f"  ✓ Inserted {inserted} records")
            return inserted
    
    def download_year(self, year, save_raw_file=False):
        """
        특정 연도 데이터 다운로드 및 DB 삽입 (공통 워크플로우)
        
        Args:
            year: 다운로드할 연도
            save_raw_file: True면 원본 파일(.dat/.asc) 저장
        
        Returns:
            bool: 성공 여부
        """
        print(f"\n{'='*60}")
        print(f"Processing year {year}")
        print(f"{'='*60}")
        
        # 1. URL 생성
        url = self.table_config['url_pattern'].format(year=year)
        
        # 2. 다운로드
        data_text = self.download_from_url(url)
        if data_text is None:
            print(f"✗ Failed to download {year}")
            return False
        
        # 3. 원본 파일 저장 (선택)
        if save_raw_file:
            self._save_raw_file(year, data_text)
        
        # 4. 파싱
        print("  Parsing data...")
        try:
            df = self.parser.parse_data(data_text)
            print(f"  ✓ Parsed: {len(df)} records, {len(df.columns)} columns")
        except Exception as e:
            print(f"  ✗ Parsing error: {e}")
            return False
        
        # 5. DB 삽입
        print("  Inserting to database...")
        try:
            inserted = self.insert_to_db(df, year)
            
            if inserted > 0:
                print(f"✓ Year {year} completed: {inserted} records")
                return True
            else:
                print(f"✗ Year {year} failed: no records inserted")
                return False
                
        except Exception as e:
            print(f"  ✗ Database insertion error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_raw_file(self, year, data_text):
        """
        원본 파일 저장 (내부 메서드)
        
        Args:
            year: 연도
            data_text: 파일 내용
        """
        import os
        
        # 저장 디렉토리
        save_dir = f"/tmp/omni_raw/{self.table_config['table_name']}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일명
        ext = self.table_config['file_extension']
        filename = f"{save_dir}/omni_{year}{ext}"
        
        try:
            with open(filename, 'w') as f:
                f.write(data_text)
            print(f"  ✓ Saved raw file: {filename}")
        except Exception as e:
            print(f"  ⚠ Could not save raw file: {e}")
    
    def download_multiple_years(self, start_year, end_year, save_raw_file=False):
        """
        여러 연도 데이터 일괄 다운로드
        
        Args:
            start_year: 시작 연도 (포함)
            end_year: 종료 연도 (포함)
            save_raw_file: True면 원본 파일 저장
        
        Returns:
            dict: 결과 통계 {'success': [], 'failed': []}
        """
        print(f"\n{'='*60}")
        print(f"Downloading {start_year} - {end_year}")
        print(f"Table: {self.table_config['table_name']}")
        print(f"{'='*60}")
        
        results = {
            'success': [],
            'failed': []
        }
        
        for year in range(start_year, end_year + 1):
            try:
                success = self.download_year(year, save_raw_file=save_raw_file)
                
                if success:
                    results['success'].append(year)
                else:
                    results['failed'].append(year)
                    
            except Exception as e:
                print(f"\n✗ Unexpected error for year {year}: {e}")
                results['failed'].append(year)
                continue
        
        # 결과 요약
        print(f"\n{'='*60}")
        print("Download Summary")
        print(f"{'='*60}")
        print(f"Success: {len(results['success'])} years")
        if results['success']:
            print(f"  {results['success']}")
        
        print(f"Failed: {len(results['failed'])} years")
        if results['failed']:
            print(f"  {results['failed']}")
        
        return results
    
    def verify_data(self, year):
        """
        삽입된 데이터 검증
        
        Args:
            year: 검증할 연도
        
        Returns:
            dict: 검증 결과
        """
        table_name = self.table_config['table_name']
        
        with PostgresManager(**self.db_config) as db:
            # 레코드 수
            count = db.count(table_name, where={'year': year})
            
            # 샘플 데이터
            sample = db.select(
                table_name,
                where={'year': year},
                limit=5,
                order_by='datetime'
            )
            
            # datetime 범위
            date_range = db.execute(f"""
                SELECT 
                    MIN(datetime) as min_date,
                    MAX(datetime) as max_date
                FROM {table_name}
                WHERE year = {year}
            """, fetch=True)
            
            return {
                'year': year,
                'count': count,
                'sample': sample,
                'date_range': date_range[0] if date_range else None
            }