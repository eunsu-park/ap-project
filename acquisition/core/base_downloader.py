"""
Base Downloader - 공통 다운로더 기능
모든 데이터 다운로더의 공통 기능을 제공하는 기반 클래스
파일 위치: core/base_downloader.py
"""

import sys
import datetime
import argparse
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time

# Python 버전 체크
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

class BaseDataSource(ABC):
    """데이터 소스의 추상 기반 클래스"""
    
    def __init__(self, mission_start_date=None):
        self.mission_start_date = mission_start_date
    
    @abstractmethod
    def get_url(self, **kwargs) -> str:
        """URL 생성"""
        pass
    
    @abstractmethod  
    def get_save_path(self, destination_root: str, **kwargs) -> str:
        """저장 경로 생성"""
        pass

class BaseDownloader:
    """공통 다운로더 기능"""
    
    def __init__(self, data_source, parallel=1, overwrite=False, max_retries=3):
        self.data_source = data_source
        self.parallel = parallel
        self.overwrite = overwrite
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
    
    def download_single_file(self, source_url: str, destination: str) -> bool:
        """단일 파일 다운로드 (간단한 재시도 포함)"""
        import requests
        
        if Path(destination).exists() and not self.overwrite:
            return True
        
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(source_url, timeout=30)
                response.raise_for_status()
                
                with open(destination, 'wb') as f:
                    f.write(response.content)
                return True
                
            except Exception as e:
                if attempt == self.max_retries:
                    self.logger.error(f"Failed to download {source_url}: {e}")
                    return False
                time.sleep(2 ** attempt)  # 지수 백오프
        
        return False
    
    def get_file_list(self, base_url: str, extensions: list) -> list:
        """웹 디렉토리에서 파일 리스트 가져오기"""
        import requests
        from bs4 import BeautifulSoup
        
        try:
            response = requests.get(f"{base_url}/", timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            files = []
            
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if (href and 
                    any(href.lower().endswith(f".{ext.lower()}") for ext in extensions) and
                    not href.startswith('/') and '?' not in href):
                    files.append(href)
            
            return [f for f in files if not any(skip in f.lower() 
                    for skip in ['parent', '..', 'index', 'readme'])]
        
        except Exception as e:
            self.logger.error(f"Error fetching file list from {base_url}: {e}")
            return []
    
    def download_parallel(self, download_tasks: list) -> dict:
        """병렬 다운로드 실행"""
        if not download_tasks:
            return {"downloaded": 0, "failed": 0}
        
        successful = 0
        failed = 0
        
        if self.parallel < 2:
            # 순차 처리
            for source, dest in download_tasks:
                if self.download_single_file(source, dest):
                    successful += 1
                else:
                    failed += 1
        else:
            # 병렬 처리
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                futures = [executor.submit(self.download_single_file, src, dst) 
                          for src, dst in download_tasks]
                
                for future in futures:
                    try:
                        if future.result():
                            successful += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1
        
        return {"downloaded": successful, "failed": failed}
    
    def run_with_retry(self, **params):
        """메인 다운로드 실행 (재시도 포함)"""
        try:
            # destination_root 분리하여 중복 전달 방지
            destination_root = params.pop('destination_root', './data')
            extensions = params.get('extensions', ['jp2', 'fts'])
            
            base_url = self.data_source.get_url(**params)
            save_dir = self.data_source.get_save_path(destination_root, **params)
            
            file_list = self.get_file_list(base_url, extensions)
            if not file_list:
                self.logger.warning(f"No files found at {base_url}")
                return
            
            # 다운로드 태스크 생성
            download_tasks = []
            for filename in file_list:
                source = f"{base_url}/{filename}"
                destination = str(Path(save_dir) / filename)
                download_tasks.append((source, destination))
            
            # 다운로드 실행
            result = self.download_parallel(download_tasks)
            self.logger.info(f"Downloaded: {result['downloaded']}, Failed: {result['failed']}")
            
        except Exception as e:
            self.logger.error(f"Download failed: {e}")

class BaseCLI:
    """공통 CLI 기능"""
    
    def __init__(self, program_name: str, description: str):
        self.program_name = program_name
        self.description = description
        self.parser = None
    
    def setup_common_args(self):
        """공통 인자들 설정"""
        self.parser = argparse.ArgumentParser(description=self.description)
        
        # 날짜 옵션
        self.parser.add_argument("--start-date", type=str, help="시작 날짜 (YYYY-MM-DD)")
        self.parser.add_argument("--end-date", type=str, help="종료 날짜 (YYYY-MM-DD)")
        self.parser.add_argument("--days", type=int, default=7, help="최근 며칠 다운로드")
        
        # 다운로드 옵션
        self.parser.add_argument("--extensions", type=str, nargs="+", default=["fts"],
                               help="파일 확장자")
        self.parser.add_argument("--destination", type=str, default="./data", 
                               help="저장 디렉토리")
        self.parser.add_argument("--parallel", type=int, default=1, help="병렬 다운로드 수")
        self.parser.add_argument("--overwrite", action="store_true", help="기존 파일 덮어쓰기")
        self.parser.add_argument("--max-retries", type=int, default=3, help="최대 재시도 횟수")
        
        # 로깅
        self.parser.add_argument("--log-level", type=str, default="INFO", help="로깅 레벨")
        self.parser.add_argument("--log-file", type=str, help="로그 파일 경로")
        
        return self.parser
    
    @staticmethod
    def setup_logging(level: str = "INFO", log_file: str = None):
        """로깅 설정"""
        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            handlers=handlers,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
    
    @staticmethod
    def parse_date_range(args) -> tuple:
        """날짜 범위 파싱"""
        if args.start_date or args.end_date:
            if args.start_date:
                start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
            else:
                start_date = datetime.date.today() - datetime.timedelta(days=args.days - 1)
            
            if args.end_date:
                end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
            else:
                end_date = datetime.date.today()
        else:
            today = datetime.date.today()
            start_date = today - datetime.timedelta(days=args.days - 1)
            end_date = today
        
        return start_date, end_date
    
    @staticmethod
    def validate_mission_dates(start_date: datetime.date, mission_start: datetime.datetime):
        """미션 시작 날짜 검증"""
        mission_start_date = mission_start.date()
        if start_date < mission_start_date:
            print(f"Warning: Start date {start_date} is before mission start {mission_start_date}")
            return mission_start_date
        return start_date

def download_date_range(downloader: BaseDownloader, start_date: datetime.date, 
                       end_date: datetime.date, **params) -> None:
    """날짜 범위 다운로드 (공통 함수)"""
    current_date = start_date
    while current_date <= end_date:
        print(f"Processing {current_date}")
        downloader.run_with_retry(
            year=current_date.year,
            month=current_date.month,
            day=current_date.day,
            **params
        )
        current_date += datetime.timedelta(days=1)