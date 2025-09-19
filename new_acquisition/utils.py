import time
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import requests
from bs4 import BeautifulSoup


def download_single_file(source_url: str, destination: str, overwrite: bool = False, max_retries: int = 3) -> bool:
    """단일 파일 다운로드 (간단한 재시도 포함)"""

    if Path(destination).exists() and not overwrite:
        return True

    Path(destination).parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(source_url, timeout=30)
            response.raise_for_status()
            
            with open(destination, 'wb') as f:
                f.write(response.content)
            return True
            
        except Exception as e:
            if attempt == max_retries:
                print(f"Failed to download {source_url}: {e}")
                return False
            time.sleep(2 ** attempt)  # 지수 백오프

    return False


def get_file_list(base_url: str, extensions: list) -> list:
    """웹 디렉토리에서 파일 리스트 가져오기"""
    
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
        print(f"Error fetching file list from {base_url}: {e}")
        return []
    

def download_parallel(download_tasks: list, overwrite: bool = False, max_retries: int = 3, parallel: int = 1) -> dict:
    """병렬 다운로드 실행"""
    if not download_tasks:
        return {"downloaded": 0, "failed": 0}
    
    successful = 0
    failed = 0

    if parallel < 2:
        # 순차 처리
        for source, dest in download_tasks:
            if download_single_file(source, dest, overwrite, max_retries):
                successful += 1
            else:
                failed += 1
    else:
        # 병렬 처리
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(download_single_file, src, dst, overwrite, max_retries)
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


import datetime
from typing import Tuple


def parse_date_range(args) -> Tuple[datetime.date, datetime.date]:
    """
    날짜 범위를 파싱하여 start_date와 end_date를 반환합니다.
    
    Args:
        args: start_date, end_date, days, mission_start_date 속성을 가진 객체
        
    Returns:
        Tuple[datetime.date, datetime.date]: (start_date, end_date)
        
    Raises:
        ValueError: 잘못된 날짜 범위나 옵션 조합인 경우
    """
    # mission_start_date를 date 객체로 변환
    mission_start = datetime.datetime.strptime(args.mission_start_date, '%Y-%m-%d').date()
    # today = datetime.date.today(datetime.timezone.utc)
    # today = datetime.datetime(datetime.timezone)
    today = datetime.datetime.utcnow().date()
    
    # 입력 유효성 검사
    if args.days and (args.start_date or args.end_date):
        raise ValueError("days 옵션은 start_date 또는 end_date와 함께 사용할 수 없습니다.")
    
    # 날짜 범위 결정
    if args.days:
        # days 옵션 사용 시
        end_date = today
        start_date = end_date - datetime.timedelta(days=args.days - 1)
    else:
        # start_date 처리
        if args.start_date:
            start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
        else:
            start_date = mission_start
            
        # end_date 처리
        if args.end_date:
            end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
        else:
            end_date = today
    
    # 날짜 유효성 검사
    if start_date > end_date:
        raise ValueError("start_date는 end_date보다 이전이어야 합니다.")
        
    if start_date < mission_start:
        raise ValueError(f"start_date는 {args.mission_start_date} 이후여야 합니다.")
        
    if end_date < mission_start:
        raise ValueError(f"end_date는 {args.mission_start_date} 이후여야 합니다.")
    
    return start_date, end_date