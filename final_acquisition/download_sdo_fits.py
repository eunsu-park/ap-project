import os
import time
import random
import datetime
from glob import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import drms
import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import pandas as pd

urllib3.disable_warnings(InsecureRequestWarning)


CSV_DIR = "/Users/eunsupark/JSOC"
DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"


def download_single_file(source_url: str, destination: str, overwrite: bool = False, max_retries: int = 3) -> bool:
    """단일 파일 다운로드 (간단한 재시도 포함)"""

    if Path(destination).exists() and not overwrite:
        return True

    Path(destination).parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(source_url, timeout=30, verify=False)
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


def run(source_url):
    file_name = os.path.basename(source_url)

    if os.path.exists(f"{DATA_ROOT}/aia/{file_name}") :
        return

    if os.path.exists(f"{DATA_ROOT}/hmi/{file_name}") :
        return
        
    if os.path.exists(f"{DATA_ROOT}/downloaded/{file_name}") :
        return

    if os.path.exists(f"{DATA_ROOT}/invalid_file/{file_name}") :
        return

    if os.path.exists(f"{DATA_ROOT}/invalid_header/{file_name}") :
        return

    if os.path.exists(f"{DATA_ROOT}/non_zero_quality/{file_name}") :
        return

    if "spike" in file_name : # do not download spike file
        return

    destination = f"{DATA_ROOT}/downloaded/{file_name}"
    download_single_file(source_url=source_url, destination=destination)


def main():

    csv_file_list = glob(f"{CSV_DIR}/*.csv")
    print(len(csv_file_list))
    if len(csv_file_list) == 0 :
        print("There is no CSV file. waiting...")
    
    else :
        df_list = []
        for file_path in csv_file_list :
            df = pd.read_csv(file_path)
            df_list.append(df)
        df = pd.concat(df_list)
        url_list = df['url'].tolist()
        random.shuffle(url_list)
        print(f"{len(url_list)} files in url_list")

        with ProcessPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}
    
        print("All Files are downloaded")
            

if __name__ == "__main__" :

    main()
