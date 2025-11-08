import os
import time
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
    if 'hmi' in source_url :
        destination = f"/Users/eunsupark/Data/sdo/fits/hmi/{file_name}"
    elif 'aia' in source_url :
        destination = f"/Users/eunsupark/Data/sdo/fits/aia/{file_name}"
    download_single_file(source_url=source_url, destination=destination)


def main():

    while True :

        csv_file_list = glob("*.csv")
        print(len(csv_file_list))
        if len(csv_file_list) == 0 :
            time.sleep(30)
        
        else :
            df_list = []
            for file_path in csv_file_list :
                df = pd.read_csv(file_path)
                df_list.append(df)
            df = pd.concat(df_list)
            url_list = df['url']

            with ProcessPoolExecutor(max_workers=4) as executor:
                future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}



            import sys
            sys.exit()
            



def hmi_45s(date_target):
    client = drms.Client(email="eunsupark@kasi.re.kr")
    date_str = (date_target - datetime.timedelta(seconds=45)).strftime("%Y.%m.%d_%H:%M:%S")
    query_str = f"hmi.m_45s[{date_str}/7d@1h]"
    print(query_str)
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    source_and_destination_list = []
    for source in export_request.urls.url:
        filename = os.path.basename(source)
        destination = f"/Users/eunsupark/Data/sdo/fits/hmi/{filename}"
        source_and_destination_list.append((source, destination))

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_date = {executor.submit(run, source_and_destination): source_and_destination for source_and_destination in source_and_destination_list}


def aia(date_target, wavelengths=[193, 211]):
    client = drms.Client(email="eunsupark@kasi.re.kr")
    date_str = date_target.strftime("%Y.%m.%d_%H:%M:%S")
    wl_str = ', '.join(map(str, wavelengths))
    query_str = f"aia.lev1_euv_12s[{date_str}/7d@1h][{wl_str}]"
    print(query_str)
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    source_and_destination_list = []
    for source in export_request.urls.url:
        filename = os.path.basename(source)
        destination = f"/Users/eunsupark/Data/sdo/fits/aia/{filename}"
        source_and_destination_list.append((source, destination))

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_date = {executor.submit(run, source_and_destination): source_and_destination for source_and_destination in source_and_destination_list}


if __name__ == "__main__" :

    main()
