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
from sunpy.map import Map

urllib3.disable_warnings(InsecureRequestWarning)


TMP_DIR = "/Users/eunsupark/Data/sdo/fits/tmp"


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
    destination = f"{TMP_DIR}/{file_name}"
    download_single_file(source_url=source_url, destination=destination)


# def query_aia(date_target, wavelengths=[193, 211]):

#     start_date = date_target - datetime.timedelta(minutes=1)
#     start_date = start_date.strftime("%Y.%m.%d_%H:%M:%S")
#     end_date = date_target + datetime.timedelta(minutes=1)
#     end_date = end_date.strftime("%Y.%m.%d_%H:%M:%S")

#     client = drms.Client(email="harim.lee@njit.edu")
#     wl_str = ', '.join(map(str, wavelengths))

#     query_str = f"aia.lev1_euv_12s[{start_date}-{end_date}][{wl_str}]"
    
#     print(f"date: {date_target} query: {query_str}")
#     export_request = client.export(query_str, method='url', protocol='fits')
#     export_request.wait()

#     url_list = []
#     for url in export_request.urls.url:
#         url_list.append(url)

    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}


# def query_hmi_45s(date_target):

#     start_date = date_target - datetime.timedelta(minutes=1)
#     start_date = start_date.strftime("%Y.%m.%d_%H:%M:%S")
#     end_date = date_target + datetime.timedelta(minutes=1)
#     end_date = end_date.strftime("%Y.%m.%d_%H:%M:%S")

#     client = drms.Client(email="harim.lee@njit.edu")

#     query_str = f"hmi.m_45s[{start_date}-{end_date}]"
    
#     print(f"date: {date_target} query: {query_str}")
#     export_request = client.export(query_str, method='url', protocol='fits')
#     export_request.wait()

#     url_list = []
#     for url in export_request.urls.url:
#         url_list.append(url)

#     with ProcessPoolExecutor(max_workers=8) as executor:
#         future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}


def query_aia(date_target, wavelengths=[193, 211]):

    date = date_target + datetime.timedelta(seconds=24)
    date_str = date.strftime("%Y.%m.%d_%H:%M:%S")

    client = drms.Client(email="harim.lee@njit.edu")
    wl_str = ', '.join(map(str, wavelengths))

    query_str = f"aia.lev1_euv_12s[{date_str}/28d@1d][{wl_str}]"
    
    print(f"date: {date_target} query: {query_str}")
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}


def query_hmi_45s(date_target):
    client = drms.Client(email="eunsupark@kasi.re.kr")
    date = date_target - datetime.timedelta(seconds=45)
    date = date - datetime.timedelta(minutes=1)
    date_str = date.strftime("%Y.%m.%d_%H:%M:%S")
    query_str = f"hmi.m_45s[{date_str}/28d@12h]"
    print(f"date: {date_target} query: {query_str}")
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}


def check_quality():
    file_list = sorted(glob(f"{TMP_DIR}/*.fits"))
    for file_path in file_list :
        # if ("193" in file_path) and (not "spike" in file_path) :
        if not "spike" in file_path :
            sdo_map = Map(file_path)
            meta = sdo_map.meta
            quality = meta["QUALITY"]
            print(f"{file_path}: {quality}")


if __name__ == "__main__" :

    # date_target = datetime.datetime(
    #     year = 2011,
    #     month = 1,
    #     day = 1,
    #     hour = 21    
    # )
    # query_aia(date_target)

    date_target = datetime.datetime(
        year = 2011,
        month = 1,
        day = 1,
        hour = 6    
    )    
    query_hmi_45s(date_target)

    # date_target = datetime.datetime(
    #     year = 2011,
    #     month = 1,
    #     day = 1,
    #     hour = 18
    # )    
    # query_hmi_45s(date_target)

    check_quality()

    date_target += datetime.timedelta(days=28)

    ## AIA : 21
    ## HMI : 6, 18


    
