import os
import time
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning
import urllib3

urllib3.disable_warnings(InsecureRequestWarning)

EXTENSIONS = ["jp2"]
MAX_TIMEDELTA = 3600
CSV_FILE_NAME = "ap_data_list.csv"

AIA_193_DIR = "ap-data/sdo_jp2/aia/193"
AIA_211_DIR = "ap-data/sdo_jp2/aia/211"
HMI_MAGNETOGRAM_DIR = "ap-data/sdo_jp2/hmi/magnetogram"

BASE_URL = "https://gs671-suske.ndc.nasa.gov/jp2"


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


def get_file_list(base_url: str, extensions: list) -> list:
    """웹 디렉토리에서 파일 리스트 가져오기"""
    
    try:
        response = requests.get(f"{base_url}/", timeout=30, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        files = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if (href and 
                any(href.lower().endswith(f".{ext.lower()}") for ext in extensions) and
                not href.startswith('/') and '?' not in href):
                files.append(f"{base_url}/{href}")
        
        return [f for f in files if not any(skip in f.lower() 
                for skip in ['parent', '..', 'index', 'readme'])]
    
    except Exception as e:
        print(f"Error fetching file list from {base_url}: {e}")
        return []


def get_date_from_files(file_list, instrument, wavelength):
    """Extract datetime objects from file names"""
    date_list = []
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        instrument_upper = instrument.upper()
        date = datetime.datetime.strptime(
            file_name, 
            f"%Y_%m_%d__%H_%M_%S_%f__SDO_{instrument_upper}_{instrument_upper}_{wavelength}.jp2"
        )
        date_list.append(date)
    return date_list


def get_timedelta(date_list, date_target):
    """Calculate time differences in seconds"""
    timedelta_list = []
    for date in date_list:
        timedelta = abs(date_target - date)
        timedelta_list.append(timedelta.total_seconds())
    return timedelta_list


def get_timedelta(date_list, date_target):
    """Calculate time differences in seconds"""
    timedelta_list = []
    for date in date_list:
        timedelta = abs(date_target - date)
        timedelta_list.append(timedelta.total_seconds())
    return timedelta_list


def get_hourly_data(date_target, instrument, wavelength):

    date_before = date_target - datetime.timedelta(days=1)

    instrument_upper = instrument.upper()
    base_url = f"{BASE_URL}/{instrument_upper}/{date_target:%Y/%m/%d}/{wavelength}"
    base_url_before = f"{BASE_URL}/{instrument_upper}/{date_before:%Y/%m/%d}/{wavelength}"

    file_list = []
    file_list += get_file_list(base_url=base_url, extensions=EXTENSIONS)
    file_list += get_file_list(base_url=base_url_before, extensions=EXTENSIONS)

    if len(file_list) == 0 :
        return None
    
    date_list = get_date_from_files(file_list=file_list, instrument=instrument, wavelength=wavelength)
    timedelta_list = get_timedelta(date_list, date_target)

    min_timedelta = min(timedelta_list)
    min_index = timedelta_list.index(min_timedelta)

    if min_timedelta < MAX_TIMEDELTA:
        return file_list[min_index]
    else:
        return None


def list_to_name(file_list):
    file_names = []
    for file_path in file_list:
        if file_path == None :
            file_names.append(None)
        else :
            file_name = os.path.basename(file_path)
            file_names.append(file_name)
    return file_names


def main(date_target):

    aia_193 = get_hourly_data(date_target, 'aia', 193)
    aia_211 = get_hourly_data(date_target, 'aia', 211)
    hmi_magnetogram = get_hourly_data(date_target, 'hmi', "magnetogram")

    print(aia_193)
    print(aia_211)
    print(hmi_magnetogram)

    df = pd.DataFrame({
        'datetime': [date_target],#.strftime('%Y-%m-%d %H:%M:%S')],
        "year" : [date_target.year],
        "month" : [date_target.month],
        "day" : [date_target.day],
        "hour" : [date_target.hour],
        "aia_193" : [os.path.basename(aia_193) if aia_193 is not None else None],
        "aia_211" : [os.path.basename(aia_211) if aia_211 is not None else None],
        "hmi_magnetogram" : [os.path.basename(hmi_magnetogram) if hmi_magnetogram is not None else None]
    })

    if os.path.exists(CSV_FILE_NAME) :
        df.to_csv(CSV_FILE_NAME, mode='a', header=False, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
    else :
        df.to_csv(CSV_FILE_NAME, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')

    if aia_193 is not None :
        source_url = aia_193
        destination = f"{AIA_193_DIR}/{os.path.basename(aia_193)}"
        download_single_file(source_url=source_url, destination=destination)
    if aia_211 is not None :
        source_url = aia_211
        destination = f"{AIA_211_DIR}/{os.path.basename(aia_211)}"
        download_single_file(source_url=source_url, destination=destination)
    if hmi_magnetogram is not None :
        source_url = hmi_magnetogram
        destination = f"{HMI_MAGNETOGRAM_DIR}/{os.path.basename(hmi_magnetogram)}"
        download_single_file(source_url=source_url, destination=destination)



if __name__ == "__main__" :

    date_target = datetime.datetime(year=2010, month=9, day=1, hour=0)
    # date_end = datetime.datetime(year=2011, month=1, day=1, hour=0)
    date_end = datetime.datetime(year=2025, month=1, day=1, hour=0)

    while date_target < date_end :
        main(date_target)

        date_target += datetime.timedelta(hours=3)


        


    # date_before = date - datetime.timedelta(days=1)

    # for wave in AIA_WAVES :
    #     base_url = f"{AIA_BASE_URL}/{date:%Y/%m/%d}/{wave}"
    #     base_url_before = f"{AIA_BASE_URL}/{date_before:%Y/%m/%d}/{wave}"
    #     file_list = []
    #     file_list += get_file_list(base_url=base_url, extensions=EXTENSIONS)
    #     file_list += get_file_list(base_url=base_url_before, extensions=EXTENSIONS)
    #     print(date, len(file_list))

    # for wave in HMI_WAVES :
    #     base_url = f"{HMI_BASE_URL}/{date:%Y/%m/%d}/{wave}"
    #     base_url_before = f"{HMI_BASE_URL}/{date_before:%Y/%m/%d}/{wave}"
    #     file_list = []
    #     file_list += get_file_list(base_url=base_url, extensions=EXTENSIONS)
    #     file_list += get_file_list(base_url=base_url_before, extensions=EXTENSIONS)
    #     print(date, len(file_list))

