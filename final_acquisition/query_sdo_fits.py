import os
import time
import datetime
import argparse

import drms
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import pandas as pd

urllib3.disable_warnings(InsecureRequestWarning)


CSV_DIR = "/Users/eunsupark/JSOC"


def query_hmi_45s(date_target):
    client = drms.Client(email="eunsupark@kasi.re.kr")
    date_str = (date_target - datetime.timedelta(seconds=45)).strftime("%Y.%m.%d_%H:%M:%S")
    query_str = f"hmi.m_45s[{date_str}/7d@1h]"
    print(f"date: {date_target} query: {query_str}")
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)
    csv_file_name =  f"{date_target:%Y%m%d}_hmi.csv"
    df = pd.DataFrame({'url': url_list})
    df.to_csv(f"{CSV_DIR}/{csv_file_name}", index=True, encoding='utf-8')


def query_aia(date_target, wavelengths=[193, 211]):
    client = drms.Client(email="harim.lee@njit.edu")
    date_str = date_target.strftime("%Y.%m.%d_%H:%M:%S")
    wl_str = ', '.join(map(str, wavelengths))
    query_str = f"aia.lev1_euv_12s[{date_str}/7d@1h][{wl_str}]"
    print(f"date: {date_target} query: {query_str}")
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)
    csv_file_name =  f"{date_target:%Y%m%d}_aia.csv"
    df = pd.DataFrame({'url': url_list})
    df.to_csv(f"{CSV_DIR}/{csv_file_name}", index=True, encoding='utf-8')


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", "-i", type=str)
    parser.add_argument("--start-date", "-s", type=str, default="2010-09-01")
    parser.add_argument("--end_date", "-e", type=str, default="2025-01-01")
    args = parser.parse_args()

    start_date = args.start_date.split('-')

    if len(start_date) != 3 :
        raise ValueError
    
    start_year, start_month, start_day = start_date

    end_date = args.end_date.split('-')

    if len(end_date) != 3 :
        raise ValueError
    
    end_year, end_month, end_day = end_date

    date_target = datetime.datetime(
        year = int(start_year),
        month = int(start_month),
        day = int(start_day)
    )
    date_end = datetime.datetime(
        year = int(end_year),
        month = int(end_month),
        day = int(end_day),
    )

    if args.instrument.lower() == 'hmi' :
        func = query_hmi_45s
    elif args.instrument.lower() == 'aia' :
        func = query_aia
    else :
        raise ValueError

    while date_target < date_end :
        try :
            func(date_target)
            date_target += datetime.timedelta(days=7)
        except :           
            time.sleep(60)