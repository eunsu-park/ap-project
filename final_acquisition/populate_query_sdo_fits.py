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
    date = date_target - datetime.timedelta(seconds=45)
    date = date - datetime.timedelta(minutes=1)
    date_str = date.strftime("%Y.%m.%d_%H:%M:%S")
    query_str = f"hmi.m_45s[{date_str}/70d@12h]"
    print(f"date: {date_target} query: {query_str}")
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)
    csv_file_name =  f"populate_{date_target:%Y%m%d}_hmi.csv"
    df = pd.DataFrame({'url': url_list})
    df.to_csv(f"{CSV_DIR}/{csv_file_name}", index=True, encoding='utf-8')


def query_aia(date_target, wavelengths=[193, 211]):
    client = drms.Client(email="harim.lee@njit.edu")
    date = date_target + datetime.timedelta(seconds=24)
    date_str = date.strftime("%Y.%m.%d_%H:%M:%S")
    client = drms.Client(email="harim.lee@njit.edu")
    wl_str = ', '.join(map(str, wavelengths))
    query_str = f"aia.lev1_euv_12s[{date_str}/70d@1d][{wl_str}]"
    print(f"date: {date_target} query: {query_str}")
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)
    csv_file_name =  f"populate_{date_target:%Y%m%d}_aia.csv"
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

    date_end = datetime.datetime(
        year = int(end_year),
        month = int(end_month),
        day = int(end_day),
    )

    if args.instrument.lower() == 'hmi' :
        func = query_hmi_45s
        date_target = datetime.datetime(
            year = int(start_year),
            month = int(start_month),
            day = int(start_day),
            hour = 6
        )
    elif args.instrument.lower() == 'aia' :
        func = query_aia
        date_target = datetime.datetime(
            year = int(start_year),
            month = int(start_month),
            day = int(start_day),
            hour = 21
        )

    else :
        raise ValueError

    while date_target < date_end :
        try :
            func(date_target)
            date_target += datetime.timedelta(days=70)
        except :           
            time.sleep(60)

# # def query_aia(date_target, wavelengths=[193, 211]):

# #     start_date = date_target - datetime.timedelta(minutes=1)
# #     start_date = start_date.strftime("%Y.%m.%d_%H:%M:%S")
# #     end_date = date_target + datetime.timedelta(minutes=1)
# #     end_date = end_date.strftime("%Y.%m.%d_%H:%M:%S")

# #     client = drms.Client(email="harim.lee@njit.edu")
# #     wl_str = ', '.join(map(str, wavelengths))

# #     query_str = f"aia.lev1_euv_12s[{start_date}-{end_date}][{wl_str}]"
    
# #     print(f"date: {date_target} query: {query_str}")
# #     export_request = client.export(query_str, method='url', protocol='fits')
# #     export_request.wait()

# #     url_list = []
# #     for url in export_request.urls.url:
# #         url_list.append(url)

#     # with ProcessPoolExecutor(max_workers=8) as executor:
#     #     future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}


# # def query_hmi_45s(date_target):

# #     start_date = date_target - datetime.timedelta(minutes=1)
# #     start_date = start_date.strftime("%Y.%m.%d_%H:%M:%S")
# #     end_date = date_target + datetime.timedelta(minutes=1)
# #     end_date = end_date.strftime("%Y.%m.%d_%H:%M:%S")

# #     client = drms.Client(email="harim.lee@njit.edu")

# #     query_str = f"hmi.m_45s[{start_date}-{end_date}]"
    
# #     print(f"date: {date_target} query: {query_str}")
# #     export_request = client.export(query_str, method='url', protocol='fits')
# #     export_request.wait()

# #     url_list = []
# #     for url in export_request.urls.url:
# #         url_list.append(url)

# #     with ProcessPoolExecutor(max_workers=8) as executor:
# #         future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}


# def query_aia(date_target, wavelengths=[193, 211]):

#     date = date_target + datetime.timedelta(seconds=24)
#     date_str = date.strftime("%Y.%m.%d_%H:%M:%S")

#     client = drms.Client(email="harim.lee@njit.edu")
#     wl_str = ', '.join(map(str, wavelengths))

#     query_str = f"aia.lev1_euv_12s[{date_str}/28d@1d][{wl_str}]"
    
#     print(f"date: {date_target} query: {query_str}")
#     export_request = client.export(query_str, method='url', protocol='fits')
#     export_request.wait()

#     url_list = []
#     for url in export_request.urls.url:
#         url_list.append(url)

#     with ProcessPoolExecutor(max_workers=8) as executor:
#         future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}


# def query_hmi_45s(date_target):
#     client = drms.Client(email="eunsupark@kasi.re.kr")
#     date = date_target - datetime.timedelta(seconds=45)
#     date = date - datetime.timedelta(minutes=1)
#     date_str = date.strftime("%Y.%m.%d_%H:%M:%S")
#     query_str = f"hmi.m_45s[{date_str}/28d@12h]"
#     print(f"date: {date_target} query: {query_str}")
#     export_request = client.export(query_str, method='url', protocol='fits')
#     export_request.wait()

#     url_list = []
#     for url in export_request.urls.url:
#         url_list.append(url)

#     with ProcessPoolExecutor(max_workers=8) as executor:
#         future_to_date = {executor.submit(run, source_url): source_url for source_url in url_list}


# def check_quality():
#     file_list = sorted(glob(f"{TMP_DIR}/*.fits"))
#     for file_path in file_list :
#         # if ("193" in file_path) and (not "spike" in file_path) :
#         if not "spike" in file_path :
#             sdo_map = Map(file_path)
#             meta = sdo_map.meta
#             quality = meta["QUALITY"]
#             print(f"{file_path}: {quality}")


# if __name__ == "__main__" :

#     # date_target = datetime.datetime(
#     #     year = 2011,
#     #     month = 1,
#     #     day = 1,
#     #     hour = 21    
#     # )
#     # query_aia(date_target)

#     date_target = datetime.datetime(
#         year = 2011,
#         month = 1,
#         day = 1,
#         hour = 6    
#     )    
#     query_hmi_45s(date_target)

#     # date_target = datetime.datetime(
#     #     year = 2011,
#     #     month = 1,
#     #     day = 1,
#     #     hour = 18
#     # )    
#     # query_hmi_45s(date_target)

#     check_quality()

#     date_target += datetime.timedelta(days=28)

#     ## AIA : 21
#     ## HMI : 6, 18


    
