import os
import uuid
import datetime

import drms
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import pandas as pd

urllib3.disable_warnings(InsecureRequestWarning)


def query_hmi_45s(date_target):
    client = drms.Client(email="eunsupark@kasi.re.kr")
    date_str = (date_target - datetime.timedelta(seconds=45)).strftime("%Y.%m.%d_%H:%M:%S")
    query_str = f"hmi.m_45s[{date_str}/7d@1h]"
    print(query_str)
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)
    csv_file_name =  f"{uuid.uuid4()}.csv"
    df = pd.DataFrame({'url': url_list})
    df.to_csv(csv_file_name, index=True, encoding='utf-8')


def query_aia(date_target, wavelengths=[193, 211]):
    client = drms.Client(email="eunsupark@kasi.re.kr")
    date_str = date_target.strftime("%Y.%m.%d_%H:%M:%S")
    wl_str = ', '.join(map(str, wavelengths))
    query_str = f"aia.lev1_euv_12s[{date_str}/7d@1h][{wl_str}]"
    print(query_str)
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)
    csv_file_name =  f"{uuid.uuid4()}.csv"
    df = pd.DataFrame({'url': url_list})
    df.to_csv(csv_file_name, index=True, encoding='utf-8')


if __name__ == "__main__" :

    date_target = datetime.datetime(
        year = 2011,
        month = 9,
        day = 28
    )
    # date_target = datetime.datetime(
    #     year = 2010,
    #     month = 9,
    #     day = 1
    # )
    date_end = datetime.datetime(
        year = 2025,
        month = 1,
        day = 1,
    )

    while date_target < date_end :
        query_hmi_45s(date_target)
        query_aia(date_target)
        date_target += datetime.timedelta(days=7)