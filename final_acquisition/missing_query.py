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
DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"


def query_hmi_45s(date_target):
    client = drms.Client(email="eunsupark@kasi.re.kr")
    date = date_target - datetime.timedelta(seconds=45)
    date_start = date - datetime.timedelta(minutes=3)
    date_end = date + datetime.timedelta(minutes=3)
    date_start = date_start.strftime("%Y.%m.%d_%H:%M:%S")
    date_end = date_end.strftime("%Y.%m.%d_%H:%M:%S")

    query_str = f"hmi.m_45s[{date_start}-{date_end}]"
    print(f"date: {date_target} query: {query_str}")
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)
    csv_file_name =  f"missing_{date_target:%Y%m%d%H}_hmi.csv"
    df = pd.DataFrame({'url': url_list})
    df.to_csv(f"{CSV_DIR}/{csv_file_name}", index=True, encoding='utf-8')


def query_aia(date_target, wavelengths=[193, 211]):
    client = drms.Client(email="harim.lee@njit.edu")
    date_start = date_target - datetime.timedelta(minutes=3)
    date_end = date_target + datetime.timedelta(minutes=3)
    date_start = date_start.strftime("%Y.%m.%d_%H:%M:%S")
    date_end = date_end.strftime("%Y.%m.%d_%H:%M:%S")
    client = drms.Client(email="harim.lee@njit.edu")
    wl_str = ', '.join(map(str, wavelengths))
    query_str = f"aia.lev1_euv_12s[{date_start}-{date_end}][{wl_str}]"
    print(f"date: {date_target} query: {query_str}")
    export_request = client.export(query_str, method='url', protocol='fits')
    export_request.wait()

    url_list = []
    for url in export_request.urls.url:
        url_list.append(url)
    csv_file_name =  f"missing_{date_target:%Y%m%d%H}_aia.csv"
    df = pd.DataFrame({'url': url_list})
    df.to_csv(f"{CSV_DIR}/{csv_file_name}", index=True, encoding='utf-8')


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", "-i", type=str)
    args = parser.parse_args()


    missing_dict = {}
    df = pd.read_csv(f"{DATA_ROOT}/missing.csv")
    for n in range(len(df)) :
        line = df.iloc[n]
        instrument = line["instrument"]
        wavelength = line["wavelength"]
        date = line["date"]
        if date in missing_dict.keys() :
            missing_dict[date].append(f"{instrument}_{wavelength}")
        else :
            missing_dict[date] = [f"{instrument}_{wavelength}"]

    for k, vs in missing_dict.items():
        date = k
        date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

        aia = []
        hmi = False

        for v in vs :
            if 'aia' in v :
                aia.append(v.split('_')[1])
            elif 'hmi' in v :
                hmi = True

        print(aia, hmi)

        if args.instrument == "aia" :
            if len(aia) > 0 :
                csv_file_name =  f"missing_{date:%Y%m%d%H}_aia.csv"
                if os.path.exists(f"{CSV_DIR}/{csv_file_name}") is False :
                    try :
                        query_aia(date_target=date, wavelengths=aia)
                    except :
                        time.sleep(60)

        if args.instrument == "hmi" :
            if hmi is True :
                csv_file_name =  f"missing_{date:%Y%m%d%H}_hmi.csv"
                if os.path.exists(f"{CSV_DIR}/{csv_file_name}") is False :
                    try :
                        query_hmi_45s(date_target=date)
                    except :
                        time.sleep(60)








