import os
from glob import glob
import datetime
import pandas as pd
from shutil import copy2


LOAD_ROOT = "/Volumes/usbshare1/data/sdo_jp2"
SAVE_ROOT = "/Users/eunsupark/Data/ap-data/sdo_jp2"
MAX_TIMEDELTA = 3600
CSV_FILE_NAME = "ap_data_list.csv"
AIA_193_DIR = "ap-data/sdo_jp2/aia/193"
AIA_211_DIR = "ap-data/sdo_jp2/aia/211"
HMI_MAGNETOGRAM_DIR = "ap-data/sdo_jp2/hmi/magnetogram"

# 전역 캐시 딕셔너리
file_cache = {}


def get_dir_path(instrument, wavelength, date):
    """Get directory path for SDO data"""
    return f"{LOAD_ROOT}/{instrument}/{wavelength}/{date:%Y}/{date:%Y%m%d}"


def get_files_from_cache(dir_path, pattern):
    """Get files from cache or read and cache them"""
    cache_key = (dir_path, pattern)
    
    if cache_key not in file_cache:
        file_list = glob(f"{dir_path}/{pattern}")
        file_cache[cache_key] = file_list
    
    return file_cache[cache_key]


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


def get_hourly_data(date_target, instrument, wavelength):
    """Get the closest file to target date within MAX_TIMEDELTA"""
    
    # Search in ±1 day range
    date_target_before = date_target - datetime.timedelta(days=1)
    date_target_after = date_target + datetime.timedelta(days=1)

    dir_path_before = get_dir_path(instrument, wavelength, date_target_before)
    dir_path = get_dir_path(instrument, wavelength, date_target)
    # dir_path_after = get_dir_path(instrument, wavelength, date_target_after)

    # Collect all matching files using cache
    instrument_upper = instrument.upper()
    pattern = f"*__SDO_{instrument_upper}_{instrument_upper}_{wavelength}.jp2"
    
    file_list = []
    file_list += get_files_from_cache(dir_path_before, pattern)
    file_list += get_files_from_cache(dir_path, pattern)
    # file_list += get_files_from_cache(dir_path_after, pattern)
    
    if len(file_list) == 0:
        return None

    # Find closest file
    date_list = get_date_from_files(file_list, instrument, wavelength)
    timedelta_list = get_timedelta(date_list, date_target)

    min_timedelta = min(timedelta_list)
    min_index = timedelta_list.index(min_timedelta)

    if min_timedelta < MAX_TIMEDELTA:
        # return file_list[min_index]
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


if __name__ == "__main__":

    # date_target = datetime.datetime(year=2011, month=1, day=1, hour=0)
    # date_end = datetime.datetime(year=2011, month=2, day=1, hour=0)

    date_target = datetime.datetime(year=2010, month=9, day=1, hour=0)
    # date_end = datetime.datetime(year=2011, month=1, day=1, hour=0)
    date_end = datetime.datetime(year=2025, month=1, day=1, hour=0)

    N = 0
    while date_target < date_end:

        aia_193 = get_hourly_data(date_target, "aia", 193)
        aia_211 = get_hourly_data(date_target, "aia", 211)
        hmi_magnetogram = get_hourly_data(date_target, "hmi", "magnetogram")

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
            source = aia_193
            destination = f"{AIA_193_DIR}/{os.path.basename(aia_193)}"
            if not os.path.exists(destination):
                copy2(source, destination)
        if aia_211 is not None :
            source = aia_211
            destination = f"{AIA_211_DIR}/{os.path.basename(aia_211)}"
            if not os.path.exists(destination):
                copy2(source, destination)
        if hmi_magnetogram is not None :
            source = hmi_magnetogram
            destination = f"{HMI_MAGNETOGRAM_DIR}/{os.path.basename(hmi_magnetogram)}"
            if not os.path.exists(destination):
                copy2(source, destination)

        print(f"{date_target}: {aia_193 is not None} {aia_211 is not None} {hmi_magnetogram is not None}")

        date_target += datetime.timedelta(hours=3)
        N += 1
        # if N == 10 :
        #     break

    # 캐시 통계 출력 (선택사항)
    print(f"\n캐시된 디렉토리 수: {len(file_cache)}")
