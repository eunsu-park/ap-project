import os
from glob import glob
import datetime
import pandas as pd


LOAD_ROOT = "/Volumes/usbshare1/data/sdo_jp2"
SAVE_ROOT = "/Users/eunsupark/Data/ap-data/sdo_jp2"
MAX_TIMEDELTA = 3600

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
    dir_path_after = get_dir_path(instrument, wavelength, date_target_after)

    # Collect all matching files using cache
    instrument_upper = instrument.upper()
    pattern = f"*__SDO_{instrument_upper}_{instrument_upper}_{wavelength}.jp2"
    
    file_list = []
    file_list += get_files_from_cache(dir_path_before, pattern)
    file_list += get_files_from_cache(dir_path, pattern)
    file_list += get_files_from_cache(dir_path_after, pattern)

    for n in range(len(file_list)) :
        file_path = file_list[n]
        file_path = file_path.replace(f"{LOAD_ROOT}/", "")
        file_list[n] = file_path
    
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


if __name__ == "__main__":

    date_list = []
    aia_193_list = []
    aia_211_list = []
    hmi_magnetogram_list = []

    date_target = datetime.datetime(year=2011, month=1, day=1, hour=0)
    date_end = datetime.datetime(year=2011, month=2, day=1, hour=0)

    # date_target = datetime.datetime(year=2010, month=9, day=1, hour=0)
    # date_end = datetime.datetime(year=2011, month=1, day=1, hour=0)

    N = 0
    while date_target < date_end:

        aia_193 = get_hourly_data(date_target, "aia", 193)
        aia_211 = get_hourly_data(date_target, "aia", 211)
        hmi_magnetogram = get_hourly_data(date_target, "hmi", "magnetogram")

        date_list.append(date_target)
        aia_193_list.append(aia_193)
        aia_211_list.append(aia_211)
        hmi_magnetogram_list.append(hmi_magnetogram)

        date_target += datetime.timedelta(hours=1)
        N += 1
        if N == 10 :
            break

    year_list = []
    month_list = []
    day_list = []
    hour_list = []

    for date in date_list:
        year_list.append(date.year)
        month_list.append(date.month)
        day_list.append(date.day)
        hour_list.append(date.hour)

    data_dict = {}
    data_dict["datetime"] = date_list
    data_dict["year"] = year_list
    data_dict["month"] = month_list
    data_dict["day"] = day_list
    data_dict["hour"] = hour_list
    data_dict["aia_193"] = aia_193_list
    data_dict["aia_211"] = aia_211_list
    data_dict["hmi_magnetogram"] = hmi_magnetogram_list

    df = pd.DataFrame(data_dict)
    df.to_csv("ap_data_list.csv", index=False, encoding='utf-8')
    
    # 캐시 통계 출력 (선택사항)
    print(f"\n캐시된 디렉토리 수: {len(file_cache)}")

# import os
# from glob import glob
# import datetime
# import pandas as pd


# LOAD_ROOT = "/Volumes/usbshare1/data/sdo_jp2"
# SAVE_ROOT = "/Users/eunsupark/Data/ap-data/sdo_jp2"
# MAX_TIMEDELTA = 3600


# def get_dir_path(instrument, wavelength, date):
#     """Get directory path for SDO data"""
#     return f"{LOAD_ROOT}/{instrument}/{wavelength}/{date:%Y}/{date:%Y%m%d}"


# def get_date_from_files(file_list, instrument, wavelength):
#     """Extract datetime objects from file names"""
#     date_list = []
#     for file_path in file_list:
#         file_name = os.path.basename(file_path)
#         instrument_upper = instrument.upper()
#         date = datetime.datetime.strptime(
#             file_name, 
#             f"%Y_%m_%d__%H_%M_%S_%f__SDO_{instrument_upper}_{instrument_upper}_{wavelength}.jp2"
#         )
#         date_list.append(date)
#     return date_list


# def get_timedelta(date_list, date_target):
#     """Calculate time differences in seconds"""
#     timedelta_list = []
#     for date in date_list:
#         timedelta = abs(date_target - date)
#         timedelta_list.append(timedelta.total_seconds())
#     return timedelta_list


# def get_hourly_data(date_target, instrument, wavelength):
#     """Get the closest file to target date within MAX_TIMEDELTA"""
    
#     # Search in ±1 day range
#     date_target_before = date_target - datetime.timedelta(days=1)
#     date_target_after = date_target + datetime.timedelta(days=1)

#     dir_path_before = get_dir_path(instrument, wavelength, date_target_before)
#     dir_path = get_dir_path(instrument, wavelength, date_target)
#     dir_path_after = get_dir_path(instrument, wavelength, date_target_after)

#     # Collect all matching files
#     instrument_upper = instrument.upper()
#     pattern = f"*__SDO_{instrument_upper}_{instrument_upper}_{wavelength}.jp2"
    
#     file_list = []
#     file_list += glob(f"{dir_path_before}/{pattern}")
#     file_list += glob(f"{dir_path}/{pattern}")
#     file_list += glob(f"{dir_path_after}/{pattern}")
#     if len(file_list) == 0 :
#         return None

#     # Find closest file
#     date_list = get_date_from_files(file_list, instrument, wavelength)
#     timedelta_list = get_timedelta(date_list, date_target)

#     min_timedelta = min(timedelta_list)
#     min_index = timedelta_list.index(min_timedelta)

#     if min_timedelta < MAX_TIMEDELTA:
#         return file_list[min_index]
#     else:
#         return None


# if __name__ == "__main__":


#     date_list = []
#     aia_193_list = []
#     aia_211_list = []
#     hmi_magnetogram_list = []

#     date_target = datetime.datetime(year=2010, month=9, day=1, hour=0)
#     date_end = datetime.datetime(year=2011, month=1, day=1, hour=0)

#     N = 0 # for code test only
#     while date_target < date_end :

#         aia_193 = get_hourly_data(date_target, "aia", 193)
#         aia_211 = get_hourly_data(date_target, "aia", 211)
#         hmi_magnetogram = get_hourly_data(date_target, "hmi", "magnetogram")

#         date_list.append(date_target)
#         aia_193_list.append(aia_193)
#         aia_211_list.append(aia_211)
#         hmi_magnetogram_list.append(hmi_magnetogram)

#         date_target += datetime.timedelta(hours=1)
#         N += 1 # for code test only
#         if N == 10 : # for code test only
#             break # for code test only

#     print(date_list)
#     print(aia_193_list)
#     print(aia_211_list)
#     print(hmi_magnetogram_list)

#     year_list = []
#     month_list = []
#     day_list = []
#     hour_list = []

#     for date in date_list :
#         year_list.append(date.year)
#         month_list.append(date.month)
#         day_list.append(date.day)
#         hour_list.append(date.hour)

#     data_dict = {}
#     data_dict["datetime"] = date_list
#     data_dict["year"] = year_list
#     data_dict["month"] = month_list
#     data_dict["day"] = day_list
#     data_dict["hour"] = hour_list
#     data_dict["aia_193"] = aia_193_list
#     data_dict["aia_211"] = aia_211_list
#     data_dict["hmi_magnetogram"] = hmi_magnetogram_list

#     df = pd.DataFrame(data_dict)
#     df.to_csv("ap_data_list.csv", index=False, encoding='utf-8')