import os
import datetime
from sunpy.map import Map


def parsing_date(t_rec, instrument):
    """T_REC 문자열을 datetime 객체로 변환"""
    if instrument == "aia" :
        date = datetime.datetime.strptime(t_rec, "%Y-%m-%dT%H:%M:%S.%f")
    elif instrument == "hmi" :
        date = datetime.datetime.strptime(t_rec, "%Y.%m.%d_%H:%M:%S.%f_TAI")
    return date


def get_file_path(original_path):
    file_name = os.path.basename(original_path)
    M = Map(original_path)
    meta = M.meta
    instrument = meta["TELESCOP"].split('/')[1].lower()
    t_rec = meta["T_REC"]

    date = parsing_date(t_rec, instrument)
    save_path = f"/archive/sdo/fits/{instrument}/{date:%Y}/{date:%Y%m%d}/{file_name}"
    return save_path

