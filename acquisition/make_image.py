import os
import pickle
import datetime
from glob import glob
from multiprocessing import Pool, freeze_support
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sunpy.map import Map
from PIL import Image
from aiapy.calibrate import register, update_pointing, correct_degradation

import pandas as pd
import matplotlib.pyplot as plt
from sunpy.map import Map
import numpy as np
from PIL import Image
from aiapy.calibrate import register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy.time import Time
from aiapy.calibrate.util import get_pointing_table

from egghouse.image import resize_image, circle_mask, pad_image
from concurrent.futures import ProcessPoolExecutor
# from egghouse.database


DATA_ROOT = "/opt/archive/sdo"
POINTING_TABLE_PATH = f"{DATA_ROOT}/fits/pointing_table.pkl"
CORRECTION_TABLE_PATH = f"{DATA_ROOT}/fits/correction_table.pkl"


def load_pointing_table(pointing_table_path):
    if os.path.exists(pointing_table_path) is False :

        # date_start = datetime.datetime(
        #     year = 2010,
        #     month = 9,
        #     day = 1,
        # )
        # date_start -= datetime.timedelta(days=1)
        # date_start = Time(date_start, format='datetime')
        # date_end = datetime.datetime(
        #     year = 2025,
        #     month = 1,
        #     day = 1,
        # )
        # date_end += datetime.timedelta(days=1)
        # date_end = Time(date_end, format='datetime')

        # pointing_table = get_pointing_table(source="JSOC",
        #                                     time_range=(date_start, date_end))
        # pickle.dump(pointing_table, open(pointing_table_path, 'wb'))
        # print(f"Pointing table saved: {pointing_table_path}")
        # del pointing_table

        pointing_tables = {}

        year = 2010
        year_end = 2025

        while year < year_end :
            date_start = datetime.datetime(
                year = year,
                month = 1,
                day = 1
            )
            date_start -= datetime.timedelta(days=1)
            date_start = Time(date_start, format='datetime')

            date_end = datetime.datetime(
                year = year+1,
                month = 1,
                day = 1
            )
            date_end += datetime.timedelta(days=1)
            date_end = Time(date_end, format='datetime')

            pointing_table = get_pointing_table(source="JSOC",
                                                time_range=(date_start, date_end))
            pointing_tables[str(year)] = pointing_table

            print(type(pointing_table))

        pickle.dump(pointing_tables, open(pointing_table_path, 'wb'))
        print(f"Pointing table saved: {pointing_table_path}")
        del pointing_tables

    pointing_tables = pickle.load(open(pointing_table_path, 'rb'))
    print(f"Pointing table loaded: {pointing_table_path}")
    return pointing_tables


def load_correction_table(correction_table_path):
    if os.path.exists(correction_table_path) is False :
        correction_table = get_correction_table(source="JSOC")
        pickle.dump(correction_table, open(correction_table_path, 'wb'))
        print(f"Pointing table saved: {correction_table_path}")
        del correction_table

    correction_table = pickle.load(open(correction_table_path, 'rb'))
    print(f"Correction table loaded: {correction_table_path}")
    return correction_table


def register_map(smap, pointing_tables, correction_table):
    # 현재 잘 쿼리가 되지 않음. 나중에 디버깅 예정
    year = smap.meta["T_REC"][0:4]
    pointing_table = pointing_tables[year]
    smap = update_pointing(smap, pointing_table=pointing_table)
    smap = register(smap)    
    smap = correct_degradation(smap, correction_table=correction_table)
    return smap


def register_aia(smap, correction_table):
    smap = register(smap)
    smap = correct_degradation(smap, correction_table=correction_table)
    return smap


def register_hmi(smap, fill_val=-5000):
    smap = register(smap)
    meta = smap.meta
    data = smap.data

    image_size = data.shape
    radius = meta["R_SUN"] * 0.99
    center = int(meta["CRPIX1"]), int(meta["CRPIX2"])
    mask_type = "outer"

    mask = circle_mask(image_size, radius, center, mask_type)
    data[np.where(mask==1)] = fill_val
    smap = Map(data, meta)
    return smap


def register_smap(smap, correction_table):
    instrument = smap.meta["TELESCOP"].split('/')[1].lower()
    if instrument == "aia":
        fill_val = 0.
        smap = register_aia(smap, correction_table=correction_table)        
    elif instrument == "hmi":
        fill_val = -5000.
        smap = register_hmi(smap, fill_val=fill_val)
        
    if smap.data.shape != (4096, 4096):
        meta = smap.meta
        data = smap.data
        data = pad_image(data, (4096, 4096), fill_val)
        meta["CRPIX1"] = 2047.5
        meta["CRPIX2"] = 2047.5
        meta["NAXIS1"] = 4096
        meta["NAXIS2"] = 4096
        smap = Map(data, meta)

    return smap


def to_image_aia(smap):
    data = smap.data / smap.meta["EXPTIME"]
    data = np.clip(data+ 1, 1, None)
    data = np.log2(data) * (255./14.)
    image = np.clip(data, 0, 255).astype(np.uint8)
    return image


def to_image_hmi(smap):
    data = (smap.data + 100.) * (255./200.)
    image = np.clip(data, 0, 255).astype(np.uint8)
    return image


def to_image(smap):
    instrument = smap.meta["TELESCOP"].split('/')[1].lower()
    if instrument == "aia":
        image = to_image_aia(smap)
    elif instrument == "hmi":
        image = to_image_hmi(smap)
    return image


def main(file_path, correction_table) :

    save_path = file_path.replace("fits", "png")
    if os.path.exists(save_path) is True :
        return True, file_path
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    try:
        smap = Map(file_path)
        quality = smap.meta["QUALITY"]
        if quality == 0 :
            smap = register_smap(smap=smap, correction_table=correction_table)
            image = to_image(smap)
            image = resize_image(image, (64, 64))
            image = Image.fromarray(image)
            image.save(save_path)#, compress_level=9, optimize=True)
            return True, file_path
        else :
            return False, file_path
    except:
        return False, file_path
    

if __name__ == "__main__" :
    # freeze_support()

    correction_table = load_correction_table(correction_table_path=CORRECTION_TABLE_PATH)
    # pointing_tables = load_pointing_table(pointing_table_path=POINTING_TABLE_PATH)
    

    file_list_aia = glob(f"{DATA_ROOT}/fits/aia/*/*/*.fits")#[:100]
    print(len(file_list_aia))

    file_list_hmi = glob(f"{DATA_ROOT}/fits/hmi/*/*/*.fits")#[:100]
    print(len(file_list_hmi))



    file_list = file_list_aia + file_list_hmi
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(main, file_path, correction_table): file_path for file_path in file_list}
        results = [future.result() for future in futures]

    for result in results :
        status, file_path = result
        if status is False :
            print(file_path)


# import os
# import shutil
# from pathlib import Path
# from typing import List, Optional, Dict
# from egghouse.database import PostgresManager

# # ============================================================================
# # 설정
# # ============================================================================

# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 5432,
#     'database': 'sdo_data',
#     'user': 'eunsupark',
#     'password': 'eunsupark',
#     'log_queries': False
# }

# TABLE_NAME = 'aia_193'
# TRASH_DIR = '/archive/sdo/trash'  # 삭제된 파일 보관 위치