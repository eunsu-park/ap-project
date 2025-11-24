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

from egghouse.image import resize_image, rotate_image, circle_mask, pad_image
from egghouse.sdo.aia import aia_intscale
from egghouse.sdo.hmi import hmi_intscale


DATA_ROOT = "/Users/eunsupark/Data/sdo"
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
correction_table = load_correction_table(correction_table_path=CORRECTION_TABLE_PATH)


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
    meta = smap.meta
    data = smap.data

    if data.shape != (4096, 4096):
        data = pad_image(data, (4096, 4096), -5000.)
        meta["CRPIX1"] = 2047.5
        meta["CRPIX2"] = 2047.5
        meta["NAXIS1"] = 4096
        meta["NAXIS2"] = 4096

        smap = Map(data, meta)

    return smap


def register_hmi(smap):
    smap = register(smap)
    meta = smap.meta
    data = smap.data

    image_size = data.shape
    radius = meta["R_SUN"] * 0.99
    center = int(meta["CRPIX1"]), int(meta["CRPIX2"])
    mask_type = "outer"

    mask = circle_mask(image_size, radius, center, mask_type)
    data[np.where(mask==1)] = -5000.

    if data.shape != (4096, 4096):
        data = pad_image(data, (4096, 4096), -5000.)
        meta["CRPIX1"] = 2047.5
        meta["CRPIX2"] = 2047.5
        meta["NAXIS1"] = 4096
        meta["NAXIS2"] = 4096

    smap = Map(data, meta)
    return smap


def main_aia(file_path) :
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    smap = Map(file_path)
    quality = smap.meta["QUALITY"]
    if quality == 0 :
        smap = register_aia(smap=smap, correction_table=correction_table)
        data = smap.data / smap.meta["EXPTIME"]
        data = np.clip(data+ 1, 1, None)
        data = np.log2(data) * (255./14.)
        image = np.clip(data, 0, 255).astype(np.uint8)
        # image = aia_intscale(smap.data, exptime=smap.meta["EXPTIME"], wavelnth=smap.meta["WAVELNTH"], bytescale=True)
        image = resize_image(image, (64, 64))
        image = Image.fromarray(image)
        save_path = f"{DATA_ROOT}/png/aia/{file_name}.png"
        image.save(save_path)#, compress_level=9, optimize=True)
    else :
        print(f"{file_path}: {quality}")


def main_hmi(file_path) :
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    smap = Map(file_path)
    quality = smap.meta["QUALITY"]
    if quality == 0 :
        smap = register_hmi(smap)
        image = hmi_intscale(smap.data)
        image = resize_image(image, (64, 64))
        image = Image.fromarray(image)
        save_path = f"{DATA_ROOT}/png/hmi/{file_name}.png"
        image.save(save_path)#, compress_level=9, optimize=True)
    else :
        print(f"{file_path}: {quality}")


if __name__ == "__main__" :
    freeze_support()

    # pointing_tables = load_pointing_table(pointing_table_path=POINTING_TABLE_PATH)
    

    file_list_aia = glob(f"{DATA_ROOT}/fits/aia/*.fits")
    print(len(file_list_aia))

    pool = Pool(8)
    pool.map(main_aia, file_list_aia)
    pool.close()

    # file_list_hmi = glob(f"{DATA_ROOT}/fits/hmi/*.fits")[:100]
    # print(len(file_list_hmi))

    # pool = Pool(8)
    # pool.map(main_hmi, file_list_hmi)
    # pool.close()



    # # median_dict = get_reference_median(reference_median_path=REFERENCE_MEDIAN_PATH)

    # for file_path in file_list_aia :
    #     main_aia(file_path, pointing_tables, correction_table)



    # file_list_hmi = glob(f"{DATA_ROOT}/hmi/*.fits") # [:1000]
    # print(len(file_list_hmi))

    # pool = Pool(4)
    # pool.map(main_hmi, file_list_hmi)
    # pool.close()

    # N = 0python
    # for file_path in file_list :
    #     main(file_path)

    #     N += 1

    #     if N == 0 :
    #         break


# # smap.peek()
# image = aia_intscale(smap.data, exptime=smap.meta["EXPTIME"], wavelnth=smap.meta["WAVELNTH"], bytescale=True)
# # plt.imshow(image)
# # plt.show()
# img = Image.fromarray(image)
# img.save("original.png")

# smap = normalize_map(smap)
# print(get_median_value(smap))
# # smap.peek()
# image = aia_intscale(smap.data, exptime=smap.meta["EXPTIME"], wavelnth=smap.meta["WAVELNTH"], bytescale=True)
# # plt.imshow(image)
# # plt.show()
# img = Image.fromarray(image)
# img.save("processed.png")


# date = datetime.datetime(
#     year = 2024,
#     month = 1,
#     day = 1,
#     hour = 1,
# )

# df = pd.read_csv(CSV_PATH, parse_dates=['datetime'])
# mask = df['datetime'] == date

# if mask.any() :
#     row = df.loc[mask].iloc[0]
#     aia_193 = row['aia_193']
#     aia_211 = row['aia_211']
#     hmi_magnetogram = row['hmi_magnetogram']

# # if not pd.isna(aia_193) :

# sdo_193 = f"{DATA_ROOT}/aia/{aia_193}"
# print(sdo_193)

# smap = register_map(sdo_193)
# print(get_median_value(smap))
# # smap.peek()
# image = aia_intscale(smap.data, exptime=smap.meta["EXPTIME"], wavelnth=smap.meta["WAVELNTH"], bytescale=True)
# # plt.imshow(image)
# # plt.show()
# img = Image.fromarray(image)
# img.save("original.png")

# smap = normalize_map(smap)
# print(get_median_value(smap))
# # smap.peek()
# image = aia_intscale(smap.data, exptime=smap.meta["EXPTIME"], wavelnth=smap.meta["WAVELNTH"], bytescale=True)
# # plt.imshow(image)
# # plt.show()
# img = Image.fromarray(image)
# img.save("processed.png")
