import os
import datetime
import pickle
from glob import glob
from multiprocessing import Pool, freeze_support

import pandas as pd
import matplotlib.pyplot as plt
from sunpy.map import Map
import numpy as np
from PIL import Image
from aiapy.calibrate import register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy.time import Time
from aiapy.calibrate.util import get_pointing_table


DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
CSV_PATH = f"{DATA_ROOT}/sdo_fits.csv"
POINTING_TABLE_PATH = f"{DATA_ROOT}/pointing_table.pkl"
CORRECTION_TABLE_PATH = f"{DATA_ROOT}/correction_table.pkl"
REFERENCE_MEDIAN_PATH = f"{DATA_ROOT}/reference_median_values.pkl"


def load_pointing_table(pointing_table_path):
    if os.path.exists(pointing_table_path) is False :

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


def circle_mask(image_size, radius, mask_type='inner'):
    center = image_size / 2.0
    y, x = np.ogrid[:image_size, :image_size]
    distance_from_center = np.sqrt((x - center)**2 + (y - center)**2)

    if mask_type == 'inner' :
        mask = distance_from_center < radius
    elif mask_type == "outer" :
        mask = distance_from_center >= radius
    else :
        raise ValueError("mask_type should be 'inner' or 'outer'.")
    return mask


def get_median_value(smap):
    data = smap.data
    meta = smap.meta
    radius = meta["R_SUN"]
    mask = circle_mask(
        image_size=data.shape[0],
        radius = radius,# * (np.sqrt(3.0) / 2.),
        mask_type = 'inner'
    )
    masked = data[mask==1]
    median_value = np.median(masked)
    return median_value


def get_reference_median(reference_median_path):
    if os.path.exists(reference_median_path) is False :
        df = pd.read_csv(CSV_PATH, parse_dates=['datetime'])
        reference_date = datetime.datetime(
            year = 2011,
            month = 1,
            day = 1,
            hour = 0,
        )
        mask = df['datetime'] == reference_date
        row = df.loc[mask].iloc[0]
        aia_193 = row['aia_193']
        aia_211 = row['aia_211']
        map_aia_193 = Map(f"{DATA_ROOT}/aia/{aia_193}")
        map_aia_193 = register_map(map_aia_193)
        median_val_aia_193 = get_median_value(map_aia_193)
        map_aia_211 = Map(f"{DATA_ROOT}/aia/{aia_211}")
        map_aia_211 = register_map(map_aia_211)
        median_val_aia_211 = get_median_value(map_aia_211)
        median_dict = {"aia_193" : median_val_aia_193, "aia_211" : median_val_aia_211}
        pickle.dump(median_dict, open(reference_median_path, 'wb'))
        print(f"Reference median values saved: {reference_median_path}")
        del median_dict

    median_dict = pickle.load(open(reference_median_path, 'rb'))
    print(f"Reference median values loaded: {reference_median_path}")
    return median_dict


def normalize_map(smap, median_dict):
    median_value = get_median_value(smap)
    data = smap.data
    meta = smap.meta
    wavelength = int(meta["WAVELNTH"])
    reference_median_value = median_dict[f"aia_{wavelength}"]
    data = data * (reference_median_value/median_value)
    smap_new = Map(data, meta)
    return smap_new


def bytscl(data, vmin, vmax, bottom=0, top=255):

    data = (data - vmin) / (vmax - vmin)
    data = (data + bottom) * (top - bottom)
    image = np.clip(data, 0, 255).astype(np.uint8)

    return image


def aia_intscale(image, exptime=None, wavelnth=None, bytescale=False):

    image[np.isnan(image)] = 0.

    wavelnth = np.rint(wavelnth)
    
    if wavelnth == 94 :
        vmin, vmax = 1.5 / 1.06, 50 / 1.06
        temp = image * (4.99803 / exptime)

    elif wavelnth == 131 :
        vmin, vmax = 7.0 / 1.49, 1200 / 1.49
        temp = image * (6.99685 / exptime)

    elif wavelnth == 171 :
        vmin, vmax = 10.0 / 1.49, 6000 / 1.49
        temp = image * (4.99803 / exptime)

    elif wavelnth == 193 :
        vmin, vmax = 120.0 / 2.2, 6000.0 / 2.2
        temp = image * (2.9995 / exptime)

    elif wavelnth == 211 :
        vmin, vmax = 30.0 / 1.10, 13000 / 1.10
        temp = image * (4.99801 / exptime)

    elif wavelnth == 304 :
        vmin, vmax = 50.0 / 12.11, 2000 / 12.11
        temp = image * (4.99941 / exptime)

    elif wavelnth == 335 :
        vmin, vmax = 3.5 / 2.97, 1000 / 2.97
        temp = image * (6.99734 / exptime)

    elif wavelnth == 1600 :
        vmin, vmax = -8, 200
        temp = image * (2.99911 / exptime)

    elif wavelnth == 1700 :
        vmin, vmax = 0, 2500
        temp = image * (1.00026 / exptime)

    elif wavelnth == 4500 :
        vmin, vmax = 0, 26000
        temp = image * (1.00026 / exptime)

    elif wavelnth == 6173 :
        vmin, vmax = 0, 65535
        temp = image / exptime

    temp = np.clip(temp, vmin, vmax)
    if wavelnth in (94, 171) :
        scaled = bytscl(np.sqrt(temp), np.sqrt(vmin), np.sqrt(vmax))
    elif wavelnth in (131, 193, 211, 304, 335) :
        scaled = bytscl(np.log10(temp), np.log10(vmin), np.log10(vmax))
    elif wavelnth in (1600, 1700, 4500, 6173):
        scaled = bytscl(temp, vmin, vmax)

    return scaled


def hmi_intscale(data):
    data = np.asarray(data, dtype=np.float64)
    data = (data + 100.) * (255./200.)
    image = np.clip(data, 0, 255).astype(np.uint8)
    return image


def main_aia(file_path, pointing_tables, correction_table) :
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    smap = Map(file_path)
    quality = smap.meta["QUALITY"]
    if quality == 0 :
        smap = register_map(smap=smap, pointing_tables=pointing_tables, correction_table=correction_table)
        image = aia_intscale(smap.data, exptime=smap.meta["EXPTIME"], wavelnth=smap.meta["WAVELNTH"], bytescale=True)
        image = Image.fromarray(image)
        save_path = f"{DATA_ROOT}/aia_png/{file_name}.png"
        # save_path = f"/Volumes/EUNSU-T9/sdo/png/aia/{file_name}.png"
        image.save(save_path, compress_level=9, optimize=True)
    else :
        print(f"{file_path}: {quality}")


def main_hmi(file_path) :
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    smap = Map(file_path)
    quality = smap.meta["QUALITY"]
    if quality == 0 :
        smap = register_map(smap)
        image = hmi_intscale(smap.data)
        image = Image.fromarray(image)
        # save_path = f"{DATA_ROOT}/hmi_png/{file_name}.png"
        save_path = f"/Volumes/EUNSU-T9/sdo/png/hmi/{file_name}.png"
        image.save(save_path, compress_level=9, optimize=True)
    else :
        print(f"{file_path}: {quality}")

if __name__ == "__main__" :
    freeze_support()

    pointing_tables = load_pointing_table(pointing_table_path=POINTING_TABLE_PATH)
    correction_table = load_correction_table(correction_table_path=CORRECTION_TABLE_PATH)

    file_list_aia = glob(f"{DATA_ROOT}/aia/*.fits")[:10] # [:1000]
    print(len(file_list_aia))

    # median_dict = get_reference_median(reference_median_path=REFERENCE_MEDIAN_PATH)

    for file_path in file_list_aia :
        main_aia(file_path, pointing_tables, correction_table)



    # file_list_hmi = glob(f"{DATA_ROOT}/hmi/*.fits") # [:1000]
    # print(len(file_list_hmi))

    # pool = Pool(4)
    # pool.map(main_hmi, file_list_hmi)
    # pool.close()

    # N = 0
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
