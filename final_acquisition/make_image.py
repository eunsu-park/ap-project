import os
from glob import glob
from multiprocessing import Pool, freeze_support

import pandas as pd
from sunpy.map import Map
from PIL import Image
from aiapy.calibrate import register, update_pointing, correct_degradation

from utils_image import resize_image, rotate_image
from utils_sdo import aia_intscale, hmi_intscale


DATA_ROOT = "/Users/eunsupark/Data/sdo"
POINTING_TABLE_PATH = f"{DATA_ROOT}/fits/pointing_table.pkl"
CORRECTION_TABLE_PATH = f"{DATA_ROOT}/fits/correction_table.pkl"
REFERENCE_MEDIAN_PATH = f"{DATA_ROOT}/fits/reference_median_values.pkl"



def register_map(smap, pointing_tables, correction_table):
    # 현재 잘 쿼리가 되지 않음. 나중에 디버깅 예정
    year = smap.meta["T_REC"][0:4]
    pointing_table = pointing_tables[year]
    smap = update_pointing(smap, pointing_table=pointing_table)
    smap = register(smap)    
    smap = correct_degradation(smap, correction_table=correction_table)
    return smap


def main_aia(file_path, pointing_tables, correction_table) :
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    smap = Map(file_path)
    quality = smap.meta["QUALITY"]
    if quality == 0 :
        smap = register_map(smap=smap, pointing_tables=pointing_tables, correction_table=correction_table)
        image = aia_intscale(smap.data, exptime=smap.meta["EXPTIME"], wavelnth=smap.meta["WAVELNTH"], bytescale=True)
        image = resize_image(image, (64, 64))
        image = Image.fromarray(image)
        save_path = f"{DATA_ROOT}/png/aia/{file_name}.png"
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
        image = resize_image(image, (64, 64))
        image = Image.fromarray(image)
        save_path = f"{DATA_ROOT}/png/hmi/{file_name}.png"
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
