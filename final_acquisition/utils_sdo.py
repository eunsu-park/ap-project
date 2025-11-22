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

from utils_image import bytscl



def load_pointing_table(pointing_table_path):
    if os.path.exists(pointing_table_path) is False :

        date_start = datetime.datetime(
            year = 2010,
            month = 9,
            day = 1,
        )
        date_start -= datetime.timedelta(days=1)
        date_start = Time(date_start, format='datetime')
        date_end = datetime.datetime(
            year = 2025,
            month = 1,
            day = 1,
        )
        date_end += datetime.timedelta(days=1)
        date_end = Time(date_end, format='datetime')

        pointing_table = get_pointing_table(source="JSOC",
                                            time_range=(date_start, date_end))
        pickle.dump(pointing_table, open(pointing_table_path, 'wb'))
        print(f"Pointing table saved: {pointing_table_path}")
        del pointing_tables


        # pointing_tables = {}

        # year = 2010
        # year_end = 2025

        # while year < year_end :
        #     date_start = datetime.datetime(
        #         year = year,
        #         month = 1,
        #         day = 1
        #     )
        #     date_start -= datetime.timedelta(days=1)
        #     date_start = Time(date_start, format='datetime')

        #     date_end = datetime.datetime(
        #         year = year+1,
        #         month = 1,
        #         day = 1
        #     )
        #     date_end += datetime.timedelta(days=1)
        #     date_end = Time(date_end, format='datetime')

        #     pointing_table = get_pointing_table(source="JSOC",
        #                                         time_range=(date_start, date_end))
        #     pointing_tables[str(year)] = pointing_table

        #     print(type(pointing_table))

        # pickle.dump(pointing_tables, open(pointing_table_path, 'wb'))
        # print(f"Pointing table saved: {pointing_table_path}")
        # del pointing_tables

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

