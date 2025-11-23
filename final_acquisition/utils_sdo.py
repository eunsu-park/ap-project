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





