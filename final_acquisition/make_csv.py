import os
from glob import glob

from sunpy.map import Map
import pandas as pd


DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
CSV_PATH = f"{DATA_ROOT}/sdo_fits.csv"


def parse(file_path):
    sdo_map = Map(file_path)
    meta = sdo_map.meta
    print(file_path)
    print(meta["QUALITY"])
    print(meta["T_REC"])
    # print(Path.)
    # print(meta[""])
    

instrument = "hmi"
instrument_lower = instrument.lower()

sdo_path = os.path.join(DATA_ROOT, instrument_lower)
file_list = glob(os.path.join(DATA_ROOT, instrument_lower, "*.fits"))

print(len(file_list))

# file_path = file_list[0]

# parse(file_path)

    
