import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import h5py



LOAD_ROOT = "/Users/eunsupark/projects/ap/data"
SAVE_ROOT = "/Users/eunsupark/projects/ap/datasets/original"


def print_info(arr):
    print(f"{arr.shape}, {arr.min()}. {arr.max()}")


def main(dir_path):
    if not os.path.isdir(dir_path) :
        return 
    
    aia_193 = np.load(f"{dir_path}/aia_193.npy")[:40]
    aia_211 = np.load(f"{dir_path}/aia_211.npy")[:40]
    hmi_magnetogram = np.load(f"{dir_path}/hmi_magnetogram.npy")[:40]
    omni = pd.read_csv(f"{dir_path}/omni.csv")
    omni.sort_values(by='datetime', ascending=True)
    columns = omni.columns.tolist()

    dir_name = os.path.basename(dir_path)
    save_path = f"{SAVE_ROOT}/{dir_name}.h5"

    with h5py.File(save_path, 'w') as f :
        f.create_dataset(
            f"sdo/aia_193",
            data=aia_193
        )
        f.create_dataset(
            f"sdo/aia_211",
            data=aia_211
        )
        f.create_dataset(
            f"sdo/hmi_magnetogram",
            data=hmi_magnetogram
        )

        for column in columns :
            if column == "datetime" :
                continue

            else :
                f.create_dataset(f"omni/{column}", data=omni[column])

if __name__ == "__main__" :
    dir_list = glob(f"{LOAD_ROOT}/*")
    print(len(dir_list))

    # dir_path = dir_list[0]
    # main(dir_path)

    with ProcessPoolExecutor(max_workers=8) as executor:
        future = {executor.submit(main, dir_path): dir_path for dir_path in dir_list}
