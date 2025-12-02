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
    
    aia_193 = np.load(f"{dir_path}/aia_193.npy")
    aia_211 = np.load(f"{dir_path}/aia_211.npy")
    hmi_magnetogram = np.load(f"{dir_path}/hmi_magnetogram.npy")
    omni = pd.read_csv(f"{dir_path}/omni.csv")
    omni.sort_values(by='datetime', ascending=True)
    columns = omni.columns.tolist()

    dir_name = os.path.basename(dir_path)
    save_path = f"{SAVE_ROOT}/{dir_name}.h5"

    with h5py.File(save_path, 'w') as f :
        f.create_dataset(
            f"sdo/aia_193",
            data=aia_193,
            compression='gzip'
        )
        f.create_dataset(
            f"sdo/aia_211",
            data=aia_211,
            compression='gzip'
        )
        f.create_dataset(
            f"sdo/hmi_magnetogram",
            data=hmi_magnetogram,
            compression='gzip'
        )

        for column in columns :
            # datetime 타입 처리
            if pd.api.types.is_datetime64_any_dtype(omni[column]):
                # 방법 1: Unix timestamp로 변환 (초 단위)
                column_data = column_data.astype('datetime64[s]').astype(np.int64)
                f.create_dataset(
                    f"omni/{column}",
                    data=column_data,
                    compression='gzip'
                )
                # 메타데이터로 원래 타입 저장
                f[f"omni/{column}"].attrs['dtype'] = 'datetime64[s]'
                
            # 문자열 타입 처리  
            elif omni[column].dtype == 'object':
                # 문자열을 bytes로 변환
                column_data = omni[column].astype(str).values.astype('S')
                f.create_dataset(
                    f"omni/{column}",
                    data=column_data,
                    compression='gzip'
                )
                
            # 숫자 타입 (정상 저장)
            else:
                f.create_dataset(
                    f"omni/{column}",
                    data=column_data,
                    compression='gzip'
                )


if __name__ == "__main__" :
    dir_list = glob(f"{LOAD_ROOT}/*")
    print(len(dir_list))

    with ProcessPoolExecutor(max_workers=8) as executor:
        future = {executor.submit(main, dir_path): dir_path for dir_path in dir_list}
