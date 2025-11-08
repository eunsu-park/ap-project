import os
import sys
import time
import pickle
import random
from glob import glob
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
import hydra


def read_h5(file_path: str, sdo_wavelengths: List[str], omni_variables: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # 변수를 try 블록 밖에서 초기화
    sdo_data = {}
    omni_data = {}

    try:
        with h5py.File(file_path, 'r') as f:
            # Read SDO data
            for wavelength in sdo_wavelengths:
                dataset_name = f"sdo_{wavelength}"
                if dataset_name in f:
                    sdo_data[wavelength] = f[dataset_name][:]
                else:
                    raise KeyError(f"SDO wavelength {wavelength} not found in {file_path}")
            
            # Read OMNI data
            for variable in omni_variables:
                dataset_name = f"omni_{variable}"
                if dataset_name in f:
                    omni_data[variable] = f[dataset_name][:]
                else:
                    raise KeyError(f"OMNI variable {variable} not found in {file_path}")

    except KeyError:
        # KeyError는 그대로 전파
        raise
    except (OSError, h5py.h5py._errors.HDF5Error) as e:
        # HDF5 관련 에러는 OSError로 래핑
        raise OSError(f"Failed to read HDF5 file {file_path}: {e}")

    return sdo_data, omni_data


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    data_root = config.environment.data_root
    dataset_name = config.data.dataset_name
    dataset_path = f"{data_root}/{dataset_name}"
    train_list_path = f"{dataset_path}_train.csv"

    file_name_key = "file_name"
    train_df = pd.read_csv(train_list_path)
    train_file_name = train_df[file_name_key].tolist()

    sdo_wavelengths = config.data.sdo_wavelengths
    input_variables = config.data.input_variables
    target_variables = config.data.target_variables
    omni_variables = list(set(input_variables + target_variables))


    for file_name in train_file_name :
        file_path = f"{data_root}/original/{file_name}"
        try :
            sdo_data, omni_data = read_h5(file_path, sdo_wavelengths, omni_variables)
        except Exception as e :
            print(f"{file_name} : {e}")


if __name__ == "__main__" :
    main()
