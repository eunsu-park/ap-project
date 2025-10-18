import os
from glob import glob
from typing import Dict, List, Tuple
import h5py
import numpy as np
import hydra
import pandas as pd


def read_h5(file_path: str, sdo_wavelengths: List[str], omni_variables: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Read SDO data
            sdo_data = {}
            for wavelength in sdo_wavelengths:
                dataset_name = f"sdo_{wavelength}"
                if dataset_name in f:
                    sdo_data[wavelength] = f[dataset_name][:]
                else:
                    raise KeyError(f"SDO wavelength {wavelength} not found in {file_path}")

            # Read OMNI data
            omni_data = {}
            for variable in omni_variables:
                dataset_name = f"omni_{variable}"
                if dataset_name in f:
                    omni_data[variable] = f[dataset_name][:]
                else:
                    raise KeyError(f"OMNI variable {variable} not found in {file_path}")

    except (OSError, h5py.error.HDF5Error) as e:
        raise OSError(f"Failed to read HDF5 file {file_path}: {e}")

    return sdo_data, omni_data


def validate_data(sdo_data:Dict[str, np.ndarray], omni_data: Dict[str, np.ndarray]) -> bool:
    # Check for NaN values in the data
    for var, data in sdo_data.items():
        if np.isnan(data).any() :
            return False
        if data.shape != (20, 1, 64, 64):
            return False
    for var, data in omni_data.items():
        if np.isnan(data).any():
            return False
        if data.shape != (65,) :
            return False
    return True


def read_and_validate(file_path:str, sdo_wavelengths:List[str], omni_variables:List[str]) -> bool:
    sdo_data, omni_data = read_h5(file_path, sdo_wavelengths, omni_variables)
    is_valid = validate_data(sdo_data, omni_data)
    if is_valid is True :
        # print(f"{file_path} : Data validation passed: No NaN values found and shapes are correct.")
        return True
    else:
        # print(f"{file_path} : Data validation failed: NaN values found or incorrect shapes.")
        return False


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    data_root = config.environment.data_root
    dataset_name = config.data.dataset_name

    data_path = f"{data_root}/original"
    save_path = f"{data_root}/{dataset_name}.csv"

    files = glob(f"{data_path}/*.h5")
    num_files = len(files)
    print(f"Number of .h5 files in {data_path}: {num_files}")
    if num_files == 0:
        print(f"No .h5 files found in {data_path}. Please check the data path.")
        return

    sdo_wavelengths = config.data.sdo_wavelengths
    input_variables = config.data.input_variables
    target_variables = config.data.target_variables
    omni_variables = list(set(input_variables + target_variables))

    file_name = []

    for file_path in files :
        is_valid = read_and_validate(file_path, sdo_wavelengths, omni_variables)
        if is_valid is True :
            file_name.append(os.path.basename(file_path))

    df = pd.DataFrame({'file_name': file_name})
    df.to_csv(save_path, index=False, encoding='utf-8')

    print(f"Number of valid files: {len(file_name)}")


if __name__ == "__main__":
    main()