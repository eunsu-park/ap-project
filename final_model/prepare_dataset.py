import os
from glob import glob
from typing import Dict, List, Tuple
import h5py
import numpy as np
import hydra
import pandas as pd
import datetime
import pickle


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


def get_test_months(year:int, start_year:int=2010, start_month:int=11):
    offset = ((year - start_year) * 2 + start_month) % 12
    m1 = offset
    m2 = (m1 % 12) + 1
    return m1, m2


def split_tv(csv_path:str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    data_list = df["file_name"].tolist()    
    train_list = []
    test_list = []
    for file_name in data_list:
        file_date = datetime.datetime.strptime(
            file_name, '%Y%m%d%H.h5')
        file_year = file_date.year
        file_month = file_date.month
        if file_month in get_test_months(file_year) :
            test_list.append(file_name)
        else :
            train_list.append(file_name)
    print(len(train_list), len(test_list))
    return train_list, test_list


def compute_stats(data_path:str, train_list:List[str], stat_file_path:str, variables:List[str]) -> dict:

    total_dict = {variable: [] for variable in variables}
    stat_dict = {}

    for file_name in train_list:
        file_path = f"{data_path}/{file_name}"
        with h5py.File(file_path, 'r') as f:
            for variable in variables:
                dataset_name = f"omni_{variable}"
                data = f[dataset_name][:]
                total_dict[variable].append(data)

    for variable in variables :
        concatenated_data = np.concatenate(total_dict[variable], axis=0)
        stat_dict[variable] = {
            'mean' : np.mean(concatenated_data),
            'std' : np.std(concatenated_data)
        }

    for variable in variables :
        mean = stat_dict[variable]['mean']
        std = stat_dict[variable]['std']
        print(f'{variable} : {mean:.3f} +- {std:.3f}')

    pickle.dump(stat_dict, open(stat_file_path, 'wb'))

    return stat_dict


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    do_validate = False
    do_split = False
    do_analyze = True

    data_root = config.environment.data_root
    dataset_name = config.data.dataset_name

    data_path = f"{data_root}/original"
    save_path = f"{data_root}/{dataset_name}"

    sdo_wavelengths = config.data.sdo_wavelengths
    input_variables = config.data.input_variables
    target_variables = config.data.target_variables
    omni_variables = list(set(input_variables + target_variables))


    ## validate data
    if do_validate is True :
        files = glob(f"{data_path}/*.h5")
        num_files = len(files)
        print(f"Number of .h5 files in {data_path}: {num_files}")
        if num_files == 0:
            print(f"No .h5 files found in {data_path}. Please check the data path.")
            return

        file_name = []

        for file_path in files :
            is_valid = read_and_validate(file_path, sdo_wavelengths, omni_variables)
            if is_valid is True :
                file_name.append(os.path.basename(file_path))

        df = pd.DataFrame({'file_name': file_name})
        df.to_csv(f"{save_path}.csv", index=False, encoding='utf-8')

        print(f"Number of valid files: {len(file_name)}")

    ## split dataset into train and validation
    if do_split is True :
        train_list, test_list = split_tv(f"{save_path}.csv")

        df = pd.DataFrame({'file_name': train_list})
        df.to_csv(f"{save_path}_train.csv", index=False, encoding='utf-8')

        df = pd.DataFrame({'file_name': test_list})
        df.to_csv(f"{save_path}_validation.csv", index=False, encoding='utf-8')

    ## get statistics from train dataset
    if do_analyze is True :
        stat_file_path = f"{save_path}.pkl"
        df = pd.read_csv(f"{save_path}_train.csv")
        train_list = df["file_name"].tolist()    
        _ = compute_stats(data_path, train_list, stat_file_path, omni_variables)
        stat_dict = pickle.load(open(stat_file_path, 'rb'))
        for variable in omni_variables:
            mean = stat_dict[variable]['mean']
            std = stat_dict[variable]['std']
            print(f'{variable} : {mean:.3f} +- {std:.3f}')


if __name__ == "__main__":
    main()