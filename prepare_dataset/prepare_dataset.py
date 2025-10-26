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


def validate_sdo(sdo_data:Dict[str, np.ndarray],
                 sdo_image_size:int,
                 sdo_sequence_length:int) -> bool:
    # Check for NaN values in the data
    sdo_shape = (sdo_sequence_length, 1, sdo_image_size, sdo_image_size)
    for var, data in sdo_data.items():
        if np.isnan(data).any() :
            return False
        if data.shape != sdo_shape:
            return False
    return True


def validate_omni(omni_data: Dict[str, np.ndarray],
                  input_sequence_length:int,
                  target_sequence_length:int) -> bool:
    omni_shape = (input_sequence_length + target_sequence_length + 1, )
    for var, data in omni_data.items():
        if np.isnan(data).any():
            return False
        if data.shape != omni_shape :
            return False
    return True



class ValidateData:
    def __init__(self, config):

        self.sdo_wavelengths = config.data.sdo_wavelengths
        self.sdo_image_size = config.data.sdo_image_size
        self.sdo_sequence_length = config.data.sdo_sequence_length

        self.input_variables = config.data.input_variables
        self.input_sequence_length = config.data.input_sequence_length

        self.target_variables = config.data.target_variables
        self.target_sequence_length = config.data.target_sequence_length

        self.omni_variables = list(set(self.input_variables + self.target_variables))
        self.split_index = config.data.split_index

        self.sdo_shape = (self.sdo_sequence_length, 1, self.sdo_image_size, self.sdo_image_size)
        self.omni_shape = (self.input_sequence_length + self.target_sequence_length + 1, )


    def read_h5(self, file_path):

        with h5py.File(file_path, 'r') as f:
            # Read SDO data
            sdo_data = {}
            for wavelength in self.sdo_wavelengths:
                dataset_name = f"sdo_{wavelength}"
                if dataset_name in f:
                    sdo_data[wavelength] = f[dataset_name][:]
                else:
                    raise KeyError(f"SDO wavelength {wavelength} not found in {file_path}")

            # Read OMNI data
            omni_data = {}
            for variable in self.omni_variables:
                dataset_name = f"omni_{variable}"
                if dataset_name in f:
                    omni_data[variable] = f[dataset_name][:]
                else:
                    raise KeyError(f"OMNI variable {variable} not found in {file_path}")

        return sdo_data, omni_data

    def validate_sdo(self, sdo_data:Dict[str, np.ndarray]) -> bool:
        for var, data in sdo_data.items():
            if np.isnan(data).any() :
                return False
            if data.shape != self.sdo_shape:
                return False
        return True
            
    def validate_omni(self, omni_data: Dict[str, np.ndarray]) -> bool:
        for var, data in omni_data.items():
            if np.isnan(data).any():
                return False
            if data.shape != self.omni_shape :
                return False
        return True
    
    def validate_data(self, sdo_data: Dict[str, np.ndarray], omni_data: Dict[str, np.ndarray]) -> bool:
        if self.validate_sdo(sdo_data) is False :
            return False
        if self.validate_omni(omni_data) is False :
            return False
        return True
    
    def parse_class(self, omni_data):
        ap_index = omni_data["ap_index"]
        targets = ap_index[self.split_index:self.split_index+self.target_sequence_length]
        ap_max = []
        ap_class = []
        for n in range(self.target_sequence_length//8):
            day = targets[n*8:(n+1)*8]
            ap_max.append(int(np.max(day)))
            if np.max(day) >= 48 :
                ap_class.append(1)
            else :
                ap_class.append(0)
        return ap_max, ap_class
    
    def __call__(self, file_path):
        sdo_data, omni_data = self.read_h5(file_path)
        if self.validate_data(sdo_data, omni_data) is False :
            return False, sdo_data, omni_data
        return True, sdo_data, omni_data


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    data_root = config.environment.data_root
    dataset_name = config.data.dataset_name

    data_path = f"{data_root}/original"
    save_path = f"{data_root}/{dataset_name}"

    validate_data = ValidateData(config)

    ## validate data
    files = glob(f"{data_path}/*.h5")
    num_files = len(files)
    print(f"Number of .h5 files in {data_path}: {num_files}")
    if num_files == 0:
        print(f"No .h5 files found in {data_path}. Please check the data path.")
        return

    file_name = []
    day1_max = []
    day2_max = []
    day3_max = []
    day4_max = []
    day5_max = []
    day1_class = []
    day2_class = []
    day3_class = []
    day4_class = []
    day5_class = []

    n = 0
    for file_path in files :
        is_valid, sdo_data, omni_data = validate_data(file_path)
        if is_valid is True :
            file_name.append(os.path.basename(file_path))
            ap_max, ap_class = validate_data.parse_class(omni_data)
            day1_max.append(ap_max[0])
            day2_max.append(ap_max[1])
            day3_max.append(ap_max[2])
            day4_max.append(ap_max[3])
            day5_max.append(ap_max[4])
            day1_class.append(ap_class[0])
            day2_class.append(ap_class[1])
            day3_class.append(ap_class[2])
            day4_class.append(ap_class[3])
            day5_class.append(ap_class[4])
            print(file_path, ap_max, ap_class)
            n += 1
        # if n == 10 :
        #     break

    df = pd.DataFrame({'file_name': file_name,
                       'max_day1': day1_max,
                       'max_day2': day2_max,
                       'max_day3': day3_max,
                       'max_day4': day4_max,
                       'max_day5': day5_max,
                       'class_day1': day1_class,
                       'class_day2': day2_class,
                       'class_day3': day3_class,
                       'class_day4': day4_class,
                       'class_day5': day5_class
                       })
    df.to_csv(f"{save_path}.csv", index=False, encoding='utf-8')

    print(f"Number of valid files: {len(file_name)}")

if __name__ == "__main__":
    main()
