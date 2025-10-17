import os
from glob import glob
from typing import Dict, List, Tuple
import h5py
import numpy as np
import hydra


def read_h5(file_path: str, wavelengths: List[str], variables: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:

            # Read sdo data
            sdo_data = {}
            for wavelength in wavelengths:
                data_name = f"sdo_{wavelength}"
                if data_name in f :
                    sdo_data[wavelength] = f[data_name][:]
                        
            # Read omni data
            omni_data = {}
            for variable in variables:
                dataset_name = f"omni_{variable}"
                if dataset_name in f:
                    omni_data[variable] = f[dataset_name][:]
                else:
                    raise KeyError(f"Input variable {variable} not found in {file_path}")
                                
    except (OSError, h5py.error.HDF5Error) as e:
        raise OSError(f"Failed to read HDF5 file {file_path}: {e}")
    
    return sdo_193, sdo_211, omni_data


def validate_data(sdo_193: np.ndarray, sdo_211: np.ndarray, omni_data: Dict[str, np.ndarray]) -> bool:
    # Check for NaN values in the data
    if np.isnan(sdo_193).any() or np.isnan(sdo_211).any():
        return False
    if sdo_193.shape != (20, 1, 64, 64) :
        return False
    if sdo_211.shape != (20, 1, 64, 64) :
        return False
    for var, data in omni_data.items():
        if np.isnan(data).any():
            return False
        if data.shape != (65,) :
            return False
    return True


def validate_and_copy(file_path, variables, save_dir):
    sdo_193, sdo_211, omni_data = read_h5(file_path, variables)
    if validate_data(sdo_193, sdo_211, omni_inputs, omni_targets) is True:
        print(f"{file_path} : Data validation passed: No NaN values found and shapes are correct.")
        os.system(f"cp {file_path} {save_dir}/")
        return True
    else:
        print(f"{file_path} : Data validation failed: NaN values found or incorrect shapes.")
        return False
    

def validate_and_thresholding(file_path, config, save_dir):
    
    input_variables = config.data.input_variables
    target_variables = config.data.target_variables
    target_event_threshold = config.data.target_event_threshold
    input_sequence_length = config.data.input_sequence_length
    target_sequence_length = config.data.target_sequence_length

    sdo_193, sdo_211, omni_inputs, omni_targets = read_h5(
        file_path, input_variables, target_variables
    )

    is_valid = validate_data(sdo_193, sdo_211, omni_inputs, omni_targets)
    if is_valid is True:
        print(f"{file_path} : Data validation passed: No NaN values found and shapes are correct.")
    else:
        print(f"{file_path} : Data validation failed: NaN values found or incorrect shapes.")
        return False
    
    sdo = []
    sdo.append(sdo_193)
    sdo.append(sdo_211)
    sdo = np.concatenate(sdo, axis=1)  # Shape: (20, 2, 64, 64)
    sdo = np.transpose(sdo, (1, 0, 2, 3))  # Shape: (2, 20, 64, 64)

    inputs = []
    for var in input_variables:
        inputs.append(omni_inputs[var][:input_sequence_length])
    inputs = np.array(inputs)  # Shape: (N, input_sequence_length)

    targets = []
    for var in target_variables:
        targets.append(omni_targets[var][input_sequence_length:input_sequence_length+target_sequence_length])
    targets = np.array(targets)  # Shape: (M, target_sequence_length)

    max_values = np.max(targets, axis=1)
    labels = np.float32(max_values >= target_event_threshold)  # (num_groups, num_vector)

    if labels[0] == 0 :
        save_path = f"{save_dir}/negative/{os.path.basename(file_path)}"
    else:
        save_path = f"{save_dir}/positive/{os.path.basename(file_path)}"
    
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('sdo', data=sdo, compression="gzip")
        f.create_dataset('inputs', data=inputs, compression="gzip")
        f.create_dataset('targets', data=targets, compression="gzip")
        f.create_dataset('labels', data=labels, compression="gzip")
    
    return True


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    data_root = config.environment.data_root
    dataset_name = config.data.dataset_name

    data_path = f"{data_root}/original"
    save_dir = f"{data_root}/{dataset_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = glob(f"{data_path}/*.h5")
    num_files = len(files)
    print(f"Number of .h5 files in {data_path}: {num_files}")
    if num_files == 0:
        print(f"No .h5 files found in {data_path}. Please check the data path.")
        return

    if not os.path.exists(f"{save_dir}/negative"):
        os.makedirs(f"{save_dir}/negative")
    if not os.path.exists(f"{save_dir}/positive"):
        os.makedirs(f"{save_dir}/positive")

    num = 0

    for file_path in files :
        # num += validate_and_copy(file_path, input_variables, target_variables, save_dir)
        num += validate_and_thresholding(file_path, config, save_dir)
        # break
    print(f"Number of valid files copied to {save_dir}: {num}")

if __name__ == "__main__":
    main()