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
    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"Data file not found: {file_path}")

    # try:
    #     with h5py.File(file_path, 'r') as f:
    #         # Read SDO data
    #         sdo_data = {}
    #         for wavelength in sdo_wavelengths:
    #             dataset_name = f"sdo_{wavelength}"
    #             if dataset_name in f:
    #                 sdo_data[wavelength] = f[dataset_name][:]
    #             else:
    #                 raise KeyError(f"SDO wavelength {wavelength} not found in {file_path}")
                
    #         omni_data = {}
    #         for variable in omni_variables:
    #             dataset_name = f"omni_{variable}"
    #             if dataset_name in f:
    #                 omni_data[variable] = f[dataset_name][:]
    #             else:
    #                 raise KeyError(f"OMNI variable {variable} not found in {file_path}")

    # except (OSError, h5py.error.HDF5Error) as e:
    #     raise OSError(f"Failed to read HDF5 file {file_path}: {e}")

    # return sdo_data, omni_data


def get_statistics(stat_file_path: str, data_root: str, data_file_list: List[str], 
                  variables: List[str], overwrite: bool = False) -> Dict[str, Dict[str, float]]:
    """Compute and cache statistics for data normalization.
    
    Args:
        stat_file_path: Path to save/load statistics pickle file.
        data_root: Root directory containing data files.
        data_file_list: List of data file names.
        variables: List of variable names to compute statistics for.
        overwrite: Whether to recompute statistics even if cache exists.
        
    Returns:
        Dictionary containing mean and std for each variable.
        
    Raises:
        FileNotFoundError: If data files cannot be found.
        ValueError: If no valid data is found for statistics computation.
    """
    # Filter for h5 files only
    data_file_list = [f"{data_root}/{f}" for f in data_file_list if f.endswith('.h5')]
    
    if not data_file_list:
        raise ValueError("No valid .h5 files found in data file list")
    
    stat_dict = {}
    
    if os.path.exists(stat_file_path) and not overwrite:
        # Load existing statistics
        try:
            loaded = pickle.load(open(stat_file_path, 'rb'))
            for variable in variables:
                stat_dict[variable] = loaded.get(variable, {})
        except (pickle.PickleError, KeyError) as e:
            print(f"Warning: Failed to load statistics from {stat_file_path}: {e}")
            print("Recomputing statistics...")
            overwrite = True
    
    if not os.path.exists(stat_file_path) or overwrite:
        # Compute statistics from scratch
        total_dict = {variable: [] for variable in variables}
        
        valid_files = 0
        for data_file_path in data_file_list:
            if not os.path.exists(data_file_path):
                print(f"Warning: File not found: {data_file_path}")
                continue
                
            try:
                with h5py.File(data_file_path, 'r') as f:
                    for variable in variables:
                        dataset_name = f"omni_{variable}"
                        if dataset_name in f:
                            data = f[dataset_name][:]
                            # Filter out NaN and infinite values
                            valid_data = data[np.isfinite(data)]
                            if len(valid_data) > 0:
                                total_dict[variable].append(valid_data)
                        else:
                            print(f"Warning: Variable {variable} not found in {data_file_path}")
                valid_files += 1
            except (OSError, KeyError) as e:
                print(f"Warning: Failed to read {data_file_path}: {e}")
                continue
        
        if valid_files == 0:
            raise ValueError("No valid data files found for statistics computation")
        
        # Compute final statistics
        for variable in variables:
            if total_dict[variable]:
                concatenated_data = np.concatenate(total_dict[variable], axis=0)
                if len(concatenated_data) > 0:
                    stat_dict[variable] = {
                        'mean': float(np.mean(concatenated_data)),
                        'std': float(np.std(concatenated_data))
                    }
                else:
                    print(f"Warning: No valid data found for variable {variable}")
                    stat_dict[variable] = {'mean': 0.0, 'std': 1.0}
            else:
                print(f"Warning: No data found for variable {variable}")
                stat_dict[variable] = {'mean': 0.0, 'std': 1.0}
        
        # Save statistics
        try:
            os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
            pickle.dump(stat_dict, open(stat_file_path, 'wb'))
            print(f"Statistics saved to {stat_file_path}")
        except (OSError, pickle.PickleError) as e:
            print(f"Warning: Failed to save statistics to {stat_file_path}: {e}")
    
    return stat_dict

# def get_statistics(stat_file_path: str, dataset_path:str,
#                    file_list: List[str], overwrite: bool = False) -> Dict[str, Dict[str, float]]:

#     data_file_list = []
#     for (file_name, file_class) in file_list :
#         data_file_list.append(f"{dataset_path}/{file_class}/{file_name}")

#     if not data_file_list:
#         raise ValueError("No valid .h5 files found in data file list")
    
#     stat_dict = {}
    
#     if os.path.exists(stat_file_path) and not overwrite:
#         # Load existing statistics
#         try:
#             stat_dict = pickle.load(open(stat_file_path, 'rb'))
#         except (pickle.PickleError, KeyError) as e:
#             print(f"Warning: Failed to load statistics from {stat_file_path}: {e}")
#             print("Recomputing statistics...")
#             overwrite = True
    
#     if not os.path.exists(stat_file_path) or overwrite:
#         # Compute statistics from scratch
#         all_data = []
#         for data_file_path in data_file_list:
#             with h5py.File(data_file_path, 'r') as f:
#                 data = f['inputs'][:]
#                 all_data.append(data)
        
#         # Compute final statistics
#         all_data = np.concatenate(all_data, axis=1)  # Shape: (num_samples, num_variables)
        
#         mean = np.mean(all_data, axis=1)
#         std = np.std(all_data, axis=1)
#         mean = np.expand_dims(mean, axis=0)
#         std = np.expand_dims(std, axis=0)
#         stat_dict = {
#             'mean': mean,
#             'std': std
#         }
        
#         # Save statistics
#         try:
#             os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
#             pickle.dump(stat_dict, open(stat_file_path, 'wb'))
#             print(f"Statistics saved to {stat_file_path}")
#         except (OSError, pickle.PickleError) as e:
#             print(f"Warning: Failed to save statistics to {stat_file_path}: {e}")
    
#     return stat_dict


def undersample(train_file_list, num_subsample, subsample_index):
    negative_pairs = []
    positive_pairs = []

    for pair in train_file_list:
        file_name, file_class = pair
        if file_class == 0 :
            negative_pairs.append(pair)
        else :
            positive_pairs.append(pair)

    random.shuffle(negative_pairs)

    n = len(negative_pairs)
    base_size = n // num_subsample
    remainder = n % num_subsample

    sublists = []
    start = 0

    for i in range(num_subsample):
        size = base_size + (1 if i < remainder else 0)
        sublists.append(negative_pairs[start:start + size])
        start += size

    sub_negative_pairs = sublists[subsample_index]
    final_pairs = positive_pairs + sub_negative_pairs
    
    return final_pairs


def oversample(train_file_list, num_oversample):

    train_file_name = []
    train_file_class = []

    for pair in train_file_list :
        file_name, file_class = pair

        if file_class == 1 :
            for m in range(num_oversample) :
                tmp = os.path.splitext(file_name)
                new_file_name = f"{tmp[0]}_{m}{tmp[1]}"
                train_file_name.append(new_file_name)
                train_file_class.append(file_class)

        else :
            train_file_name.append(file_name)
            train_file_class.append(file_class)

    return list(zip(train_file_name, train_file_class))




def get_pos_weight(train_file_list):
    negative_pairs = []
    positive_pairs = []

    for pair in train_file_list:
        _, file_class = pair
        if file_class == 0 :
            negative_pairs.append(pair)
        else :
            positive_pairs.append(pair)

    pos_weight = len(negative_pairs) / len(positive_pairs)
    return pos_weight


class CustomDataset(Dataset):
    def __init__(self, config, logger=None):
        self.phase = config.experiment.phase
        self.data_root = config.environment.data_root
        self.dataset_name = config.data.dataset_name
        self.dataset_path = f"{self.data_root}/{self.dataset_name}"

        self.train_list_path = f"{self.dataset_path}_train.csv"
        self.validation_list_path = f"{self.dataset_path}_validation.csv"
        self.stat_file_path = f"{self.dataset_path}_statistics.pkl"
        self.enable_oversampling = config.experiment.enable_oversampling

        file_name_key = "file_name"
        class_key = f"class_day{config.data.target_day}"

        # enable_undersampleing

        train_df = pd.read_csv(self.train_list_path)
        train_file_name = train_df[file_name_key].tolist()
        train_file_class = train_df[class_key].tolist()

        self.train_file_list = list(zip(train_file_name, train_file_class))

        self.pos_weight = get_pos_weight(self.train_file_list)

        validation_df = pd.read_csv(self.validation_list_path)
        validation_file_name = validation_df[file_name_key].tolist()
        validation_file_class = validation_df[class_key].tolist()
        self.validation_file_list = list(zip(validation_file_name, validation_file_class))

        print(f"Training samples: {len(self.train_file_list)}, Validation samples: {len(self.validation_file_list)}")

        self.sdo_wavelengths = config.data.sdo_wavelengths
        self.sdo_sequence_length = config.data.sdo_sequence_length
        self.sdo_image_size = config.data.sdo_image_size

        self.input_variables = config.data.input_variables
        self.input_sequence_length = config.data.input_sequence_length

        self.target_variables = config.data.target_variables
        self.target_sequence_length = config.data.target_sequence_length
        
        self.split_index = config.data.split_index
        self.omni_variables = list(set(self.input_variables + self.target_variables))

        try:
            self.stat_dict = get_statistics(
                stat_file_path = self.stat_file_path,
                data_root = f"{self.data_root}/original",
                data_file_list = train_file_name,
                variables = self.omni_variables,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load/compute statistics: {e}")
        print(f"Loaded statistics for {len(self.stat_dict)} variables.")

        if config.experiment.enable_undersampling is True :
            self.train_file_list = undersample(self.train_file_list, config.experiment.num_subsample, config.experiment.subsample_index)
            print(f"After undersamplig, Training samples: {len(self.train_file_list)}, Validation samples: {len(self.validation_file_list)}")

        if config.experiment.enable_oversampling is True :
            self.train_file_list = oversample(self.train_file_list, config.experiment.num_oversample)
            print(f"After oversamplig, Training samples: {len(self.train_file_list)}, Validation samples: {len(self.validation_file_list)}")

        if config.experiment.phase == 'train':
            self.list_data = self.train_file_list
        elif config.experiment.phase == 'validation':
            self.list_data = self.validation_file_list
        else:
            raise ValueError(f"Unknown phase: {config.experiment.phase}. Must be 'train' or 'validation'.")
        
        self.nb_data = len(self.list_data)

        self.memory_cache = {}
        self.enable_memory_cache = config.experiment.enable_memory_cache #True  # Can be disabled for low-memory scenarios


    def __len__(self):
        return self.nb_data
    
    def __getitem__(self, idx):
        file_name, file_class = self.list_data[idx]
        if self.enable_memory_cache and file_name in self.memory_cache:
            return self.memory_cache[file_name]
        
        if (self.enable_oversampling is True) and (self.phase == "train" is True):
            file_path = f"{self.data_root}/oversampling/{file_name}"
        else :
            file_path = f"{self.data_root}/original/{file_name}"
            
        sdo_data, omni_data = read_h5(file_path, self.sdo_wavelengths, self.omni_variables)
        sdo_array = []
        for wavelength in self.sdo_wavelengths :
            data = sdo_data[wavelength]
            data = (data * (2./255.)) - 1.
            sdo_array.append(data)
        sdo_array = np.concatenate(sdo_array, 1)
        sdo_array = sdo_array[-self.sdo_sequence_length:]
        sdo_array = np.transpose(sdo_array, (1, 0, 2, 3))

        input_array = []
        for variable in self.input_variables :
            data = omni_data[variable][:, None]
            mean = self.stat_dict[variable]['mean']
            std = self.stat_dict[variable]['std']
            data = (data - mean) / std
            input_array.append(data)
        input_array = np.concatenate(input_array, 1)
        input_array = input_array[:self.split_index]
        input_array = input_array[-self.input_sequence_length:]

        if (self.enable_oversampling is True) and (self.phase == "train" is True) :
            factors = (0.8, 0.9, 1.0, 1.1, 1.2)
            input_shape = input_array.shape
            for i in range(input_shape[0]):
                for j in range(input_shape[1]) :
                    value = input_array[i, j]
                    new_value = value * factors[random.randint(0, 4)]
                    input_array[i, j] = new_value

        target_array = []
        for variable in self.target_variables :
            data = omni_data[variable][:, None]
            mean = self.stat_dict[variable]['mean']
            std = self.stat_dict[variable]['std']
            data = (data - mean) / std
            target_array.append(data)
        target_array = np.concatenate(target_array, 1)
        target_array = target_array[self.split_index:]
        target_array = target_array[:self.target_sequence_length]

        label_array = np.zeros((1, 1))
        label_array[0, 0] = file_class

        sdo_tensor = torch.tensor(sdo_array, dtype=torch.float32)
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        target_tensor = torch.tensor(target_array, dtype=torch.float32)
        label_tensor = torch.tensor(label_array, dtype=torch.float32)

        data_dict = {
                "sdo": sdo_tensor,
                "inputs": input_tensor,
                "targets": target_tensor,
                "labels": label_tensor,
                "file_names": os.path.basename(file_path)
        }
        
        if self.enable_memory_cache:
            self.memory_cache[file_name] = data_dict
        
        return data_dict


def create_dataloader(config, logger=None):
    dataset = CustomDataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=(config.experiment.phase == 'train'),
        num_workers=config.experiment.num_workers,
        pin_memory=(config.environment.device == 'cuda'),
        # pin_memory=False,
        drop_last=False  # Keep all samples
    )
    return dataloader


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    dataloader = create_dataloader(config)

    for _ in range(3):
        t0 = time.time()
        for batch in dataloader:
            sdo = batch['sdo'].numpy()
            inputs = batch['inputs'].numpy()
            targets = batch['targets'].numpy()
            labels = batch['labels'].numpy()
            # print(f"sdo: {sdo.shape}, {sdo.mean():.3f}, {sdo.std():.3f}")
            # print(f"inputs: {inputs.shape}, {inputs.mean():.3f}, {inputs.std():.3f}")
            # print(f"targets: {targets.shape}, {targets.mean():.3f}, {targets.std():.3f}")
            # print(f"labels: {labels.shape}, {labels.mean():.3f}, {labels.std():.3f}")
            # break
        print(time.time() - t0)

    print(f"pos_weight: {dataloader.dataset.pos_weight}")


if __name__ == "__main__" :
    main()