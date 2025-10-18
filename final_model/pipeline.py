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


class ReadH5:
    def __init__(self, config):
    
        self.sdo_wavelengths = config.data.sdo_wavelengths
        self.input_variables = config.data.input_variables
        self.target_variables = config.data.target_variables
        self.omni_variables = list(set(self.input_variables + self.target_variables))

    def __call__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
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

        except (OSError, h5py.error.HDF5Error) as e:
            raise OSError(f"Failed to read HDF5 file {file_path}: {e}")

        return sdo_data, omni_data


class CustomDataset:
    def __init__(self, config, logger=None):
        self.data_root = config.environment.data_root
        self.data_dir = f"{self.data_root}/original"
        self.dataset_name = config.data.dataset_name
        self.data_list_path = f"{self.data_root}/{self.dataset_name}.csv"
        df = pd.read_csv(self.data_list_path, encoding='utf-8')
        self.data_list = df["file_name"].tolist()
        self.data_num = len(self.data_list)

        self.sdo_wavelengths = sorted(config.data.sdo_wavelengths)
        self.sdo_sequence_length = config.data.sdo_sequence_length
        self.sdo_image_size = config.data.sdo_image_size

        self.input_variables = sorted(config.data.input_variables)
        self.input_sequence_length = config.data.input_sequence_length

        self.target_variables = sorted(config.data.target_variables)
        self.target_sequence_length = config.data.target_sequence_length

        self.cache_enable = config.experiment.cache_enable

        self.memory_cache = {}

        self.reader = ReadH5(config)

    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        if file_name in self.memory_cache :
            return self.memory_cache[file_name]

        file_path = f"{self.data_dir}/{file_name}"

        dict_sdo_data, dict_omni_data = self.reader(file_path)

        sdo_data = []
        for wavelength in self.sdo_wavelengths:
            data = dict_sdo_data[wavelength]
            sdo_data.append(data)
        sdo_data = np.concatenate(sdo_data, axis=1)  # Shape: (frames, C, H, W) 
        sdo_data = np.transpose(sdo_data, (1, 0, 2, 3))  # Shape: (C, frames, H, W)

        input_data = []
        for variable in self.input_variables:
            data = dict_omni_data[variable]
            data = np.expand_dims(data, axis=-1)
            input_data.append(data)
        input_data = np.concatenate(input_data, axis=-1) # Shape: (65, num_input_variables) 
        input_data = input_data[:self.input_sequence_length] # Shape: (input_sequence_length, num_input_variables) 

        target_data = []
        for variable in self.target_variables:
            data = dict_omni_data[variable]
            data = np.expand_dims(data, axis=-1)
            target_data.append(data)
        target_data = np.concatenate(target_data, axis=-1) # Shape: (65, num_target_variables) 
        target_data = target_data[self.input_sequence_length:self.input_sequence_length + self.target_sequence_length] # Shape: (target_sequence_length, num_target_variables) 

        sdo_tensor = torch.tensor(sdo_data, dtype=torch.float32)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        data_dict = {
            "sdo": sdo_tensor,
            "input": input_tensor,
            "target": target_tensor,
            "file_name": file_name    
        }

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
        drop_last=False  # Keep all samples
    )
    return dataloader


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    dataloader = create_dataloader(config)

    for batch in dataloader:
        print(batch['sdo'].shape, batch['input'].shape, batch['target'].shape)
        break

if __name__ == "__main__" :
    main()        






# def get_statistics(stat_file_path: str, dataset_path:str,
#                    file_list: List[str], overwrite: bool = False) -> Dict[str, Dict[str, float]]:
#     """Compute and cache statistics for data normalization.
    
#     Args:
#         stat_file_path: Path to save/load statistics pickle file.
#         data_root: Root directory containing data files.
#         data_file_list: List of data file names.
#         variables: List of variable names to compute statistics for.
#         overwrite: Whether to recompute statistics even if cache exists.
        
#     Returns:
#         Dictionary containing mean and std for each variable.
        
#     Raises:
#         FileNotFoundError: If data files cannot be found.
#         ValueError: If no valid data is found for statistics computation.
#     """
#     # Filter for h5 files only

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


# class CustomDataset(Dataset):
#     def __init__(self, options, logger=None):
#         self.data_root = options.environment.data_root
#         self.dataset_name = options.data.dataset_name
#         self.dataset_path = f"{self.data_root}/{self.dataset_name}"
#         self.variables = self.input_variables = options.data.input_variables

#         self.train_list_path = f"{self.dataset_path}/train_list.csv"
#         self.validation_list_path = f"{self.dataset_path}/validation_list.csv"
#         self.stat_file_path = f"{self.dataset_path}/statistics.pkl"

#         train_df = pd.read_csv(self.train_list_path)
#         validation_df = pd.read_csv(self.validation_list_path)

#         train_file_names = train_df['filename'].tolist()
#         train_file_class = train_df['class'].tolist()
#         self.train_file_list = list(zip(train_file_names, train_file_class))

#         self.validation_file_names = validation_df['filename'].tolist()
#         self.validation_file_class = validation_df['class'].tolist()
#         self.validation_file_list = list(zip(self.validation_file_names, self.validation_file_class))

#         print(f"Training samples: {len(self.train_file_list)}, Validation samples: {len(self.validation_file_list)}")

#         if options.experiment.phase == 'train':
#             self.list_data = self.train_file_list
#         elif options.experiment.phase == 'validation':
#             self.list_data = self.validation_file_list
#         else:
#             raise ValueError(f"Unknown phase: {options.experiment.phase}. Must be 'train' or 'validation'.")
        
#         self.nb_data = len(self.list_data)

#         try:
#             self.stat_dict = get_statistics(
#                 self.stat_file_path, self.dataset_path, 
#                 self.train_file_list
#             )
#         except Exception as e:
#             raise RuntimeError(f"Failed to load/compute statistics: {e}")
#         print(f"Loaded statistics for {len(self.stat_dict)} variables.")

#         self.memory_cache = {}
#         self.cache_enabled = True  # Can be disabled for low-memory scenarios

#     def __len__(self):
#         return self.nb_data
    
#     def __getitem__(self, idx):
#         file_name, file_class = self.list_data[idx]
#         if self.cache_enabled and file_name in self.memory_cache:
#             return self.memory_cache[file_name]
        
#         file_path = f"{self.dataset_path}/{file_class}/{file_name}"

#         sdo, inputs, targets, labels = read_h5(file_path)
#         inputs = np.transpose(inputs, (1, 0))  # Shape: (sequence_length, num_variables)
#         targets = np.transpose(targets, (1, 0))  # Shape: (num_groups, num_vectors)
#         labels = np.expand_dims(labels, axis=-1)  # Shape: (num_groups, 1)

#         inputs = (inputs - self.stat_dict['mean']) / self.stat_dict['std']
        
#         sdo_tensor = torch.tensor(sdo, dtype=torch.float32)
#         inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
#         targets_tensor = torch.tensor(targets, dtype=torch.float32)
#         labels_tensor = torch.tensor(labels, dtype=torch.float32)

#         if self.cache_enabled:
#             self.memory_cache[file_name] = {
#                 "sdo": sdo_tensor,
#                 "inputs": inputs_tensor,
#                 "targets": targets_tensor,
#                 "labels": labels_tensor,
#                 "file_names": os.path.basename(file_path)
#             }
        
#         return {
#             "sdo": sdo_tensor,
#             "inputs": inputs_tensor,
#             "targets": targets_tensor,
#             "labels": labels_tensor,
#             "file_names": os.path.basename(file_path)
#         }


# def create_dataloader(config, logger=None):
#     dataset = CustomDataset(config)
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=config.experiment.batch_size,
#         shuffle=(config.experiment.phase == 'train'),
#         num_workers=config.experiment.num_workers,
#         pin_memory=(config.environment.device == 'cuda'),
#         drop_last=False  # Keep all samples
#     )
#     return dataloader


# @hydra.main(config_path="./configs", version_base=None)
# def main(config):

#     dataloader = create_dataloader(config)

#     for batch in dataloader:
#         print(batch['sdo'].shape, batch['inputs'].shape, batch['targets'].shape, batch['labels'].shape, batch['file_names'])
#         tmp = batch['inputs'].numpy()
#         print(tmp.mean(), tmp.std())
#         break


# if __name__ == "__main__" :
#     main()