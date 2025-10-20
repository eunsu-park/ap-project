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


class Normalize:
    def __init__(self, config):
        self.stat_dict = pickle.load(open(
            f"{config.environment.data_root}/{config.data.dataset_name}.pkl", 'rb'
        ))

    def __call__(self, data_arr, data_key):
        mean = self.stat_dict[data_key]['mean']
        std = self.stat_dict[data_key]['std']
        return (data_arr - mean) / std


class CustomDataset:
    def __init__(self, config, logger=None):
        self.data_root = config.environment.data_root
        self.data_dir = f"{self.data_root}/original"
        self.dataset_name = config.data.dataset_name

        df = pd.read_csv(
            f"{self.data_root}/{self.dataset_name}_train.csv"
        )
        self.train_list = df["file_name"].tolist()

        df = pd.read_csv(
            f"{self.data_root}/{self.dataset_name}_validation.csv"
        )
        self.validation_list = df["file_name"].tolist()

        if config.experiment.phase == "train" :
            self.data_list = self.train_list
        elif config.experiment.phase == "validation" :
            self.data_list = self.validation_list
        else :
            raise ValueError(f"Unknown phase: {config.experiment.phase}. Must be 'train' or 'validation'.")
        self.data_num = len(self.data_list)

        self.stat_dict = pickle.load(open(
            f"{self.data_root}/{self.dataset_name}.pkl", 'rb'
        ))        

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
        self.normalizer = Normalize(config)

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
            data = data * (2./255.) - 1.
            sdo_data.append(data)
        sdo_data = np.concatenate(sdo_data, axis=1)  # Shape: (frames, C, H, W) 
        sdo_data = np.transpose(sdo_data, (1, 0, 2, 3))  # Shape: (C, frames, H, W)

        input_data = []
        for variable in self.input_variables:
            data = dict_omni_data[variable]
            data = self.normalizer(data, variable)
            data = np.expand_dims(data, axis=-1)
            input_data.append(data)
        input_data = np.concatenate(input_data, axis=-1) # Shape: (65, num_input_variables) 
        input_data = input_data[:self.input_sequence_length] # Shape: (input_sequence_length, num_input_variables) 

        target_data = []
        for variable in self.target_variables:
            data = dict_omni_data[variable]
            data = self.normalizer(data, variable)
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
    try :
        dataloader = create_dataloader(config)
        for i, batch in enumerate(dataloader):
            sdo = batch['sdo'].numpy()
            input = batch['input'].numpy()
            target = batch['target'].numpy()

            print(f"sdo: {sdo.shape}, {sdo.mean():.3f} +- {sdo.std():.3f}")
            print(f"input: {input.shape}, {input.mean():.3f} +- {input.std():.3f}")
            print(f"target: {target.shape}, {target.mean():.3f} +- {target.std():.3f}")
            print("")
            break

        ## Cache Test
        config.experiment.cache_enable = True
        config.experiment.num_workers = 0
        dataloader = create_dataloader(config)
        for n in range(3):
            for i, batch in enumerate(dataloader):
                if i == 0 :
                    t0 = time.time()
                sdo = batch['sdo'].numpy()
                input = batch['input'].numpy()
            print(time.time() - t0)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Failed: {e}")
        return 1
    return 0


if __name__ == "__main__" :
    exit(main())