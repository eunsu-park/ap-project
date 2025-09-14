import os
import sys
import time
import pickle
import random
from glob import glob
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd


def get_statistics(stat_file_path, data_root, data_file_list, variables, overwrite=False):
    data_file_list = [f"{data_root}/{f}" for f in data_file_list if f.endswith('.h5')]
    stat_dict = {}
    if os.path.exists(stat_file_path) and not overwrite :
        loaded = pickle.load(open(stat_file_path, 'rb'))
        for variable in variables:
            stat_dict[variable] = loaded.get(variable, {})
    else:
        total_dict = {}
        for variable in variables:
            total_dict[variable] = []
            stat_dict[variable] = {}
        for data_file_path in data_file_list:
            with h5py.File(data_file_path, 'r') as f:
                for variable in variables:
                    data = f[f"omni_{variable}"][:]
                    total_dict[variable].append(data)
        for variable in variables:
            total_dict[variable] = np.concatenate(total_dict[variable], axis=0)
            stat_dict[variable]['mean'] = np.nanmean(total_dict[variable])
            stat_dict[variable]['std'] = np.nanstd(total_dict[variable])
        pickle.dump(stat_dict, open(stat_file_path, 'wb'))
        print(f"Statistics saved to {stat_file_path}")
    return stat_dict


def read_h5(file_path, input_variables, target_variables):
    with h5py.File(file_path, 'r') as f:
        sdo_193 = f['sdo_193'][:]
        sdo_211 = f['sdo_211'][:]
        omni_inputs = {variable: f[f"omni_{variable}"][:] for variable in input_variables}
        omni_targets = {variable: f[f"omni_{variable}"][:] for variable in target_variables}
    return sdo_193, sdo_211, omni_inputs, omni_targets


class CustomDataset(Dataset):
    def __init__(self, options):
        self.data_root = options.data_root
        self.train_list_path = options.train_list_path
        self.test_list_path = options.test_list_path
        
        self.train_df = pd.read_csv(self.train_list_path)[:1000]
        self.test_df = pd.read_csv(self.test_list_path)[:1000]

        self.train_file_list = self.train_df['file_name'].tolist()
        self.test_file_list = self.test_df['file_name'].tolist()

        if options.phase == 'train':
            self.list_data = self.train_file_list
        elif options.phase == 'test':
            self.list_data = self.test_file_list
        else:
            raise ValueError(f"Unknown phase: {options.phase}")
        self.nb_data = len(self.list_data)
        print(f"Using {self.nb_data} samples for {options.phase} phase.")

        self.input_variables = options.input_variables
        self.input_sequence_length = options.input_sequence_length
        self.num_input_variables = options.num_input_variables

        self.target_variables = options.target_variables
        self.target_sequence_length = options.target_sequence_length
        self.num_target_variables = options.num_target_variables

        self.inception_in_channels = options.inception_in_channels
        self.inception_in_image_frames = options.inception_in_image_frames
        self.inception_in_image_size = options.inception_in_image_size

        self.variables = list(set(self.input_variables + self.target_variables))
        self.sdo_shape = (
            self.inception_in_channels,
            self.inception_in_image_frames,
            self.inception_in_image_size,
            self.inception_in_image_size
        )

        self.inputs_shape = (
            self.input_sequence_length,
            self.num_input_variables
        )

        self.targets_shape = (
            self.target_sequence_length,
            self.num_target_variables
        )

        self.stat_file_path = options.stat_file_path
        self.stat_dict = get_statistics(self.stat_file_path, self.data_root, self.train_file_list, self.variables)

        self.memory_cache = {}

    def cache_item(self, file_name, data_dict):
        """Cache processed data item.
        
        Args:
            file_name: Name of the file.
            data_dict: Processed data dictionary to cache.
        """
        self.memory_cache[file_name] = data_dict

    def __len__(self):
        return self.nb_data

    def __getitem__(self, idx):
        file_name = self.list_data[idx]
        if file_name in self.memory_cache:
            return self.memory_cache[file_name]

        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            # Update file_name for current attempt
            if attempts > 0:
                idx = random.randint(0, self.nb_data - 1)
                file_name = self.list_data[idx]
            
            # Check cache again for new random file
            if file_name in self.memory_cache:
                return self.memory_cache[file_name]
            
            # File reading with exception handling
            try:
                file_path = f"{self.data_root}/{file_name}"
                sdo_193, sdo_211, omni_inputs, omni_targets = read_h5(file_path, self.input_variables, self.target_variables)
            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    return self._create_dummy_data()
                    # raise RuntimeError(f"Failed to read files after {max_attempts} attempts. Last error: {e}")
                continue

            # Process SDO data efficiently
            sdo = np.concatenate([sdo_193, sdo_211], axis=1)  # Shape: (20, 2, 64, 64)
            del sdo_193, sdo_211  # Free memory immediately
            sdo = np.transpose(sdo, (1, 0, 2, 3))  # Shape: (2, 20, 64, 64)

            # Process time series data efficiently
            inputs = np.stack([
                (omni_inputs[k][:self.input_sequence_length] - self.stat_dict[k]['mean']) / self.stat_dict[k]['std']
                for k in self.input_variables
            ], axis=-1)
            
            targets = np.stack([
                (omni_targets[k][self.input_sequence_length:self.input_sequence_length + self.target_sequence_length] - 
                 self.stat_dict[k]['mean']) / self.stat_dict[k]['std']
                for k in self.target_variables
            ], axis=-1)
            del omni_inputs, omni_targets  # Free memory immediately
            
            # Normalize image data from [0, 255] to [-1, 1]
            sdo = (sdo / 255.0) * 2.0 - 1.0

            # Data validation (normal flow, not exceptions)
            if self._has_nan(sdo, inputs, targets):
                attempts += 1
                continue

            if self._has_invalid_shape(sdo, inputs, targets):
                attempts += 1
                continue

            # Convert to tensors and cache
            data_dict = {
                "sdo": torch.tensor(sdo, dtype=torch.float32),
                "inputs": torch.tensor(inputs, dtype=torch.float32),
                "targets": torch.tensor(targets, dtype=torch.float32),
                "file_names": file_name
            }
            del sdo, inputs, targets  # Free numpy arrays

            self.cache_item(file_name, data_dict)
            return data_dict
        
        raise RuntimeError(f"Failed to load valid data after {max_attempts} attempts")

    def _has_nan(self, *tensors):
        """Check if any tensor contains NaN values.
        
        Args:
            *tensors: Variable number of numpy arrays or torch tensors to check.
            
        Returns:
            bool: True if NaN found, False if no NaN.
        """
        for tensor in tensors:
            if isinstance(tensor, np.ndarray):
                if np.isnan(tensor).any():
                    return True
            elif isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any():
                    return True
        return False

    def _create_dummy_data(self):
        """Create dummy data using statistical means when all attempts fail.
        
        Returns:
            dict: Dictionary containing dummy tensors with mean values.
        """
        # Create dummy SDO data (normalized image data should be around 0)
        dummy_sdo = torch.zeros(self.sdo_shape, dtype=torch.float32)
        
        # Create dummy SW inputs using statistical means (already normalized to ~0)
        dummy_inputs = torch.zeros(self.inputs_shape, dtype=torch.float32)
        
        # Create dummy SW targets using statistical means (already normalized to ~0)
        dummy_targets = torch.zeros(self.targets_shape, dtype=torch.float32)
        
        return {
            "sdo": dummy_sdo,
            "inputs": dummy_inputs,
            "targets": dummy_targets
        }

    def _has_invalid_shape(self, sdo, inputs, targets):
        """Check if tensors have invalid shapes.
        
        Args:
            sdo: SDO image data array.
            inputs: Solar wind input data array.
            targets: Solar wind target data array.
            
        Returns:
            bool: True if shapes are invalid, False if shapes are correct.
        """
        if sdo.shape != self.sdo_shape:
            return True
        if inputs.shape != self.inputs_shape:
            return True
        if targets.shape != self.targets_shape:
            return True
        return False


def create_dataloader(options):
    dataset = CustomDataset(options)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=options.batch_size,
                                             shuffle=(options.phase == 'train'),
                                             num_workers=options.num_workers,
                                             pin_memory=(options.device == 'cuda'))
    return dataloader
