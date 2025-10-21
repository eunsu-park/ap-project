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


def read_h5(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Read data from HDF5 file.
    
    Args:
        file_path: Path to HDF5 file.
        input_variables: List of input variable names.
        target_variables: List of target variable names.
        
    Returns:
        Tuple of (sdo_193, sdo_211, omni_inputs, omni_targets).
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        KeyError: If required datasets are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        with h5py.File(file_path, 'r') as f:
            sdo = f['sdo'][:]  
            inputs = f['inputs'][:]
            targets = f['targets'][:]
            labels = f['labels'][()]
    except (OSError, h5py.error.HDF5Error) as e:
        raise OSError(f"Failed to read HDF5 file {file_path}: {e}")

    return sdo, inputs, targets, labels


def get_statistics(stat_file_path: str, dataset_path:str,
                   file_list: List[str], overwrite: bool = False) -> Dict[str, Dict[str, float]]:
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

    data_file_list = []
    for (file_name, file_class) in file_list :
        data_file_list.append(f"{dataset_path}/{file_class}/{file_name}")

    if not data_file_list:
        raise ValueError("No valid .h5 files found in data file list")
    
    stat_dict = {}
    
    if os.path.exists(stat_file_path) and not overwrite:
        # Load existing statistics
        try:
            stat_dict = pickle.load(open(stat_file_path, 'rb'))
        except (pickle.PickleError, KeyError) as e:
            print(f"Warning: Failed to load statistics from {stat_file_path}: {e}")
            print("Recomputing statistics...")
            overwrite = True
    
    if not os.path.exists(stat_file_path) or overwrite:
        # Compute statistics from scratch
        all_data = []
        for data_file_path in data_file_list:
            with h5py.File(data_file_path, 'r') as f:
                data = f['inputs'][:]
                all_data.append(data)
        
        # Compute final statistics
        all_data = np.concatenate(all_data, axis=1)  # Shape: (num_samples, num_variables)
        
        mean = np.mean(all_data, axis=1)
        std = np.std(all_data, axis=1)
        mean = np.expand_dims(mean, axis=0)
        std = np.expand_dims(std, axis=0)
        stat_dict = {
            'mean': mean,
            'std': std
        }
        
        # Save statistics
        try:
            os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
            pickle.dump(stat_dict, open(stat_file_path, 'wb'))
            print(f"Statistics saved to {stat_file_path}")
        except (OSError, pickle.PickleError) as e:
            print(f"Warning: Failed to save statistics to {stat_file_path}: {e}")
    
    return stat_dict


class CustomDataset(Dataset):
    def __init__(self, options, logger=None):
        self.data_root = options.environment.data_root
        self.dataset_name = options.data.dataset_name
        self.dataset_path = f"{self.data_root}/{self.dataset_name}"
        self.variables = self.input_variables = options.data.input_variables

        self.train_list_path = f"{self.dataset_path}/train_list.csv"
        self.validation_list_path = f"{self.dataset_path}/validation_list.csv"
        self.stat_file_path = f"{self.dataset_path}/statistics.pkl"

        train_df = pd.read_csv(self.train_list_path)
        validation_df = pd.read_csv(self.validation_list_path)

        train_file_names = train_df['filename'].tolist()
        train_file_class = train_df['class'].tolist()
        self.train_file_list = list(zip(train_file_names, train_file_class))

        self.validation_file_names = validation_df['filename'].tolist()
        self.validation_file_class = validation_df['class'].tolist()
        self.validation_file_list = list(zip(self.validation_file_names, self.validation_file_class))

        print(f"Training samples: {len(self.train_file_list)}, Validation samples: {len(self.validation_file_list)}")

        if options.experiment.phase == 'train':
            self.list_data = self.train_file_list
        elif options.experiment.phase == 'validation':
            self.list_data = self.validation_file_list
        else:
            raise ValueError(f"Unknown phase: {options.experiment.phase}. Must be 'train' or 'validation'.")
        
        self.nb_data = len(self.list_data)

        try:
            self.stat_dict = get_statistics(
                self.stat_file_path, self.dataset_path, 
                self.train_file_list
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load/compute statistics: {e}")
        print(f"Loaded statistics for {len(self.stat_dict)} variables.")

        self.input_sequence_length = options.data.input_sequence_length
        self.target_sequence_length = options.data.target_sequence_length

        self.memory_cache = {}
        self.cache_enabled = True  # Can be disabled for low-memory scenarios

    def __len__(self):
        return self.nb_data
    
    def __getitem__(self, idx):
        file_name, file_class = self.list_data[idx]
        if self.cache_enabled and file_name in self.memory_cache:
            return self.memory_cache[file_name]
        
        file_path = f"{self.dataset_path}/{file_class}/{file_name}"

        sdo, inputs, targets, labels = read_h5(file_path)
        sdo = sdo[:, 20 - self.input_sequence_length//2 : 20]
        sdo = sdo * (2./255.) - 1.
        inputs = inputs[:, 40-self.input_sequence_length : 40]
        inputs = np.transpose(inputs, (1, 0))  # Shape: (sequence_length, num_variables)
        targets = np.transpose(targets, (1, 0))  # Shape: (num_groups, num_vectors)
        labels = np.expand_dims(labels, axis=-1)  # Shape: (num_groups, 1)

        inputs = (inputs - self.stat_dict['mean']) / self.stat_dict['std']
        
        sdo_tensor = torch.tensor(sdo, dtype=torch.float32)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        if self.cache_enabled:
            self.memory_cache[file_name] = {
                "sdo": sdo_tensor,
                "inputs": inputs_tensor,
                "targets": targets_tensor,
                "labels": labels_tensor,
                "file_names": os.path.basename(file_path)
            }
        
        return {
            "sdo": sdo_tensor,
            "inputs": inputs_tensor,
            "targets": targets_tensor,
            "labels": labels_tensor,
            "file_names": os.path.basename(file_path)
        }


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
        sdo = batch['sdo'].numpy()
        inputs = batch['inputs'].numpy()
        targets = batch['targets'].numpy()
        labels = batch['labels'].numpy()
        print(f"sdo: {sdo.shape}, {sdo.mean():.3f}, {sdo.std():.3f}")
        print(f"inputs: {inputs.shape}, {inputs.mean():.3f}, {inputs.std():.3f}")
        print(f"targets: {targets.shape}, {targets.mean():.3f}, {targets.std():.3f}")
        print(f"labels: {labels.shape}, {labels.mean():.3f}, {labels.std():.3f}")
        break


if __name__ == "__main__" :
    main()