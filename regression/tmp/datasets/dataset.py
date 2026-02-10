"""
Multimodal dataset for solar wind prediction.

Consolidated module containing:
- HDF5Reader: File I/O
- DataProcessor: Preprocessing and normalization
- MultimodalDataset: PyTorch Dataset
"""

import os
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py

from .statistics import Normalizer, compute_statistics
from .sampling import SamplingStrategy


# ============================================================================
# File I/O
# ============================================================================

class HDF5Reader:
    """Handle HDF5 file reading."""
    
    @staticmethod
    def read(
        file_path: str,
        sdo_wavelengths: List[str],
        omni_variables: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Read SDO and OMNI data from HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            sdo_wavelengths: SDO wavelengths to read
            omni_variables: OMNI variables to read
            
        Returns:
            (sdo_data, omni_data) dictionaries
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sdo_data = {}
        omni_data = {}
        
        with h5py.File(file_path, 'r') as f:
            # Read SDO data
            for wavelength in sdo_wavelengths:
                dataset_name = f"sdo/{wavelength}"
                if dataset_name not in f:
                    raise KeyError(f"SDO wavelength {wavelength} not found")
                sdo_data[wavelength] = f[dataset_name][:]
            
            # Read OMNI data
            for variable in omni_variables:
                dataset_name = f"omni/{variable}"
                if dataset_name not in f:
                    raise KeyError(f"OMNI variable {variable} not found")
                omni_data[variable] = f[dataset_name][:]
        
        return sdo_data, omni_data


# ============================================================================
# Data Preprocessing
# ============================================================================

class DataProcessor:
    """Handle preprocessing of SDO and OMNI data."""
    
    def __init__(
        self,
        normalizer: Normalizer,
        sdo_wavelengths: List[str],
        sdo_sequence_length: int,
        input_variables: List[str],
        target_variables: List[str],
        input_sequence_length: int,
        target_sequence_length: int,
        split_index: int
    ):
        self.normalizer = normalizer
        self.sdo_wavelengths = sdo_wavelengths
        self.sdo_sequence_length = sdo_sequence_length
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.split_index = split_index
    
    def process_sdo(self, sdo_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Process SDO imagery: normalize, stack, select timesteps.
        
        Returns shape: (C, T, H, W)
        """
        sdo_arrays = []
        
        for wavelength in self.sdo_wavelengths:
            data = sdo_data[wavelength]
            # Normalize [0, 255] -> [-1, 1]
            data = self.normalizer.normalize_sdo(data)
            sdo_arrays.append(data)
        
        # Stack: (T, C, H, W)
        sdo_array = np.concatenate(sdo_arrays, axis=1)
        
        # Select last N timesteps
        sdo_array = sdo_array[-self.sdo_sequence_length:]
        
        # Transpose to (C, T, H, W)
        sdo_array = np.transpose(sdo_array, (1, 0, 2, 3))
        
        return sdo_array
    
    def process_omni_input(self, omni_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Process OMNI input (before split_index).
        
        Returns shape: (T, C)
        """
        omni_arrays = []
        
        for variable in self.input_variables:
            data = omni_data[variable]
            # Z-score normalization
            data = self.normalizer.normalize_omni(data, variable)
            omni_arrays.append(data)
        
        # Stack: (T, C)
        omni_array = np.stack(omni_arrays, axis=-1)
        
        # Select data before split_index
        omni_array = omni_array[:self.split_index]
        
        # Select last N timesteps
        omni_array = omni_array[-self.input_sequence_length:]
        
        return omni_array
    
    def process_omni_target(self, omni_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Process OMNI target (after split_index).
        
        Returns shape: (T, C)
        """
        target_arrays = []
        
        for variable in self.target_variables:
            data = omni_data[variable]
            # Z-score normalization
            data = self.normalizer.normalize_omni(data, variable)
            target_arrays.append(data)
        
        # Stack: (T, C)
        target_array = np.stack(target_arrays, axis=-1)
        
        # Select data after split_index
        target_array = target_array[self.split_index:]
        
        # Select first N timesteps
        target_array = target_array[:self.target_sequence_length]
        
        return target_array
    
    def apply_augmentation(self, data: np.ndarray, sample_idx: int, 
                          file_name: str) -> np.ndarray:
        """
        Apply deterministic data augmentation.
        
        Uses hash of (file_name + sample_idx) for reproducibility.
        """
        # Create deterministic seed from file_name and index
        seed = hash(f"{file_name}_{sample_idx}") % (2**32)
        rng = np.random.RandomState(seed)
        
        # Add small Gaussian noise (5% of std)
        noise_std = 0.05
        noise = rng.normal(0, noise_std, data.shape)
        data = data + noise
        
        return data


# ============================================================================
# PyTorch Dataset
# ============================================================================

class MultimodalDataset(Dataset):
    """
    Multimodal dataset for solar wind prediction.
    
    Handles both training and validation phases with optional
    undersampling and oversampling.
    """
    
    def __init__(self, config, statistics: Dict[str, Dict[str, float]]):
        """
        Args:
            config: Config object with data settings
            statistics: Precomputed statistics for normalization
        """
        self.config = config
        
        # Paths
        self.data_root = config.environment.data_root
        self.dataset_dir_name = config.data.dataset_dir_name
        
        # Load file lists
        train_list_path = f"{self.data_root}/{config.data.dataset_name}_train.csv"
        val_list_path = f"{self.data_root}/{config.data.dataset_name}_validation.csv"
        
        self.train_file_list = self._load_file_list(train_list_path, config.data.target_day)
        self.validation_file_list = self._load_file_list(val_list_path, config.data.target_day)
        
        # Calculate pos_weight (for loss function)
        self.pos_weight = SamplingStrategy.get_pos_weight(self.train_file_list)
        
        # Apply sampling strategies
        self.indices = None
        
        if config.experiment.enable_undersampling:
            self.train_file_list, _, _ = SamplingStrategy.undersample(
                self.train_file_list,
                num_folds=config.experiment.num_subsample,
                fold_index=config.experiment.subsample_index
            )
        
        if config.experiment.enable_oversampling:
            self.indices, self.train_file_list, _, _ = SamplingStrategy.oversample(
                self.train_file_list,
                oversample_factor=config.experiment.num_oversample
            )
        
        # Select file list based on phase
        if config.experiment.phase == 'train':
            self.file_list = self.train_file_list
        else:
            self.file_list = self.validation_file_list
        
        # Initialize normalizer and processor
        self.normalizer = Normalizer(statistics)
        self.processor = DataProcessor(
            normalizer=self.normalizer,
            sdo_wavelengths=config.data.sdo_wavelengths,
            sdo_sequence_length=config.data.sdo_sequence_length,
            input_variables=config.data.input_variables,
            target_variables=config.data.target_variables,
            input_sequence_length=config.data.input_sequence_length,
            target_sequence_length=config.data.target_sequence_length,
            split_index=config.data.split_index
        )
        
        # Expose statistics for plotting
        self.stat_dict = statistics
        
        print(f"Dataset: {len(self)} samples in {config.experiment.phase} phase")
    
    def _load_file_list(self, csv_path: str, target_day: int) -> List[Tuple[str, int]]:
        """Load file list from CSV."""
        df = pd.read_csv(csv_path)
        file_names = df['file_name'].tolist()
        labels = df[f'class_day{target_day}'].tolist()
        return list(zip(file_names, labels))
    
    def __len__(self) -> int:
        """Return dataset size."""
        if self.indices is not None:
            return len(self.indices)
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'sdo', 'inputs', 'targets', 'labels', 'file_names'
        """
        # Map index if oversampling
        if self.indices is not None:
            real_idx = self.indices[idx]
        else:
            real_idx = idx
        
        # Get file info
        file_name, label = self.file_list[real_idx]
        file_path = f"{self.data_root}/{self.dataset_dir_name}/{file_name}"
        
        # Read data
        sdo_data, omni_data = HDF5Reader.read(
            file_path,
            self.config.data.sdo_wavelengths,
            self.config.data.omni_variables
        )
        
        # Process data
        sdo_array = self.processor.process_sdo(sdo_data)
        input_array = self.processor.process_omni_input(omni_data)
        target_array = self.processor.process_omni_target(omni_data)
        
        # Apply augmentation if training
        if self.config.experiment.enable_oversampling and self.config.experiment.phase == 'train':
            input_array = self.processor.apply_augmentation(input_array, idx, file_name)
        
        # Create label array
        label_array = np.array([[label]], dtype=np.float32)
        
        # Convert to tensors
        return {
            'sdo': torch.tensor(sdo_array, dtype=torch.float32),
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': os.path.basename(file_path)
        }


# ============================================================================
# DataLoader Factory
# ============================================================================

def create_dataloader(config) -> DataLoader:
    """
    Create DataLoader with statistics computation.
    
    Args:
        config: Config object
        
    Returns:
        Configured DataLoader
    """
    # Compute or load statistics
    train_df = pd.read_csv(f"{config.environment.data_root}/{config.data.dataset_name}_train.csv")
    train_file_names = train_df['file_name'].tolist()
    
    stat_file_path = f"{config.environment.data_root}/{config.data.dataset_name}_statistics.pkl"
    
    statistics = compute_statistics(
        data_root=config.environment.data_root,
        data_file_list=train_file_names,
        variables=config.data.omni_variables,
        stat_file_path=stat_file_path,
        dataset_dir_name=config.data.dataset_dir_name,
        overwrite=False
    )
    
    # Create dataset
    dataset = MultimodalDataset(config, statistics)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=(config.experiment.phase == 'train'),
        num_workers=config.experiment.num_workers,
        pin_memory=(config.environment.device == 'cuda'),
        drop_last=False,
        persistent_workers=(config.experiment.num_workers > 0),
        prefetch_factor=2 if config.experiment.num_workers > 0 else None,
    )
    
    return dataloader
