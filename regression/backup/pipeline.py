"""
Refactored multimodal dataset pipeline for solar wind prediction.

Major improvements:
- Removed memory cache (simpler, more reproducible)
- Unified sampling strategy (index-based, no file name manipulation)
- Modular preprocessing pipeline
- Online statistics computation (memory efficient)
- Enhanced reproducibility (sorting, explicit seeding)
- Cleaner separation of concerns
"""

import os
import sys
import time
import pickle
import hashlib
from glob import glob
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
import hydra


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Data configuration container for type safety and clarity."""
    
    # Paths
    data_root: str
    dataset_name: str
    
    # SDO configuration
    sdo_wavelengths: List[str]
    sdo_sequence_length: int
    sdo_image_size: int
    
    # OMNI configuration
    input_variables: List[str]
    target_variables: List[str]
    input_sequence_length: int
    target_sequence_length: int
    split_index: int
    
    # Training configuration
    target_day: int
    phase: str  # 'train' or 'validation'
    
    # Sampling configuration
    enable_undersampling: bool = False
    enable_oversampling: bool = False
    num_subsample: int = 1
    subsample_index: int = 0
    num_oversample: int = 1
    
    @property
    def dataset_path(self):
        return f"{self.data_root}/{self.dataset_name}"
    
    @property
    def train_list_path(self):
        return f"{self.dataset_path}_train.csv"
    
    @property
    def validation_list_path(self):
        return f"{self.dataset_path}_validation.csv"
    
    @property
    def stat_file_path(self):
        return f"{self.dataset_path}_statistics.pkl"
    
    @property
    def omni_variables(self):
        return list(set(self.input_variables + self.target_variables))


# ============================================================================
# Normalization
# ============================================================================

class Normalizer:
    """Handles data normalization for SDO and OMNI data."""
    
    def __init__(self, stats: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Args:
            stats: Dictionary of statistics {variable: {'mean': x, 'std': y}}
        """
        self.stats = stats or {}
    
    def normalize_sdo(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize SDO data from [0, 255] to [-1, 1].
        
        Args:
            data: Raw SDO data in range [0, 255]
            
        Returns:
            Normalized data in range [-1, 1]
        """
        return (data * (2.0 / 255.0)) - 1.0
    
    def normalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        """
        Z-score normalization for OMNI data.
        
        Args:
            data: Raw OMNI data
            variable: Variable name for statistics lookup
            
        Returns:
            Normalized data (z-score)
        """
        if variable not in self.stats:
            raise KeyError(f"Statistics not found for variable: {variable}")
        
        mean = self.stats[variable]['mean']
        std = self.stats[variable]['std']
        
        return (data - mean) / std
    
    def denormalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Reverse z-score normalization."""
        if variable not in self.stats:
            raise KeyError(f"Statistics not found for variable: {variable}")
        
        mean = self.stats[variable]['mean']
        std = self.stats[variable]['std']
        
        return data * std + mean


# ============================================================================
# Statistics Computation
# ============================================================================

class OnlineStatistics:
    """
    Compute mean and std using Welford's online algorithm.
    Memory efficient - O(1) space complexity.
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def update(self, batch: np.ndarray):
        """
        Update statistics with a new batch of data.
        
        Args:
            batch: Data array of any shape
        """
        # Flatten and filter finite values
        values = batch.flatten()
        valid_values = values[np.isfinite(values)]
        
        for x in valid_values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
    
    @property
    def std(self) -> float:
        """Return standard deviation."""
        if self.n < 2:
            return 1.0  # Fallback for insufficient data
        return float(np.sqrt(self.M2 / self.n))
    
    def get_stats(self) -> Dict[str, float]:
        """Return statistics as dictionary."""
        return {
            'mean': float(self.mean),
            'std': self.std
        }


def compute_statistics(
    data_root: str,
    data_file_list: List[str],
    variables: List[str],
    stat_file_path: str,
    overwrite: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Compute and cache statistics for OMNI variables.
    
    Args:
        data_root: Root directory containing data files
        data_file_list: List of data file names
        variables: List of OMNI variable names
        stat_file_path: Path to save/load statistics pickle file
        overwrite: Whether to recompute even if cache exists
        
    Returns:
        Dictionary of statistics {variable: {'mean': x, 'std': y}}
    """
    # Load existing statistics if available
    if os.path.exists(stat_file_path) and not overwrite:
        try:
            with open(stat_file_path, 'rb') as f:
                loaded_stats = pickle.load(f)
            
            # Verify all variables are present
            if all(var in loaded_stats for var in variables):
                print(f"Loaded statistics from {stat_file_path}")
                return {var: loaded_stats[var] for var in variables}
            else:
                print("Incomplete statistics, recomputing...")
        except (pickle.PickleError, KeyError) as e:
            print(f"Failed to load statistics: {e}, recomputing...")
    
    # Filter for h5 files only
    h5_files = [f"{data_root}/oversampling/{f}" for f in data_file_list if f.endswith('.h5')]
    
    if not h5_files:
        raise ValueError("No valid .h5 files found in data file list")
    
    # Initialize online statistics for each variable
    stats_computers = {var: OnlineStatistics() for var in variables}
    
    # Process files
    valid_files = 0
    for file_path in h5_files:
        if not os.path.exists(file_path):
            continue
        
        try:
            with h5py.File(file_path, 'r') as f:
                for variable in variables:
                    dataset_name = f"omni_{variable}"
                    if dataset_name in f:
                        data = f[dataset_name][:]
                        stats_computers[variable].update(data)
            
            valid_files += 1
            
            if valid_files % 100 == 0:
                print(f"Processed {valid_files}/{len(h5_files)} files...")
                
        except (OSError, KeyError) as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            continue
    
    if valid_files == 0:
        raise ValueError("No valid data files found for statistics computation")
    
    # Compile final statistics
    stat_dict = {}
    for variable in variables:
        stat_dict[variable] = stats_computers[variable].get_stats()
        print(f"{variable}: mean={stat_dict[variable]['mean']:.3f}, "
              f"std={stat_dict[variable]['std']:.3f}")
    
    # Save statistics
    try:
        os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
        with open(stat_file_path, 'wb') as f:
            pickle.dump(stat_dict, f)
        print(f"Statistics saved to {stat_file_path}")
    except (OSError, pickle.PickleError) as e:
        print(f"Warning: Failed to save statistics: {e}")
    
    return stat_dict


# ============================================================================
# Sampling Strategy
# ============================================================================

class SamplingStrategy:
    """
    Handle dataset sampling strategies (undersampling, oversampling).
    Uses index-based approach instead of file name manipulation.
    """
    
    @staticmethod
    def split_by_class(file_list: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Split file list into positive and negative samples.
        
        Returns:
            (positive_samples, negative_samples)
        """
        positive = []
        negative = []
        
        for file_name, label in file_list:
            if label == 0:
                negative.append((file_name, label))
            else:
                positive.append((file_name, label))
        
        # Sort for reproducibility (independent of input order)
        positive.sort(key=lambda x: x[0])
        negative.sort(key=lambda x: x[0])
        
        return positive, negative
    
    @staticmethod
    def undersample(
        file_list: List[Tuple[str, int]],
        num_folds: int,
        fold_index: int
    ) -> Tuple[List[Tuple[str, int]], int, int]:
        """
        Undersample negative class by splitting into folds.
        
        Args:
            file_list: List of (file_name, label) tuples
            num_folds: Number of folds to split negative samples
            fold_index: Which fold to use (0 to num_folds-1)
            
        Returns:
            (sampled_file_list, num_positive, num_negative_total)
            
        Note:
            Uses global random state (should be set by train.py)
        """
        positive, negative = SamplingStrategy.split_by_class(file_list)
        
        # Shuffle negative samples using global random state
        import random
        random.shuffle(negative)
        
        # Split into folds
        n = len(negative)
        base_size = n // num_folds
        remainder = n % num_folds
        
        start = 0
        for i in range(num_folds):
            size = base_size + (1 if i < remainder else 0)
            if i == fold_index:
                selected_negative = negative[start:start + size]
                break
            start += size
        
        # Combine positive and selected negative
        sampled_list = positive + selected_negative
        
        return sampled_list, len(positive), len(negative)
    
    @staticmethod
    def oversample(
        file_list: List[Tuple[str, int]],
        oversample_factor: int
    ) -> Tuple[List[int], List[Tuple[str, int]], int, int]:
        """
        Oversample positive class by creating index mapping.
        
        Args:
            file_list: List of (file_name, label) tuples
            oversample_factor: How many times to replicate positive samples
            
        Returns:
            (indices, file_list, num_positive, num_negative)
            indices: List of indices into file_list (with repetitions)
        """
        positive, negative = SamplingStrategy.split_by_class(file_list)
        
        # Build index mapping
        indices = []
        
        # Add negative indices (once each)
        for i, (file_name, label) in enumerate(file_list):
            if label == 0:
                indices.append(i)
        
        # Add positive indices (multiple times)
        for i, (file_name, label) in enumerate(file_list):
            if label == 1:
                for _ in range(oversample_factor):
                    indices.append(i)
        
        return indices, file_list, len(positive), len(negative)
    
    @staticmethod
    def get_pos_weight(file_list: List[Tuple[str, int]]) -> float:
        """
        Calculate positive class weight for BCEWithLogitsLoss.
        
        Returns:
            Weight = num_negative / num_positive
        """
        positive, negative = SamplingStrategy.split_by_class(file_list)
        
        if len(positive) == 0:
            raise ValueError("No positive samples found")
        
        return len(negative) / len(positive)


# ============================================================================
# HDF5 Reader
# ============================================================================

class HDF5Reader:
    """Handle HDF5 file reading with proper error handling."""
    
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
            sdo_wavelengths: List of SDO wavelengths to read
            omni_variables: List of OMNI variables to read
            
        Returns:
            (sdo_data, omni_data) dictionaries
            
        Raises:
            FileNotFoundError: If file doesn't exist
            KeyError: If required dataset not found
            OSError: If HDF5 file is corrupted
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        sdo_data = {}
        omni_data = {}
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Read SDO data
                for wavelength in sdo_wavelengths:
                    dataset_name = f"sdo_{wavelength}"
                    if dataset_name not in f:
                        raise KeyError(f"SDO wavelength {wavelength} not found in {file_path}")
                    sdo_data[wavelength] = f[dataset_name][:]
                
                # Read OMNI data
                for variable in omni_variables:
                    dataset_name = f"omni_{variable}"
                    if dataset_name not in f:
                        raise KeyError(f"OMNI variable {variable} not found in {file_path}")
                    omni_data[variable] = f[dataset_name][:]
        
        except KeyError:
            raise
        except Exception as e:
            raise OSError(f"Failed to read HDF5 file {file_path}: {e}")
        
        return sdo_data, omni_data


# ============================================================================
# Data Processor
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
        Process SDO data.
        
        Steps:
        1. Normalize to [-1, 1]
        2. Stack wavelengths
        3. Select last N timesteps
        4. Transpose to (C, T, H, W)
        
        Args:
            sdo_data: Dictionary of {wavelength: data}
            
        Returns:
            Processed SDO array of shape (C, T, H, W)
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
        Process OMNI input data (before split_index).
        
        Steps:
        1. Z-score normalization
        2. Stack variables
        3. Select data before split_index
        4. Select last N timesteps
        
        Args:
            omni_data: Dictionary of {variable: data}
            
        Returns:
            Processed input array of shape (T, C)
        """
        input_arrays = []
        
        for variable in self.input_variables:
            data = omni_data[variable][:, None]  # Add channel dim
            # Z-score normalization
            data = self.normalizer.normalize_omni(data, variable)
            input_arrays.append(data)
        
        # Stack variables
        input_array = np.concatenate(input_arrays, axis=1)
        
        # Select before split_index
        input_array = input_array[:self.split_index]
        
        # Select last N timesteps
        input_array = input_array[-self.input_sequence_length:]
        
        return input_array
    
    def process_omni_target(self, omni_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Process OMNI target data (after split_index).
        
        Steps:
        1. Z-score normalization
        2. Stack variables
        3. Select data after split_index
        4. Select first N timesteps
        
        Args:
            omni_data: Dictionary of {variable: data}
            
        Returns:
            Processed target array of shape (T, C)
        """
        target_arrays = []
        
        for variable in self.target_variables:
            data = omni_data[variable][:, None]  # Add channel dim
            # Z-score normalization
            data = self.normalizer.normalize_omni(data, variable)
            target_arrays.append(data)
        
        # Stack variables
        target_array = np.concatenate(target_arrays, axis=1)
        
        # Select after split_index
        target_array = target_array[self.split_index:]
        
        # Select first N timesteps
        target_array = target_array[:self.target_sequence_length]
        
        return target_array
    
    def apply_augmentation(
        self,
        input_array: np.ndarray,
        sample_idx: int,
        file_name: str
    ) -> np.ndarray:
        """
        Apply data augmentation to input array.
        
        Applies random scaling factor to each element.
        Uses deterministic seeding based on sample index and file name
        for reproducibility.
        
        Args:
            input_array: Input data array
            sample_idx: Sample index in dataset (for deterministic augmentation)
            file_name: File name (for deterministic augmentation)
            
        Returns:
            Augmented array
            
        Note:
            Same sample (same idx + file_name) will always get same augmentation
        """
        # Deterministic seed based on file name and index
        seed = hash(file_name + str(sample_idx)) % (2**32)
        rng = np.random.RandomState(seed)
        factors = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        
        augmented = input_array.copy()
        for i in range(augmented.shape[0]):
            for j in range(augmented.shape[1]):
                factor = rng.choice(factors)
                augmented[i, j] *= factor
        
        return augmented


# ============================================================================
# Dataset
# ============================================================================

class MultimodalDataset(Dataset):
    """
    Multimodal dataset for solar wind prediction.
    
    Combines SDO imagery and OMNI time series data.
    """
    
    def __init__(self, config, statistics: Dict[str, Dict[str, float]]):
        """
        Args:
            config: Hydra config or DataConfig object
            statistics: Precomputed statistics for normalization
        """
        # Extract config
        self.config = self._extract_config(config)
        
        # Load file lists
        self.train_file_list = self._load_file_list(
            self.config.train_list_path,
            self.config.target_day
        )
        self.validation_file_list = self._load_file_list(
            self.config.validation_list_path,
            self.config.target_day
        )
        
        # Calculate pos_weight before sampling
        self.pos_weight = SamplingStrategy.get_pos_weight(self.train_file_list)
        
        # Apply sampling strategies
        self.indices = None  # For oversampling
        
        if self.config.enable_undersampling:
            self.train_file_list, num_pos, num_neg = SamplingStrategy.undersample(
                self.train_file_list,
                num_folds=self.config.num_subsample,
                fold_index=self.config.subsample_index
            )
            print(f"After undersampling: {len(self.train_file_list)} samples "
                  f"({num_pos} positive, {num_neg} total negative)")
        
        if self.config.enable_oversampling:
            self.indices, self.train_file_list, num_pos, num_neg = SamplingStrategy.oversample(
                self.train_file_list,
                oversample_factor=self.config.num_oversample
            )
            print(f"After oversampling: {len(self.indices)} samples "
                  f"({num_pos} positive, {num_neg} negative)")
        
        # Select appropriate file list based on phase
        if self.config.phase == 'train':
            self.file_list = self.train_file_list
        elif self.config.phase == 'validation':
            self.file_list = self.validation_file_list
        else:
            raise ValueError(f"Unknown phase: {self.config.phase}")
        
        # Initialize components
        self.normalizer = Normalizer(statistics)
        self.processor = DataProcessor(
            normalizer=self.normalizer,
            sdo_wavelengths=self.config.sdo_wavelengths,
            sdo_sequence_length=self.config.sdo_sequence_length,
            input_variables=self.config.input_variables,
            target_variables=self.config.target_variables,
            input_sequence_length=self.config.input_sequence_length,
            target_sequence_length=self.config.target_sequence_length,
            split_index=self.config.split_index
        )
        
        # Expose statistics as stat_dict for backward compatibility with plotting
        self.stat_dict = statistics
        
        print(f"Dataset initialized: {len(self)} samples in {self.config.phase} phase")
    
    def _extract_config(self, config) -> DataConfig:
        """Extract configuration into DataConfig object."""
        if isinstance(config, DataConfig):
            return config
        
        # Extract from Hydra config
        return DataConfig(
            data_root=config.environment.data_root,
            dataset_name=config.data.dataset_name,
            sdo_wavelengths=config.data.sdo_wavelengths,
            sdo_sequence_length=config.data.sdo_sequence_length,
            sdo_image_size=config.data.sdo_image_size,
            input_variables=config.data.input_variables,
            target_variables=config.data.target_variables,
            input_sequence_length=config.data.input_sequence_length,
            target_sequence_length=config.data.target_sequence_length,
            split_index=config.data.split_index,
            target_day=config.data.target_day,
            phase=config.experiment.phase,
            enable_undersampling=config.experiment.enable_undersampling,
            enable_oversampling=config.experiment.enable_oversampling,
            num_subsample=config.experiment.num_subsample if hasattr(config.experiment, 'num_subsample') else 1,
            subsample_index=config.experiment.subsample_index if hasattr(config.experiment, 'subsample_index') else 0,
            num_oversample=config.experiment.num_oversample if hasattr(config.experiment, 'num_oversample') else 1,
        )
    
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
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - sdo: SDO imagery (C, T, H, W)
                - inputs: OMNI input sequence (T, C)
                - targets: OMNI target sequence (T, C)
                - labels: Binary class label (1, 1)
                - file_names: File name string
        """
        # Map index if oversampling
        if self.indices is not None:
            real_idx = self.indices[idx]
        else:
            real_idx = idx
        
        # Get file info
        file_name, label = self.file_list[real_idx]
        file_path = f"{self.config.data_root}/oversampling/{file_name}"
        
        # Read data
        sdo_data, omni_data = HDF5Reader.read(
            file_path,
            self.config.sdo_wavelengths,
            self.config.omni_variables
        )
        
        # Process data
        sdo_array = self.processor.process_sdo(sdo_data)
        input_array = self.processor.process_omni_input(omni_data)
        target_array = self.processor.process_omni_target(omni_data)
        
        # Apply augmentation if enabled (training phase only)
        if self.config.enable_oversampling and self.config.phase == 'train':
            # Deterministic augmentation based on file name and index
            input_array = self.processor.apply_augmentation(
                input_array, 
                sample_idx=idx,
                file_name=file_name
            )
        
        # Create label array
        label_array = np.array([[label]], dtype=np.float32)
        
        # Convert to tensors
        sdo_tensor = torch.tensor(sdo_array, dtype=torch.float32)
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        target_tensor = torch.tensor(target_array, dtype=torch.float32)
        label_tensor = torch.tensor(label_array, dtype=torch.float32)
        
        return {
            'sdo': sdo_tensor,
            'inputs': input_tensor,
            'targets': target_tensor,
            'labels': label_tensor,
            'file_names': os.path.basename(file_path)
        }


# ============================================================================
# DataLoader Factory
# ============================================================================

def create_dataloader(config, logger=None) -> DataLoader:
    """
    Create a DataLoader with appropriate settings.
    
    Args:
        config: Hydra configuration
        logger: Optional logger (not used, kept for compatibility)
        
    Returns:
        Configured DataLoader
    """
    # Extract config
    if isinstance(config, DataConfig):
        data_config = config
        batch_size = 32  # Default
        num_workers = 4
        device = 'cuda'
    else:
        data_config = DataConfig(
            data_root=config.environment.data_root,
            dataset_name=config.data.dataset_name,
            sdo_wavelengths=config.data.sdo_wavelengths,
            sdo_sequence_length=config.data.sdo_sequence_length,
            sdo_image_size=config.data.sdo_image_size,
            input_variables=config.data.input_variables,
            target_variables=config.data.target_variables,
            input_sequence_length=config.data.input_sequence_length,
            target_sequence_length=config.data.target_sequence_length,
            split_index=config.data.split_index,
            target_day=config.data.target_day,
            phase=config.experiment.phase,
            enable_undersampling=config.experiment.enable_undersampling,
            enable_oversampling=config.experiment.enable_oversampling,
            num_subsample=config.experiment.num_subsample if hasattr(config.experiment, 'num_subsample') else 1,
            subsample_index=config.experiment.subsample_index if hasattr(config.experiment, 'subsample_index') else 0,
            num_oversample=config.experiment.num_oversample if hasattr(config.experiment, 'num_oversample') else 1,
        )
        batch_size = config.experiment.batch_size
        num_workers = config.experiment.num_workers
        device = config.environment.device
    
    # Compute or load statistics
    train_df = pd.read_csv(data_config.train_list_path)
    train_file_names = train_df['file_name'].tolist()
    
    statistics = compute_statistics(
        data_root=data_config.data_root,
        data_file_list=train_file_names,
        variables=data_config.omni_variables,
        stat_file_path=data_config.stat_file_path,
        overwrite=False
    )
    
    # Create dataset
    dataset = MultimodalDataset(data_config, statistics)
    
    # Create dataloader
    # Note: PyTorch automatically propagates the main process seed to workers
    # So if train.py sets torch.manual_seed(), workers will be seeded deterministically
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(data_config.phase == 'train'),
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return dataloader


# ============================================================================
# Main (for testing)
# ============================================================================

@hydra.main(config_path="./configs", version_base=None)
def main(config):
    """Test the pipeline."""
    # Set seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("="*80)
    print("Testing Refactored Pipeline")
    print("="*80)
    
    # Create dataloader
    dataloader = create_dataloader(config)
    
    print(f"\nDataLoader created:")
    print(f"  Dataset size: {len(dataloader.dataset)}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    print(f"  Pos weight: {dataloader.dataset.pos_weight:.3f}")
    
    # Test loading speed
    print(f"\nTesting loading speed (3 epochs):")
    for epoch in range(3):
        t0 = time.time()
        for i, batch in enumerate(dataloader):
            sdo = batch['sdo'].numpy()
            inputs = batch['inputs'].numpy()
            targets = batch['targets'].numpy()
            labels = batch['labels'].numpy()
            
            if i == 0:  # Print first batch info
                print(f"\nEpoch {epoch} - First batch:")
                print(f"  SDO shape: {sdo.shape}, mean: {sdo.mean():.3f}, std: {sdo.std():.3f}")
                print(f"  Inputs shape: {inputs.shape}, mean: {inputs.mean():.3f}, std: {inputs.std():.3f}")
                print(f"  Targets shape: {targets.shape}, mean: {targets.mean():.3f}, std: {targets.std():.3f}")
                print(f"  Labels shape: {labels.shape}, mean: {labels.mean():.3f}")
            
            # Only process a few batches for speed test
            if i >= 2:
                break
        
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}: {elapsed:.2f}s (first 3 batches)")
    
    print("\n" + "="*80)
    print("Pipeline test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()