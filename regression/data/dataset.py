"""
PyTorch Dataset classes for multimodal solar wind prediction.

This module provides:
- BaseMultimodalDataset: Common functionality
- TrainDataset: Training with sampling and augmentation
- ValidationDataset: Validation without sampling or augmentation
"""

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from .config import DataConfig
from .statistics import Normalizer
from .preprocessing import DataProcessor
from .sampling import SamplingStrategy
from .io import HDF5Reader


class BaseMultimodalDataset(Dataset):
    """
    Base class for multimodal dataset.
    
    Provides common functionality for both training and validation datasets.
    """
    
    def __init__(self, config: DataConfig, statistics: Dict[str, Dict[str, float]]):
        """
        Args:
            config: DataConfig object
            statistics: Precomputed statistics for normalization
        """
        self.config = config
        
        # Load file lists
        self.train_file_list = self._load_file_list(
            self.config.train_list_path,
            self.config.target_day
        )
        self.validation_file_list = self._load_file_list(
            self.config.validation_list_path,
            self.config.target_day
        )
        
        # Calculate pos_weight before sampling (for loss function)
        self.pos_weight = SamplingStrategy.get_pos_weight(self.train_file_list)
        
        # Initialize normalizer and processor
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
        
        # Expose statistics for backward compatibility with plotting
        self.stat_dict = statistics
        
        # To be set by subclasses
        self.file_list = None
        self.indices = None
    
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
    
    def _load_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
        """
        Load and process data for a given index.
        
        Args:
            idx: Sample index
            
        Returns:
            (sdo_array, input_array, target_array, label, file_name)
        """
        # Map index if oversampling is used
        if self.indices is not None:
            real_idx = self.indices[idx]
        else:
            real_idx = idx
        
        # Get file info
        file_name, label = self.file_list[real_idx]
        file_path = f"{self.config.data_root}/original/{file_name}"
        
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
        
        return sdo_array, input_array, target_array, label, file_name
    
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
        # Load data
        sdo_array, input_array, target_array, label, file_name = self._load_data(idx)
        
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
            'file_names': os.path.basename(file_name)
        }


class TrainDataset(BaseMultimodalDataset):
    """
    Training dataset with sampling and augmentation support.
    
    Applies:
    - Undersampling (optional)
    - Oversampling (optional)
    - Data augmentation (optional)
    """
    
    def __init__(self, config: DataConfig, statistics: Dict[str, Dict[str, float]]):
        """
        Args:
            config: DataConfig object
            statistics: Precomputed statistics for normalization
        """
        super().__init__(config, statistics)
        
        # Start with train file list
        self.file_list = self.train_file_list.copy()
        
        # Apply undersampling if enabled
        if self.config.enable_undersampling:
            self.file_list, num_pos, num_neg = SamplingStrategy.undersample(
                self.file_list,
                num_folds=self.config.num_subsample,
                fold_index=self.config.subsample_index
            )
            print(f"After undersampling: {len(self.file_list)} samples "
                  f"({num_pos} positive, {num_neg} total negative)")
        
        # Apply oversampling if enabled
        if self.config.enable_oversampling:
            self.indices, self.file_list, num_pos, num_neg = SamplingStrategy.oversample(
                self.file_list,
                oversample_factor=self.config.num_oversample
            )
            print(f"After oversampling: {len(self.indices)} samples "
                  f"({num_pos} positive, {num_neg} negative)")
        
        print(f"TrainDataset initialized: {len(self)} samples")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample with augmentation.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing processed and augmented data
        """
        # Load data
        sdo_array, input_array, target_array, label, file_name = self._load_data(idx)
        
        # Apply augmentation if enabled
        if self.config.enable_oversampling:
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
            'file_names': os.path.basename(file_name)
        }


class ValidationDataset(BaseMultimodalDataset):
    """
    Validation dataset without sampling or augmentation.
    
    Uses original data distribution for accurate evaluation.
    """
    
    def __init__(self, config: DataConfig, statistics: Dict[str, Dict[str, float]]):
        """
        Args:
            config: DataConfig object
            statistics: Precomputed statistics for normalization
        """
        super().__init__(config, statistics)
        
        # Use validation file list as-is (no sampling)
        self.file_list = self.validation_file_list
        
        print(f"ValidationDataset initialized: {len(self)} samples")
