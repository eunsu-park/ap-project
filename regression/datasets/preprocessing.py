"""
Data preprocessing for multimodal dataset.

This module handles preprocessing of SDO imagery and OMNI time series data,
including normalization, sequence extraction, and data augmentation.
"""

from typing import Dict, List

import numpy as np

from .statistics import Normalizer


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
