"""
Configuration dataclass for multimodal solar wind prediction dataset.

This module contains the DataConfig dataclass that holds all configuration
parameters for the dataset pipeline.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Data configuration container for type safety and clarity."""
    
    # Paths
    data_root: str
    dataset_dir_name: str
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
    def dataset_path(self) -> str:
        """Get full dataset path."""
        return f"{self.data_root}/{self.dataset_name}"
    
    @property
    def train_list_path(self) -> str:
        """Get training list CSV path."""
        return f"{self.dataset_path}_train.csv"
    
    @property
    def validation_list_path(self) -> str:
        """Get validation list CSV path."""
        return f"{self.dataset_path}_validation.csv"
    
    @property
    def stat_file_path(self) -> str:
        """Get statistics pickle file path."""
        return f"{self.dataset_path}_statistics.pkl"
    
    @property
    def omni_variables(self) -> List[str]:
        """Get all OMNI variables (input + target, deduplicated)."""
        return list(set(self.input_variables + self.target_variables))
