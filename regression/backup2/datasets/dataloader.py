"""
DataLoader factory for multimodal dataset.

This module provides a factory function to create PyTorch DataLoaders
with appropriate settings for training and validation.
"""

import pandas as pd
from torch.utils.data import DataLoader

from .config import DataConfig
from .statistics import compute_statistics
from .dataset import TrainDataset, ValidationDataset


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
            dataset_dir_name=config.data.dataset_dir_name,
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
        dataset_dir_name=data_config.dataset_dir_name,
        data_file_list=train_file_names,
        variables=data_config.omni_variables,
        stat_file_path=data_config.stat_file_path,
        overwrite=False
    )
    
    # Create dataset based on phase
    if data_config.phase == 'train':
        dataset = TrainDataset(data_config, statistics)
    elif data_config.phase == 'validation':
        dataset = ValidationDataset(data_config, statistics)
    else:
        raise ValueError(f"Unknown phase: {data_config.phase}. Must be 'train' or 'validation'")
    
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
