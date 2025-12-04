"""
Multimodal dataset pipeline for solar wind prediction.

This package provides data loading and preprocessing utilities for
combining SDO imagery and OMNI time series data.

Main components:
- DataConfig: Configuration management
- TrainDataset, ValidationDataset: PyTorch datasets
- create_dataloader: Factory function for creating dataloaders
- Normalizer: Data normalization
- compute_statistics: Statistics computation and caching

Example usage:
    from data import TrainDataset, ValidationDataset, create_dataloader
    
    # Using the factory function (recommended)
    train_loader = create_dataloader(config)  # config.experiment.phase = 'train'
    val_loader = create_dataloader(config)    # config.experiment.phase = 'validation'
    
    # Or directly
    from data import DataConfig, compute_statistics
    
    config = DataConfig(...)
    stats = compute_statistics(...)
    train_dataset = TrainDataset(config, stats)
    val_dataset = ValidationDataset(config, stats)
"""

# Configuration
from .config import DataConfig

# Statistics and normalization
from .statistics import Normalizer, OnlineStatistics, compute_statistics

# I/O
from .io import HDF5Reader

# Preprocessing
from .preprocessing import DataProcessor

# Sampling
from .sampling import SamplingStrategy

# Datasets
from .dataset import BaseMultimodalDataset, TrainDataset, ValidationDataset

# DataLoader factory
from .dataloader import create_dataloader


__all__ = [
    # Configuration
    'DataConfig',
    
    # Statistics and normalization
    'Normalizer',
    'OnlineStatistics',
    'compute_statistics',
    
    # I/O
    'HDF5Reader',
    
    # Preprocessing
    'DataProcessor',
    
    # Sampling
    'SamplingStrategy',
    
    # Datasets (most commonly used)
    'BaseMultimodalDataset',
    'TrainDataset',
    'ValidationDataset',
    
    # DataLoader factory (most commonly used)
    'create_dataloader',
]
