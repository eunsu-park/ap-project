"""
Utility functions for solar wind prediction project.

This package provides various utility functions including:
- Random seed setting for reproducibility
- Logger configuration
- Device setup (CPU/CUDA/MPS)
- Model loading from checkpoints
- Visualization and plotting
- Metrics calculation
- SLURM job submission

Main components:
- set_seed: Set random seeds for reproducibility
- setup_logger: Configure logging
- setup_device: Configure computation device
- load_model: Load model from checkpoint
- save_plot: Save comparison plots
- calculate_metrics: Calculate evaluation metrics
- WulverSubmitter: SLURM job submission

Example usage:
    from utils import set_seed, setup_logger, setup_device
    
    # Setup environment
    set_seed(42)
    logger = setup_logger(__name__, log_dir='./logs')
    device = setup_device(config, logger)
    
    # Visualization
    from utils import save_plot
    save_plot(targets, outputs, var_names, stats, 'plot', 'Title')
"""

# Seed utilities
from .seed import set_seed

# Logging utilities
from .logging_utils import setup_logger, log_message

# Device utilities
from .device import setup_device

# Model I/O utilities
from .model_io import load_model

# Visualization utilities
from .visualization import (
    save_plot,
    denormalize_predictions,
    create_comparison_plot,
    save_data_h5
)

# Metrics utilities
from .metrics import calculate_metrics

# SLURM utilities
from .slurm import WulverSubmitter


__all__ = [
    # Seed
    'set_seed',
    
    # Logging
    'setup_logger',
    'log_message',
    
    # Device
    'setup_device',
    
    # Model I/O
    'load_model',
    
    # Visualization (most commonly used)
    'save_plot',
    'denormalize_predictions',
    'create_comparison_plot',
    'save_data_h5',
    
    # Metrics
    'calculate_metrics',
    
    # SLURM
    'WulverSubmitter',
]
