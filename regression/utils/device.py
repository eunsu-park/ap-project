"""
Device configuration utilities.

This module provides functions to setup computation devices
(CPU, CUDA, MPS) with fallback options.
"""

import logging
from typing import Optional

import torch


def setup_device(options, logger: Optional[logging.Logger] = None) -> torch.device:
    """Setup computation device with fallback options.
    
    Args:
        options: Configuration object containing device preference.
        logger: Optional logger for output.
        
    Returns:
        Configured torch.device.
    """
    requested_device = options.environment.device
    
    # Check CUDA availability
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            message = f"Using CUDA device: {gpu_name}"
        else:
            device = torch.device('cpu')
            message = "CUDA requested but not available, falling back to CPU"
            _log_message(logger, message, logging.WARNING)
    
    # Check MPS availability (Apple Silicon)
    elif requested_device == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            message = "Using MPS device (Apple Silicon)"
        else:
            device = torch.device('cpu')
            message = "MPS requested but not available, falling back to CPU"
            _log_message(logger, message, logging.WARNING)
    
    # CPU device
    elif requested_device == 'cpu':
        device = torch.device('cpu')
        message = "Using CPU device"
    
    else:
        device = torch.device('cpu')
        message = f"Unknown device '{requested_device}', falling back to CPU"
        _log_message(logger, message, logging.WARNING)
    
    _log_message(logger, f"Device configured: {device}", logging.INFO)
    return device


def _log_message(logger: Optional[logging.Logger], message: str, 
                level: int = logging.INFO) -> None:
    """Helper function for logging."""
    if logger:
        logger.log(level, message)
    else:
        print(message)
