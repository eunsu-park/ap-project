"""
Experiment utilities: logging, device setup, and seed configuration.

Consolidated from logging_utils, device, and seed modules.
"""

import os
import random
import logging
from datetime import datetime
from typing import Optional

import torch
import numpy as np


# ============================================================================
# Global Logger
# ============================================================================

_global_logger: Optional[logging.Logger] = None


def setup_experiment(config, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Setup complete experiment environment: logger, seed, and device.
    
    Args:
        config: Configuration object with environment settings
        log_dir: Optional log directory (defaults to config.log_dir if available)
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    
    # Determine log directory
    if log_dir is None and hasattr(config, 'log_dir'):
        log_dir = config.log_dir
    
    # Setup logger
    _global_logger = _setup_logger('solar_wind', log_dir)
    
    # Set seed
    set_seed(config.environment.seed)
    
    # Setup device (info logged automatically)
    device = setup_device(config)
    
    return _global_logger


def get_logger() -> logging.Logger:
    """
    Get the global logger instance.
    
    Returns:
        Global logger. If not initialized, returns a basic logger.
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = logging.getLogger('solar_wind')
        if not _global_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            _global_logger.addHandler(handler)
            _global_logger.setLevel(logging.INFO)
    return _global_logger


def _setup_logger(name: str, log_dir: Optional[str] = None, 
                  level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'run_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file: {log_file}")
    
    return logger


# ============================================================================
# Seed Configuration
# ============================================================================

def set_seed(seed: int = 250104) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    logger = get_logger()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed: {seed}")


# ============================================================================
# Device Configuration
# ============================================================================

def setup_device(config) -> torch.device:
    """
    Setup computation device with fallback options.
    
    Args:
        config: Configuration object with environment.device
        
    Returns:
        Configured torch.device
    """
    logger = get_logger()
    requested_device = config.environment.device
    
    # CUDA
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA: {gpu_name}")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
    
    # MPS (Apple Silicon)
    elif requested_device == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            logger.warning("MPS not available, using CPU")
    
    # CPU
    elif requested_device == 'cpu':
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Unknown
    else:
        device = torch.device('cpu')
        logger.warning(f"Unknown device '{requested_device}', using CPU")
    
    return device


# ============================================================================
# Legacy compatibility
# ============================================================================

def setup_logger(name: str, log_dir: Optional[str] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """Legacy function for backward compatibility."""
    return _setup_logger(name, log_dir, level)
