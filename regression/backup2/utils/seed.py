"""
Random seed utilities for reproducibility.

This module provides functions to set random seeds across all
relevant libraries (Python, NumPy, PyTorch).
"""

import os
import random
import logging
from typing import Optional

import torch
import numpy as np


def set_seed(seed: int = 250104, logger: Optional[logging.Logger] = None) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
        logger: Optional logger for output.
    """
    random.seed(seed)  # For built-in random module
    np.random.seed(seed)  # For numpy
    torch.manual_seed(seed)  # For CPU
    
    if torch.cuda.is_available():  # For GPUs
        torch.cuda.manual_seed(seed)  # For single GPU
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  # For hash-based operations
    
    message = f"Random seed set to: {seed}"
    if logger:
        logger.info(message)
    else:
        print(message)
