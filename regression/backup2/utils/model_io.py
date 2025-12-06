"""
Model I/O utilities for loading and saving models.

This module provides functions for loading PyTorch models
from checkpoints with comprehensive error handling.
"""

import os
import logging
from typing import Optional

import torch


def load_model(model: torch.nn.Module, checkpoint_path: str, 
              device: torch.device, logger: Optional[logging.Logger] = None) -> torch.nn.Module:
    """Load model from checkpoint with comprehensive error handling.
    
    Args:
        model: PyTorch model instance.
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.
        logger: Optional logger for output.
        
    Returns:
        Loaded model.
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If model loading fails.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict into model
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        message = f"Model loaded successfully from: {checkpoint_path}"
        _log_message(logger, message, logging.INFO)
        
        return model
        
    except Exception as e:
        error_msg = f"Failed to load model from {checkpoint_path}: {e}"
        _log_message(logger, error_msg, logging.ERROR)
        raise RuntimeError(error_msg)


def _log_message(logger: Optional[logging.Logger], message: str, 
                level: int = logging.INFO) -> None:
    """Helper function for logging."""
    if logger:
        logger.log(level, message)
    else:
        print(message)
