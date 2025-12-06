"""
Model utilities: loading/saving models and computing metrics.

Consolidated from model_io and metrics modules.
"""

import os
from typing import List, Optional

import torch
import numpy as np

from .experiment import get_logger


# ============================================================================
# Model I/O
# ============================================================================

def load_model(model: torch.nn.Module, checkpoint_path: str, 
               device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model: PyTorch model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger = get_logger()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
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
    
    # Load into model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded: {checkpoint_path}")
    return model


def save_model(model: torch.nn.Module, save_path: str, 
               epoch: Optional[int] = None, optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        save_path: Path to save checkpoint
        epoch: Optional epoch number
        optimizer: Optional optimizer state
    """
    logger = get_logger()
    
    checkpoint = {
        'model_state_dict': model.state_dict()
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved: {save_path}")


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_metrics(targets: np.ndarray, predictions: np.ndarray, 
                     variable_names: List[str]) -> dict:
    """
    Calculate regression metrics for each variable.
    
    Args:
        targets: Ground truth (n_samples, seq_len, n_vars)
        predictions: Predictions (n_samples, seq_len, n_vars)
        variable_names: List of variable names
        
    Returns:
        Dictionary of metrics per variable
    """
    if targets.shape != predictions.shape:
        raise ValueError(f"Shape mismatch: {targets.shape} != {predictions.shape}")
    
    metrics = {}
    
    for var_idx, var_name in enumerate(variable_names):
        target_var = targets[..., var_idx].flatten()
        pred_var = predictions[..., var_idx].flatten()
        
        # Compute metrics
        mse = float(np.mean((target_var - pred_var) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(target_var - pred_var)))
        
        # Correlation
        correlation = np.corrcoef(target_var, pred_var)[0, 1]
        correlation = float(correlation) if not np.isnan(correlation) else 0.0
        
        # R²
        ss_res = np.sum((target_var - pred_var) ** 2)
        ss_tot = np.sum((target_var - target_var.mean()) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        metrics[var_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'r2': r2
        }
    
    return metrics
