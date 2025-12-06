"""
Metrics calculation utilities.

This module provides functions for calculating evaluation metrics
such as MSE, RMSE, MAE, correlation, and R-squared.
"""

from typing import List

import numpy as np


def calculate_metrics(targets: np.ndarray, predictions: np.ndarray, 
                     variable_names: List[str]) -> dict:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        targets: Ground truth values of shape (n_samples, seq_len, n_vars).
        predictions: Predicted values of shape (n_samples, seq_len, n_vars).
        variable_names: List of variable names.
        
    Returns:
        Dictionary containing metrics for each variable.
        
    Raises:
        ValueError: If input shapes don't match.
    """
    if targets.shape != predictions.shape:
        raise ValueError(f"Shape mismatch: targets {targets.shape} != predictions {predictions.shape}")
    
    metrics = {}
    
    for var_idx, var_name in enumerate(variable_names):
        target_var = targets[..., var_idx]
        pred_var = predictions[..., var_idx]
        
        # Calculate metrics
        mse = np.mean((target_var - pred_var) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(target_var - pred_var))
        
        # Calculate correlation coefficient
        target_flat = target_var.flatten()
        pred_flat = pred_var.flatten()
        correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
        
        # Calculate R-squared
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics[var_name] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'r_squared': float(r_squared)
        }
    
    return metrics
