"""
Regression loss functions with various weighting strategies.

This module provides weighted regression losses for handling temporal
sequences, outliers, and rapid changes.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss with custom time-based weighting.
    
    Applies different weights based on prediction time horizons:
    - First 8 points: weight 0.5
    - Next 16 points (9-24): weight 0.3  
    - Last 24 points would be: weight 0.2 (but sequence length is 24 total)
    
    Args:
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def _compute_time_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute time-based weights for the sequence.
        
        Args:
            seq_len: Length of the sequence.
            device: Device to create tensor on.
            
        Returns:
            Weight tensor of shape (seq_len,).
        """
        weights = torch.zeros(seq_len, device=device)
        
        # First 8 points: weight 0.5
        end_first = min(8, seq_len)
        weights[:end_first] = 0.5
        
        # Next 16 points (9-24): weight 0.3
        if seq_len > 8:
            end_second = min(24, seq_len)
            weights[8:end_second] = 0.3
        
        # Remaining points: weight 0.2
        if seq_len > 24:
            weights[24:] = 0.2
            
        return weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the weighted MSE loss function.
        
        Args:
            pred: Predicted values of shape (batch_size, seq_len, feature_dim).
            target: Target values of shape (batch_size, seq_len, feature_dim).
            
        Returns:
            Computed weighted loss value.
        """
        batch_size, seq_len, feature_dim = pred.shape
        
        # Compute base MSE loss
        base_loss = self.mse_loss(pred, target)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Compute time-based weights
        time_weights = self._compute_time_weights(seq_len, pred.device)
        time_weights = time_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)
        
        # Apply weights to loss
        weighted_loss = base_loss * time_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class HuberMultiCriteriaLoss(nn.Module):
    """Huber Loss with Multi-criteria Weighting (Highest Priority).
    
    Combines Huber loss with temporal weighting (future emphasis) and 
    gradient-based weighting (emphasizes rapid changes).
    
    Args:
        beta: Threshold for Huber loss transition between L2 and L1.
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        gradient_weight_scale: Scale factor for gradient-based weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(self, 
                 beta: float = 0.3,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 gradient_weight_scale: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.temporal_weight_range = temporal_weight_range
        self.gradient_weight_scale = gradient_weight_scale
        self.reduction = reduction
        self.huber_loss = nn.SmoothL1Loss(beta=beta, reduction='none')
    
    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps.
        
        Args:
            seq_len: Length of the sequence.
            device: Device to create tensor on.
            
        Returns:
            Temporal weights tensor of shape (seq_len,).
        """
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)
    
    def _compute_gradient_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute weights based on gradient magnitude (emphasizes rapid changes).
        
        Args:
            target: Target tensor of shape (batch_size, seq_len, feature_dim).
            
        Returns:
            Gradient-based weights tensor of shape (batch_size, seq_len).
        """
        # Compute temporal gradient (difference between consecutive timesteps)
        grad = torch.diff(target, dim=1)  # Shape: (batch_size, seq_len-1, feature_dim)
        grad_magnitude = torch.norm(grad, dim=2)  # Shape: (batch_size, seq_len-1)
        
        # Pad to match original sequence length
        grad_magnitude = F.pad(grad_magnitude, (1, 0), value=0)  # Shape: (batch_size, seq_len)
        
        # Apply exponential scaling to emphasize high gradients
        max_grad = grad_magnitude.max(dim=1, keepdim=True)[0] + 1e-8
        normalized_grad = grad_magnitude / max_grad
        grad_weights = torch.exp(normalized_grad * self.gradient_weight_scale)
        
        return torch.clamp(grad_weights, min=0.1, max=3.0)  # Prevent extreme weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function.
        
        Args:
            pred: Predicted values of shape (batch_size, seq_len, feature_dim).
            target: Target values of shape (batch_size, seq_len, feature_dim).
            
        Returns:
            Computed loss value.
        """
        batch_size, seq_len, feature_dim = pred.shape
        
        # Compute base Huber loss
        base_loss = self.huber_loss(pred, target)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Compute temporal weights (future emphasis)
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)
        
        # Compute gradient weights (rapid change emphasis)
        gradient_weights = self._compute_gradient_weights(target)
        gradient_weights = gradient_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)
        
        # Combine weights
        combined_weights = temporal_weights * gradient_weights
        
        # Apply weights to loss
        weighted_loss = base_loss * combined_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class MAEOutlierFocusedLoss(nn.Module):
    """MAE Loss with Outlier Detection and Future Temporal Weighting (Second Priority).
    
    Uses MAE loss (robust to outliers) with outlier-based weighting and future emphasis.
    
    Args:
        outlier_threshold: Z-score threshold for outlier detection.
        outlier_weight_multiplier: Multiplier for outlier regions.
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(self, 
                 outlier_threshold: float = 2.0,
                 outlier_weight_multiplier: float = 3.0,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 reduction: str = 'mean'):
        super().__init__()
        self.outlier_threshold = outlier_threshold
        self.outlier_weight_multiplier = outlier_weight_multiplier
        self.temporal_weight_range = temporal_weight_range
        self.reduction = reduction
        self.mae_loss = nn.L1Loss(reduction='none')
    
    def _detect_outliers(self, target: torch.Tensor) -> torch.Tensor:
        """Detect outliers using Z-score method.
        
        Args:
            target: Target tensor of shape (batch_size, seq_len, feature_dim).
            
        Returns:
            Outlier weights tensor of shape (batch_size, seq_len).
        """
        # Compute statistics across feature dimensions
        mean = target.mean(dim=2, keepdim=True)  # Shape: (batch_size, seq_len, 1)
        std = target.std(dim=2, keepdim=True) + 1e-8  # Add epsilon for stability
        
        # Compute Z-scores
        z_scores = torch.abs((target - mean) / std)  # Shape: (batch_size, seq_len, feature_dim)
        max_z_score = z_scores.max(dim=2)[0]  # Shape: (batch_size, seq_len)
        
        # Create outlier weights
        outlier_mask = (max_z_score > self.outlier_threshold).float()
        outlier_weights = 1.0 + outlier_mask * (self.outlier_weight_multiplier - 1.0)
        
        return outlier_weights
    
    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function."""
        batch_size, seq_len, feature_dim = pred.shape
        
        # Compute base MAE loss
        base_loss = self.mae_loss(pred, target)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Compute temporal weights (future emphasis)
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)
        
        # Compute outlier weights
        outlier_weights = self._detect_outliers(target)
        outlier_weights = outlier_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)
        
        # Combine weights
        combined_weights = temporal_weights * outlier_weights
        
        # Apply weights to loss
        weighted_loss = base_loss * combined_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss
