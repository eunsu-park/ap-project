"""
Advanced loss functions with sophisticated weighting strategies.

This module provides advanced regression losses including adaptive weighting,
gradient-based emphasis, quantile regression, and multi-task learning.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWeightLoss(nn.Module):
    """Adaptive Weight Loss with Dynamic Error-based Weighting (Third Priority).
    
    Automatically assigns higher weights to larger errors (outliers/rapid changes).
    
    Args:
        base_loss_type: Type of base loss ('mse', 'mae', 'huber').
        beta: Beta parameter for Huber loss (if applicable).
        adaptive_power: Power for adaptive weighting (higher = more emphasis on large errors).
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(self, 
                 base_loss_type: str = 'huber',
                 beta: float = 0.5,
                 adaptive_power: float = 1.5,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 reduction: str = 'mean'):
        super().__init__()
        self.base_loss_type = base_loss_type
        self.adaptive_power = adaptive_power
        self.temporal_weight_range = temporal_weight_range
        self.reduction = reduction
        
        if base_loss_type == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss_type == 'mae':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss_type == 'huber':
            self.base_loss_fn = nn.SmoothL1Loss(beta=beta, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {base_loss_type}")
    
    def _compute_adaptive_weights(self, errors: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on error magnitude.
        
        Args:
            errors: Error tensor of shape (batch_size, seq_len, feature_dim).
            
        Returns:
            Adaptive weights tensor of shape (batch_size, seq_len).
        """
        # Compute error magnitude across feature dimensions
        error_magnitude = torch.norm(errors, dim=2)  # Shape: (batch_size, seq_len)
        
        # Apply power scaling and normalization
        adaptive_weights = torch.pow(error_magnitude + 1e-8, self.adaptive_power)
        
        # Normalize weights per sequence to prevent extreme scaling
        adaptive_weights = adaptive_weights / (adaptive_weights.mean(dim=1, keepdim=True) + 1e-8)
        
        return torch.clamp(adaptive_weights, min=0.1, max=5.0)
    
    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function."""
        batch_size, seq_len, feature_dim = pred.shape
        
        # Compute base loss
        base_loss = self.base_loss_fn(pred, target)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Compute errors for adaptive weighting
        errors = torch.abs(pred - target)
        adaptive_weights = self._compute_adaptive_weights(errors)
        adaptive_weights = adaptive_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)
        
        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)
        
        # Combine weights
        combined_weights = temporal_weights * adaptive_weights
        
        # Apply weights to loss
        weighted_loss = base_loss * combined_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class GradientBasedWeightLoss(nn.Module):
    """Gradient-based Weight Loss focusing on rapid changes (Fourth Priority).
    
    Emphasizes timesteps with high temporal gradients (rapid changes).
    
    Args:
        base_loss_type: Type of base loss ('mse', 'mae', 'huber').
        beta: Beta parameter for Huber loss (if applicable).
        gradient_weight_scale: Scale factor for gradient-based weighting.
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(self, 
                 base_loss_type: str = 'mae',
                 beta: float = 0.5,
                 gradient_weight_scale: float = 3.0,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 reduction: str = 'mean'):
        super().__init__()
        self.base_loss_type = base_loss_type
        self.gradient_weight_scale = gradient_weight_scale
        self.temporal_weight_range = temporal_weight_range
        self.reduction = reduction
        
        if base_loss_type == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss_type == 'mae':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss_type == 'huber':
            self.base_loss_fn = nn.SmoothL1Loss(beta=beta, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {base_loss_type}")
    
    def _compute_gradient_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute weights based on temporal gradient magnitude."""
        # Compute temporal gradients
        grad = torch.diff(target, dim=1)  # Shape: (batch_size, seq_len-1, feature_dim)
        grad_magnitude = torch.norm(grad, dim=2)  # Shape: (batch_size, seq_len-1)
        
        # Pad to match original sequence length
        grad_magnitude = F.pad(grad_magnitude, (1, 0), value=0)  # Shape: (batch_size, seq_len)
        
        # Apply exponential weighting to emphasize high gradients
        max_grad = grad_magnitude.max(dim=1, keepdim=True)[0] + 1e-8
        normalized_grad = grad_magnitude / max_grad
        gradient_weights = torch.exp(normalized_grad * self.gradient_weight_scale)
        
        return torch.clamp(gradient_weights, min=0.2, max=4.0)
    
    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function."""
        batch_size, seq_len, feature_dim = pred.shape
        
        # Compute base loss
        base_loss = self.base_loss_fn(pred, target)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Compute gradient weights
        gradient_weights = self._compute_gradient_weights(target)
        gradient_weights = gradient_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)
        
        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)
        
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


class QuantileLoss(nn.Module):
    """Quantile Loss with uncertainty-based weighting (Fifth Priority).
    
    Provides prediction intervals along with point estimates.
    
    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        uncertainty_weight_scale: Scale factor for uncertainty-based weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(self, 
                 quantiles: list = [0.1, 0.5, 0.9],
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 uncertainty_weight_scale: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.quantiles = quantiles
        self.temporal_weight_range = temporal_weight_range
        self.uncertainty_weight_scale = uncertainty_weight_scale
        self.reduction = reduction
    
    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
        """Compute quantile loss for a specific quantile.
        
        Args:
            pred: Predicted values.
            target: Target values.
            quantile: Quantile level.
            
        Returns:
            Quantile loss.
        """
        errors = target - pred
        loss = torch.max((quantile - 1) * errors, quantile * errors)
        return loss
    
    def _compute_uncertainty_weights(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty weights based on prediction interval width.
        
        Args:
            pred: Predicted quantiles of shape (batch_size, seq_len, feature_dim, num_quantiles).
            
        Returns:
            Uncertainty weights of shape (batch_size, seq_len).
        """
        # Compute prediction interval width (difference between high and low quantiles)
        interval_width = pred[..., -1] - pred[..., 0]  # Shape: (batch_size, seq_len, feature_dim)
        interval_width = torch.mean(interval_width, dim=2)  # Shape: (batch_size, seq_len)
        
        # Normalize and apply exponential weighting
        max_width = interval_width.max(dim=1, keepdim=True)[0] + 1e-8
        normalized_width = interval_width / max_width
        uncertainty_weights = torch.exp(normalized_width * self.uncertainty_weight_scale)
        
        return torch.clamp(uncertainty_weights, min=0.3, max=3.0)
    
    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the quantile loss function.
        
        Args:
            pred: Predicted quantiles of shape (batch_size, seq_len, feature_dim, num_quantiles).
            target: Target values of shape (batch_size, seq_len, feature_dim).
            
        Returns:
            Computed loss value.
        """
        batch_size, seq_len, feature_dim = target.shape
        
        # Ensure pred has the correct number of quantiles
        if pred.shape[-1] != len(self.quantiles):
            raise ValueError(f"Expected {len(self.quantiles)} quantiles, got {pred.shape[-1]}")
        
        # Compute quantile losses
        total_loss = 0
        for i, quantile in enumerate(self.quantiles):
            pred_q = pred[..., i]  # Shape: (batch_size, seq_len, feature_dim)
            q_loss = self._quantile_loss(pred_q, target, quantile)
            total_loss += q_loss
        
        # Average over quantiles
        total_loss = total_loss / len(self.quantiles)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Compute uncertainty weights
        uncertainty_weights = self._compute_uncertainty_weights(pred)
        uncertainty_weights = uncertainty_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)
        
        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)
        
        # Combine weights
        combined_weights = temporal_weights * uncertainty_weights
        
        # Apply weights to loss
        weighted_loss = total_loss * combined_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class MultiTaskLoss(nn.Module):
    """Multi-task Learning Loss for simultaneous regression and outlier detection (Sixth Priority).
    
    Combines regression loss with auxiliary outlier detection task.
    
    Args:
        regression_loss_type: Type of regression loss ('mse', 'mae', 'huber').
        beta: Beta parameter for Huber loss (if applicable).
        outlier_loss_weight: Weight for outlier detection loss.
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        outlier_threshold: Z-score threshold for outlier labeling.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(self, 
                 regression_loss_type: str = 'huber',
                 beta: float = 0.5,
                 outlier_loss_weight: float = 0.3,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 outlier_threshold: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.regression_loss_type = regression_loss_type
        self.outlier_loss_weight = outlier_loss_weight
        self.temporal_weight_range = temporal_weight_range
        self.outlier_threshold = outlier_threshold
        self.reduction = reduction
        
        # Regression loss
        if regression_loss_type == 'mse':
            self.regression_loss_fn = nn.MSELoss(reduction='none')
        elif regression_loss_type == 'mae':
            self.regression_loss_fn = nn.L1Loss(reduction='none')
        elif regression_loss_type == 'huber':
            self.regression_loss_fn = nn.SmoothL1Loss(beta=beta, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {regression_loss_type}")
        
        # Outlier detection loss
        self.outlier_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    def _generate_outlier_labels(self, target: torch.Tensor) -> torch.Tensor:
        """Generate outlier labels using Z-score method.
        
        Args:
            target: Target tensor of shape (batch_size, seq_len, feature_dim).
            
        Returns:
            Outlier labels tensor of shape (batch_size, seq_len).
        """
        # Compute statistics across feature dimensions
        mean = target.mean(dim=2, keepdim=True)
        std = target.std(dim=2, keepdim=True) + 1e-8
        
        # Compute Z-scores
        z_scores = torch.abs((target - mean) / std)
        max_z_score = z_scores.max(dim=2)[0]  # Shape: (batch_size, seq_len)
        
        # Create binary outlier labels
        outlier_labels = (max_z_score > self.outlier_threshold).float()
        
        return outlier_labels
    
    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                outlier_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the multi-task loss function.
        
        Args:
            pred: Predicted regression values of shape (batch_size, seq_len, feature_dim).
            target: Target values of shape (batch_size, seq_len, feature_dim).
            outlier_logits: Predicted outlier logits of shape (batch_size, seq_len).
                          If None, outlier detection loss is skipped.
            
        Returns:
            Combined loss value.
        """
        batch_size, seq_len, feature_dim = pred.shape
        
        # Compute regression loss
        regression_loss = self.regression_loss_fn(pred, target)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)
        
        # Apply temporal weights to regression loss
        weighted_regression_loss = regression_loss * temporal_weights
        
        # Compute total loss
        total_loss = weighted_regression_loss
        
        # Add outlier detection loss if outlier logits are provided
        if outlier_logits is not None:
            outlier_labels = self._generate_outlier_labels(target)  # Shape: (batch_size, seq_len)
            outlier_loss = self.outlier_loss_fn(outlier_logits, outlier_labels)  # Shape: (batch_size, seq_len)
            
            # Apply temporal weights to outlier loss
            temporal_weights_2d = temporal_weights.squeeze(2)  # Shape: (1, seq_len)
            weighted_outlier_loss = outlier_loss * temporal_weights_2d
            
            # Add to total loss
            total_loss = total_loss + self.outlier_loss_weight * weighted_outlier_loss.unsqueeze(2)
        
        # Apply reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss
