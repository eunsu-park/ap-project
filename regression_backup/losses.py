# python standard library
from typing import Tuple

# third-party library
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

# custom library


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
"""
Contrastive loss functions for multimodal alignment.

This module provides loss functions for aligning features from different
modalities (transformer and ConvLSTM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalContrastiveLoss(nn.Module):
    """Contrastive loss for aligning multimodal representations.
    
    Uses InfoNCE-style loss to align transformer (solar wind) and ConvLSTM (image)
    features in the same embedding space. Features from the same sample are treated
    as positive pairs, while features from different samples in the batch serve as
    negative pairs (in-batch negatives).
    
    Args:
        temperature: Temperature parameter for scaling similarity scores.
                    Higher values (0.3-0.5) recommended for small batch sizes.
                    Lower values (0.07-0.1) for large batch sizes.
        normalize: Whether to L2-normalize features before computing similarity.
                  True is strongly recommended for stable training.
    """
    
    def __init__(self, temperature: float = 0.3, normalize: bool = True):
        super().__init__()
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two sets of features.
        
        Args:
            features1: Features from first modality of shape (batch_size, feature_dim).
            features2: Features from second modality of shape (batch_size, feature_dim).
            
        Returns:
            Contrastive loss scalar.
            
        Raises:
            ValueError: If input tensors have mismatched dimensions.
        """
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")
        
        if features1.size(0) != features2.size(0):
            raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")
        
        if features1.size(1) != features2.size(1):
            raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")
        
        batch_size = features1.size(0)
        
        # L2 normalization for cosine similarity
        if self.normalize:
            features1 = F.normalize(features1, p=2, dim=1)
            features2 = F.normalize(features2, p=2, dim=1)
        
        # Concatenate features from both modalities
        # Shape: (2 * batch_size, feature_dim)
        features = torch.cat([features1, features2], dim=0)
        
        # Compute similarity matrix
        # Shape: (2 * batch_size, 2 * batch_size)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for positive pairs
        # For each sample i in features1, its positive pair is sample i in features2
        # which is at index (i + batch_size) in the concatenated tensor
        labels = torch.arange(batch_size, device=features1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Create mask to exclude self-similarity (diagonal elements)
        # We don't want to compare a sample with itself
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Compute cross-entropy loss
        # This encourages high similarity with positive pairs and low similarity with negatives
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class MultiModalMSELoss(nn.Module):
    """MSE-based consistency loss for multimodal alignment.
    
    Directly minimizes the Euclidean distance between features from two modalities.
    This approach encourages the same sample's features from different modalities
    to have similar representations in the embedding space.
    
    Unlike contrastive learning approaches (InfoNCE), MSE loss does not use negative
    samples, making it more suitable for small batch sizes where negative samples
    would be limited.
    
    Args:
        reduction: Specifies the reduction to apply to the output.
                  'mean' (default): average of all elements
                  'sum': sum of all elements
                  'none': no reduction
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'")
        
        self.reduction = reduction
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between two sets of features.
        
        Args:
            features1: Features from first modality of shape (batch_size, feature_dim).
            features2: Features from second modality of shape (batch_size, feature_dim).
            
        Returns:
            MSE loss scalar (or tensor if reduction='none').
            
        Raises:
            ValueError: If input tensors have mismatched dimensions.
        """
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")
        
        if features1.size(0) != features2.size(0):
            raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")
        
        if features1.size(1) != features2.size(1):
            raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")
        
        # Compute MSE loss - directly aligns features from both modalities
        loss = F.mse_loss(features1, features2, reduction=self.reduction)
        
        return loss
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


def create_loss_functions(config):
    regression_loss_type = config.training.regression_loss_type.lower()

    if regression_loss_type == "mse" :
        regression_criterion = nn.MSELoss()
        regression_loss_name = "MSE"
    elif regression_loss_type == "mae" :
        regression_criterion = nn.L1Loss()
        regression_loss_name = "MAE"
    elif regression_loss_type == "huber" :
        regression_criterion = nn.HuberLoss(delta=10.0)
        regression_loss_name = "Huber"
    else :
        regression_criterion = nn.MSELoss()
        regression_loss_name = "MSE"
        print(f"Unknown regression loss type '{regression_loss_type}', using MSE")

    contrastive_loss_type = config.training.contrastive_loss_type.lower()
    if contrastive_loss_type == 'infonce':
        contrastive_criterion = MultiModalContrastiveLoss(
            temperature=config.training.contrastive_temperature,
            normalize=True
        )
        contrastive_loss_name = "InfoNCE"
    elif contrastive_loss_type == 'consistency':
        contrastive_criterion = MultiModalMSELoss(reduction='mean')
        contrastive_loss_name = "Consistency"
    else :
        contrastive_criterion = MultiModalMSELoss(reduction='mean')
        print(f"Unknown contrastive loss type '{contrastive_loss_type}', using consistency")
    
    print(f"Losses: Regression={regression_loss_name}, Contrastive={contrastive_loss_name}")

    return regression_criterion, contrastive_criterion


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    from networks import create_model
    model = create_model(config)
    print(model)

    regression_criterion, contrastive_criterion = create_loss_functions(config)
    print(regression_criterion)
    print(contrastive_criterion)

    sdo_shape = (
        config.experiment.batch_size,
        len(config.data.sdo_wavelengths),
        config.data.sdo_end_index - config.data.sdo_start_index,
        config.data.sdo_image_size,
        config.data.sdo_image_size
    )

    inputs_shape = (
        config.experiment.batch_size,
        config.data.input_end_index - config.data.input_start_index,
        len(config.data.input_variables)
    )

    targets_shape = (
        config.experiment.batch_size,
        config.data.target_end_index - config.data.target_start_index,
        len(config.data.target_variables)
    )

    sdo = torch.randn(size=sdo_shape)
    print(f"sdo: {sdo.shape}")

    inputs = torch.randn(size=inputs_shape)
    print(f"inputs: {inputs.shape}")

    targets = torch.randn(size=targets_shape)
    print(f"targets: {targets.shape}")

    outputs, transformer_features, convlstm_features = model(inputs, sdo, return_features=True)
    print(f"outputs: {outputs.shape}")

    regression_loss = regression_criterion(outputs, targets)
    print(f"regression loss: {regression_loss}")

    contrastive_loss = contrastive_criterion(transformer_features, convlstm_features)
    print(f"contrastive loss: {contrastive_loss}")


if __name__ == "__main__" :
    main()
