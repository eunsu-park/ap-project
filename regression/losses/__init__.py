"""
Loss functions for multimodal solar wind prediction.

This package provides various loss functions for regression and
multimodal alignment, including:
- Contrastive losses (InfoNCE, MSE consistency)
- Regression losses (MSE, MAE, Huber with various weighting strategies)
- Advanced losses (adaptive weighting, gradient-based, quantile, multi-task)

Main components:
- MultiModalContrastiveLoss: InfoNCE-style contrastive loss
- MultiModalMSELoss: MSE-based consistency loss
- WeightedMSELoss: Time-weighted MSE loss
- HuberMultiCriteriaLoss: Huber with temporal and gradient weighting
- MAEOutlierFocusedLoss: MAE with outlier detection
- AdaptiveWeightLoss: Error-adaptive weighting
- GradientBasedWeightLoss: Emphasis on rapid changes
- QuantileLoss: Quantile regression with uncertainty weighting
- MultiTaskLoss: Regression + outlier detection
- create_loss_functions: Factory for regression + contrastive losses
- create_loss: Factory for single loss function

Example usage:
    from loss import create_loss_functions
    
    # Using the factory function (recommended)
    regression_loss, contrastive_loss = create_loss_functions(config, logger)
    
    # Or directly
    from loss import HuberMultiCriteriaLoss, MultiModalContrastiveLoss
    
    reg_loss = HuberMultiCriteriaLoss(beta=0.3)
    cont_loss = MultiModalContrastiveLoss(temperature=0.3)
"""

# Contrastive losses
from .contrastive import MultiModalContrastiveLoss, MultiModalMSELoss

# Regression losses
from .regression import (
    WeightedMSELoss,
    HuberMultiCriteriaLoss,
    MAEOutlierFocusedLoss
)

# Advanced losses
from .advanced import (
    AdaptiveWeightLoss,
    GradientBasedWeightLoss,
    QuantileLoss,
    MultiTaskLoss
)

# Factory functions
from .factory import create_loss_functions, create_loss


__all__ = [
    # Contrastive losses
    'MultiModalContrastiveLoss',
    'MultiModalMSELoss',
    
    # Regression losses
    'WeightedMSELoss',
    'HuberMultiCriteriaLoss',
    'MAEOutlierFocusedLoss',
    
    # Advanced losses
    'AdaptiveWeightLoss',
    'GradientBasedWeightLoss',
    'QuantileLoss',
    'MultiTaskLoss',
    
    # Factory functions (most commonly used)
    'create_loss_functions',
    'create_loss',
]
