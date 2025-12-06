"""
Factory functions for creating loss instances.

This module provides convenience functions to create loss functions
from configuration objects.
"""

import torch.nn as nn

from .contrastive import MultiModalContrastiveLoss, MultiModalMSELoss
from .regression import WeightedMSELoss, HuberMultiCriteriaLoss, MAEOutlierFocusedLoss
from .advanced import AdaptiveWeightLoss, GradientBasedWeightLoss, QuantileLoss, MultiTaskLoss


def create_loss_functions(config, logger=None):
    """Create regression and contrastive loss functions from config.
    
    Args:
        config: Configuration object containing loss parameters.
        logger: Optional logger for output.
        
    Returns:
        Tuple of (regression_criterion, contrastive_criterion)
    """
    # Regression loss
    if config.training.loss_type == 'mse':
        regression_criterion = nn.MSELoss()
        loss_name = "MSE"
    elif config.training.loss_type == 'mae':
        regression_criterion = nn.L1Loss()
        loss_name = "MAE (L1)"
    else:
        regression_criterion = nn.MSELoss()
        loss_name = "MSE (default)"
        if logger:
            logger.warning(f"Unknown loss type '{config.training.loss_type}', using MSE")
    
    # Contrastive loss
    if config.training.contrastive_type == 'infonce':
        contrastive_criterion = MultiModalContrastiveLoss(
            temperature=config.training.contrastive_temperature,
            normalize=True
        )
        cont_name = "InfoNCE"
    elif config.training.contrastive_type == 'mse':
        contrastive_criterion = MultiModalMSELoss(reduction='mean')
        cont_name = "MSE Consistency"
    else:
        contrastive_criterion = MultiModalMSELoss(reduction='mean')
        cont_name = "MSE Consistency (default)"
        if logger:
            logger.warning(f"Unknown contrastive type '{config.training.contrastive_type}', using MSE")
    
    message = f"Loss functions created: Regression={loss_name}, Contrastive={cont_name}"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return regression_criterion, contrastive_criterion


def create_loss(options, logger=None):
    """Create loss function based on configuration.
    
    Args:
        options: Configuration object containing loss_type.
        logger: Optional logger for output.
        
    Returns:
        Loss function instance.
        
    Raises:
        ValueError: If unsupported loss type is specified.
    """
    loss_message = f"Using {options.loss_type.upper()} Loss"
    
    if logger:
        logger.info(loss_message)
    else:
        print(loss_message)
    
    loss_map = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': lambda: nn.HuberLoss(delta=10.0),
        'huber_multi_criteria': HuberMultiCriteriaLoss,
        'mae_outlier_focused': MAEOutlierFocusedLoss,
        'adaptive_weight': AdaptiveWeightLoss,
        'gradient_based_weight': GradientBasedWeightLoss,
        'quantile': QuantileLoss,
        'multi_task': MultiTaskLoss,
        'weighted_mse': WeightedMSELoss
    }
    
    if options.loss_type not in loss_map:
        raise ValueError(f"Unsupported loss type: {options.loss_type}")
    
    return loss_map[options.loss_type]()
