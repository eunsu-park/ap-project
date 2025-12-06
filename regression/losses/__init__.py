"""
Loss functions package for solar wind prediction.
"""

import torch.nn as nn

from .losses import (
    WeightedMSELoss,
    HuberMultiCriteriaLoss,
    MAEOutlierFocusedLoss,
    MultiModalContrastiveLoss,
    MultiModalMSELoss,
    AdaptiveWeightLoss,
    GradientBasedWeightLoss,
    QuantileLoss,
    MultiTaskLoss
)

from utils import get_logger


def create_loss_functions(config):
    """Create regression and contrastive loss functions from config."""
    logger = get_logger()
    
    loss_type = config.training.loss_type.lower()
    
    if loss_type == 'mse':
        regression_criterion = nn.MSELoss()
        loss_name = "MSE"
    elif loss_type == 'mae':
        regression_criterion = nn.L1Loss()
        loss_name = "MAE"
    elif loss_type == 'huber':
        regression_criterion = nn.HuberLoss(delta=10.0)
        loss_name = "Huber"
    else:
        regression_criterion = nn.MSELoss()
        loss_name = "MSE (default)"
        logger.warning(f"Unknown loss type '{loss_type}', using MSE")
    
    contrastive_type = config.training.contrastive_type.lower()
    
    if contrastive_type == 'infonce':
        contrastive_criterion = MultiModalContrastiveLoss(
            temperature=config.training.contrastive_temperature,
            normalize=True
        )
        cont_name = "InfoNCE"
    else:
        contrastive_criterion = MultiModalMSELoss(reduction='mean')
        cont_name = "MSE Consistency"
        if contrastive_type != 'mse':
            logger.warning(f"Unknown contrastive type '{contrastive_type}', using MSE")
    
    logger.info(f"Losses: Regression={loss_name}, Contrastive={cont_name}")
    
    return regression_criterion, contrastive_criterion


def create_loss(config):
    """Create a single regression loss function."""
    logger = get_logger()
    loss_type = config.training.loss_type.lower()
    
    loss_map = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': lambda: nn.HuberLoss(delta=10.0),
        'huber_multi': HuberMultiCriteriaLoss,
        'mae_outlier': MAEOutlierFocusedLoss,
        'adaptive': AdaptiveWeightLoss,
        'gradient': GradientBasedWeightLoss,
        'quantile': QuantileLoss,
        'multitask': MultiTaskLoss,
        'weighted_mse': WeightedMSELoss
    }
    
    if loss_type not in loss_map:
        logger.warning(f"Unknown loss type '{loss_type}', using MSE")
        return nn.MSELoss()
    
    logger.info(f"Using {loss_type.upper()} loss")
    return loss_map[loss_type]()


__all__ = [
    'create_loss_functions',
    'create_loss',
    'WeightedMSELoss',
    'HuberMultiCriteriaLoss',
    'MAEOutlierFocusedLoss',
    'MultiModalContrastiveLoss',
    'MultiModalMSELoss',
    'AdaptiveWeightLoss',
    'GradientBasedWeightLoss',
    'QuantileLoss',
    'MultiTaskLoss',
]