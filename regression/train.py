"""
Simplified training script for solar wind prediction.

Uses consolidated config and utilities for clean, concise code.
"""

import os
from multiprocessing import freeze_support
import hydra
import torch
import torch.optim as optim

from config import Config
from utils import setup_experiment, get_logger
from datasets import create_dataloader
from models import create_model
from losses import create_loss_functions
from trainers import Trainer, save_training_history, plot_training_curves


def create_optimizer(config, model):
    """Create optimizer from config."""
    if config.training.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=0.9)
    else:
        return optim.Adam(model.parameters(), lr=config.training.learning_rate)


def create_scheduler(optimizer):
    """Create learning rate scheduler."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )


@hydra.main(config_path="./configs", version_base=None)
def main(hydra_cfg):
    """Run training process."""
    # Convert Hydra config to structured config
    config = Config.from_hydra(hydra_cfg)
    
    # Setup experiment (logger, seed, device)
    setup_experiment(config, log_dir=config.log_dir)
    logger = get_logger()
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Create dataloader
    dataloader = create_dataloader(config)
    logger.info(f"Dataloader: {len(dataloader.dataset)} samples, {len(dataloader)} batches")
    
    # Create model
    model = create_model(config).to(config.environment.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Create training components
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(optimizer)
    criterion, contrastive_criterion = create_loss_functions(config)
    
    logger.info(f"Optimizer: {config.training.optimizer.upper()}, LR: {config.training.learning_rate}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        contrastive_criterion=contrastive_criterion,
        device=torch.device(config.environment.device),
        logger=logger
    )
    
    # Train
    try:
        history = trainer.fit(dataloader, config.training.num_epochs)
        
        # Save results
        save_training_history(history, config, logger)
        plot_training_curves(history, config, logger)
        
        logger.info("Training completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    freeze_support()
    main()
