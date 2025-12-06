"""
Refactored training script for multimodal solar wind prediction.

Major improvements:
- Modular design with Trainer class
- Separated concerns (training, metrics, checkpoints)
- Cleaner main function
- Better error handling
"""

import os
from multiprocessing import freeze_support
import hydra
import torch
import torch.optim as optim

from utils import set_seed, setup_logger, setup_device
from datasets import create_dataloader
from models import create_model
from losses import create_loss_functions
from trainers import Trainer, save_training_history, plot_training_curves


def setup_training_environment(config):
    """Setup logging, seeding, and device.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (logger, device)
    """
    # Create directories
    save_root = config.environment.save_root
    experiment_name = config.experiment.experiment_name
    experiment_dir = f"{save_root}/{experiment_name}"
    
    directories = [
        f"{experiment_dir}/checkpoint",
        f"{experiment_dir}/log",
        f"{experiment_dir}/snapshot",
        f"{experiment_dir}/validation"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(__name__, log_dir=f"{experiment_dir}/log")
    logger.info(f"Training configuration:\n{config}")
    
    # Set seed and device
    set_seed(config.environment.seed, logger=logger)
    device = setup_device(config, logger=logger)
    
    return logger, device


def create_training_components(config, model, logger=None):
    """Create optimizer, scheduler, and loss functions.
    
    Args:
        config: Configuration object.
        model: PyTorch model.
        logger: Optional logger for output.
        
    Returns:
        Tuple of (optimizer, scheduler, criterion, contrastive_criterion)
    """
    # Optimizer
    if config.training.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate
        )
        opt_name = "Adam"
    elif config.training.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9
        )
        opt_name = "SGD"
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate
        )
        opt_name = "Adam (default)"
        if logger:
            logger.warning(f"Unknown optimizer '{config.training.optimizer}', using Adam")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Loss functions
    criterion, contrastive_criterion = create_loss_functions(config, logger)
    
    if logger:
        logger.info(f"Optimizer: {opt_name}, LR: {config.training.learning_rate}")
        logger.info(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    return optimizer, scheduler, criterion, contrastive_criterion


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    """Run training process.
    
    Args:
        config: Configuration object containing training parameters.
        
    Raises:
        RuntimeError: If training setup or execution fails.
    """
    # Setup environment
    logger, device = setup_training_environment(config)
    
    # Create dataloader
    try:
        dataloader = create_dataloader(config, logger=logger)
        logger.info(f"Dataloader created: {len(dataloader.dataset)} samples, "
                   f"{len(dataloader)} batches")
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        raise RuntimeError(f"Failed to create dataloader: {e}")
    
    # Create model
    try:
        model = create_model(config, logger=logger)
        model.to(device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise RuntimeError(f"Failed to create model: {e}")
    
    # Create training components
    try:
        optimizer, scheduler, criterion, contrastive_criterion = create_training_components(
            config, model, logger
        )
    except Exception as e:
        logger.error(f"Failed to create training components: {e}")
        raise RuntimeError(f"Failed to create training components: {e}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        contrastive_criterion=contrastive_criterion,
        device=device,
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