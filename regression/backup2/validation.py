"""
Refactored validation script for multimodal solar wind prediction.

Major improvements:
- Modular design with Validator class
- Separated concerns (validation, metrics, results writing)
- Cleaner main function
- Better error handling
"""

import os
import hydra
import torch
import torch.nn as nn

from utils import set_seed, setup_logger, setup_device, load_model
from datasets import create_dataloader
from models import create_model
from validators import Validator


def setup_validation_environment(config):
    """Setup logging, seeding, and device for validation.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (logger, device)
    """
    # Create output directory
    output_dir = config.validation.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(__name__, log_dir=output_dir)
    logger.info(f"Validation configuration:\n{config}")
    
    # Set seed and device
    set_seed(config.environment.seed, logger=logger)
    device = setup_device(config, logger=logger)
    
    return logger, device


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    """Run validation process.
    
    Args:
        config: Configuration object containing validation parameters.
        
    Raises:
        RuntimeError: If validation setup or execution fails.
    """
    # Setup environment
    logger, device = setup_validation_environment(config)
    
    # Override phase to validation
    original_phase = config.experiment.phase
    config.experiment.phase = 'validation'
    
    # Create validation dataloader
    try:
        validation_dataloader = create_dataloader(config, logger=logger)
        logger.info(f"Validation dataloader created: {len(validation_dataloader.dataset)} samples, "
                   f"{len(validation_dataloader)} batches")
    except Exception as e:
        logger.error(f"Failed to create validation dataloader: {e}")
        raise RuntimeError(f"Failed to create validation dataloader: {e}")
    
    # Restore original phase
    config.experiment.phase = original_phase
    
    # Create model
    try:
        model = create_model(config, logger=logger)
        model.to(device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,} total")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise RuntimeError(f"Failed to create model: {e}")
    
    # Load checkpoint
    checkpoint_path = config.validation.checkpoint_path
    try:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        model = load_model(model, checkpoint_path, device, logger)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Create loss criterion
    criterion = nn.MSELoss()
    
    # Create validator
    save_plots = config.validation.get('save_plots', True)
    validator = Validator(
        config=config,
        model=model,
        criterion=criterion,
        device=device,
        logger=logger,
        save_plots=save_plots
    )
    
    # Run validation
    try:
        results = validator.validate(validation_dataloader)
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Average Loss: {results['overall']['average_loss']:.6f}")
        print(f"Average MAE:  {results['overall']['average_mae']:.4f}")
        print(f"Average RMSE: {results['overall']['average_rmse']:.4f}")
        print(f"Average R²:   {results['overall']['average_r2']:.4f}")
        
        if results['overall']['average_cosine_sim'] is not None:
            print(f"Average Cosine Similarity: {results['overall']['average_cosine_sim']:.4f}")
        
        print(f"\nSuccess Rate: {results['success_rate']:.1f}%")
        print(f"Results saved to: {results['output_directory']}")
        print("=" * 80 + "\n")
        
        logger.info("Validation completed successfully")
        
        return results
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == '__main__':
    main()