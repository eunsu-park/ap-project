"""
Simplified validation script for solar wind prediction.

Uses consolidated config and utilities for clean, concise code.
"""

import os
import hydra
import torch
import torch.nn as nn

from config import Config
from utils import setup_experiment, get_logger
from datasets import create_dataloader
from models import create_model
from validators import Validator


@hydra.main(config_path="./configs", version_base=None)
def main(hydra_cfg):
    """Run validation process."""
    # Convert Hydra config to structured config
    config = Config.from_hydra(hydra_cfg)
    
    # Setup experiment (logger, seed, device)
    output_dir = config.validation.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    setup_experiment(config, log_dir=output_dir)
    logger = get_logger()
    
    # Override phase to validation
    config.experiment.phase = 'validation'
    
    # Create validation dataloader
    validation_dataloader = create_dataloader(config)
    logger.info(
        f"Validation dataloader: {len(validation_dataloader.dataset)} samples, "
        f"{len(validation_dataloader)} batches"
    )
    
    # Create model
    model = create_model(config).to(config.environment.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")
    
    # Load checkpoint
    checkpoint_path = config.validation.checkpoint_path
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=config.environment.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Checkpoint loaded successfully")
    
    # Create loss criterion
    criterion = nn.MSELoss()
    
    # Create validator
    validator = Validator(
        config=config,
        model=model,
        criterion=criterion,
        device=torch.device(config.environment.device),
        logger=logger,
        save_plots=config.validation.save_plots
    )
    
    # Run validation
    results = validator.validate(validation_dataloader)
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETED")
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


if __name__ == '__main__':
    main()
