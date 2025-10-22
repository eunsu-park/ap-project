import os
import copy
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import numpy as np
import hydra

from utils import set_seed, setup_logger, setup_device, load_model, save_plot, calculate_metrics
from config import Config
from pipeline import create_dataloader
from networks import create_model


def validation_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
                   criterion: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """Perform a single validation step.
    
    Args:
        model: PyTorch model.
        data_dict: Dictionary containing input data.
        criterion: Loss function.
        device: Device for computation.
        
    Returns:
        Dictionary containing loss and prediction results.
        
    Raises:
        RuntimeError: If validation step fails.
    """
    model.eval()
    try:
        sdo = data_dict["sdo"].to(device)
        inputs = data_dict["inputs"].to(device)
        targets = data_dict["targets"].to(device)

        with torch.no_grad():
            outputs = model(inputs, sdo)
            loss = criterion(outputs, targets)

        return {
            'loss': loss.item(),
            'targets': targets.cpu().numpy(),
            'outputs': outputs.cpu().detach().numpy()
        }
    except Exception as e:
        raise RuntimeError(f"Validation step failed: {e}")


def run_validation(options: Config, checkpoint_path: str, 
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run validation on the dataset.
    
    Args:
        options: Configuration object.
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory to save validation results. If None, uses options.validation_dir.
        
    Returns:
        Dictionary containing validation results including average loss and metrics.
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If validation fails completely.
    """
    # Setup logging
    logger = setup_logger(__name__, log_dir=options.log_dir)
    logger.info("Starting validation...")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Setup device
    device = setup_device(options, logger)

    # Create validation dataloader
    validation_options = copy.deepcopy(options)
    validation_options.phase = 'validation'
    validation_options.batch_size = 1

    try:
        validation_dataloader = create_dataloader(validation_options, logger=logger)
        logger.info(f"Validation dataloader created with {len(validation_dataloader)} batches.")
    except Exception as e:
        raise RuntimeError(f"Failed to create validation dataloader: {e}")

    # Create and load model
    try:
        model = create_model(options, logger=logger)
        model = load_model(model, checkpoint_path, device, logger=logger)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Setup loss function
    criterion = nn.MSELoss()

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"{options.validation_dir}/validation_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Validation results will be saved to: {output_dir}")

    # Initialize tracking variables
    validation_losses = []
    all_targets = []
    all_outputs = []
    failed_batches = 0
    successful_samples = 0

    logger.info("Running validation...")
    
    # Run validation loop
    for i, data_dict in enumerate(validation_dataloader):
        try:
            validation_dict = validation_step(model, data_dict, criterion, device)
            validation_losses.append(validation_dict['loss'])
            
            file_names = data_dict["file_names"]
            
            # Process each sample in the batch
            for j in range(len(file_names)):
                file_name = os.path.splitext(file_names[j])[0]
                
                # Extract individual sample data
                sample_targets = validation_dict['targets'][j]
                sample_outputs = validation_dict['outputs'][j]
                
                # Save individual validation result
                plot_path = f"{output_dir}/{file_name}"
                plot_title = f'Validation - File {file_name}'

                try:
                    save_plot(
                        targets=sample_targets, 
                        outputs=sample_outputs,
                        target_variables=options.target_variables, 
                        stat_dict=validation_dataloader.dataset.stat_dict,
                        plot_path=plot_path, 
                        plot_title=plot_title, 
                        logger=logger
                    )
                except Exception as e:
                    logger.warning(f"Failed to save plot for {file_name}: {e}")
                
                # Accumulate for overall statistics
                all_targets.append(sample_targets)
                all_outputs.append(sample_outputs)
                successful_samples += 1
            
            # Log progress periodically
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(validation_dataloader)} batches")
                
        except Exception as e:
            logger.warning(f"Validation failed for batch {i}: {e}")
            failed_batches += 1
            continue

    # Calculate overall statistics if we have successful validations
    if not validation_losses:
        logger.error("No successful validation steps completed")
        raise RuntimeError("Validation failed completely")

    # Compute validation metrics
    avg_loss = np.mean(validation_losses)
    std_loss = np.std(validation_losses)
    
    # Calculate per-variable metrics using improved utils function
    all_targets = np.array(all_targets)  # Shape: (n_samples, seq_len, n_vars)
    all_outputs = np.array(all_outputs)
    
    # Calculate comprehensive metrics
    detailed_metrics = calculate_metrics(
        all_targets[..., np.newaxis, :].transpose(0, 2, 1), 
        all_outputs[..., np.newaxis, :].transpose(0, 2, 1),
        options.target_variables
    )
    
    # Extract RMSE for backward compatibility
    rmse_per_var = {var: metrics['rmse'] for var, metrics in detailed_metrics.items()}
    
    # Log individual variable metrics
    for var_name, metrics in detailed_metrics.items():
        logger.info(f"Metrics for {var_name}:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  Correlation: {metrics['correlation']:.4f}")
        logger.info(f"  R²: {metrics['r_squared']:.4f}")

    # Create overall validation plot
    try:
        overall_plot_path = f"{output_dir}/overall_validation_results"
        overall_plot_title = 'Overall Validation Results'
        
        # Use mean across all samples for overall plot
        mean_targets = np.mean(all_targets, axis=0)
        mean_outputs = np.mean(all_outputs, axis=0)
        
        save_plot(
            targets=mean_targets, 
            outputs=mean_outputs,
            target_variables=options.target_variables, 
            stat_dict=validation_dataloader.dataset.stat_dict,
            plot_path=overall_plot_path, 
            plot_title=overall_plot_title, 
            logger=logger
        )
    except Exception as e:
        logger.warning(f"Failed to create overall validation plot: {e}")

    # Compile results
    results = {
        'average_loss': avg_loss,
        'std_loss': std_loss,
        'rmse_per_variable': rmse_per_var,
        'detailed_metrics': detailed_metrics,
        'total_samples': successful_samples,
        'failed_batches': failed_batches,
        'success_rate': (len(validation_losses) / len(validation_dataloader)) * 100,
        'output_directory': str(output_dir)
    }

    # Log summary
    logger.info("=" * 50)
    logger.info("Validation Results Summary")
    logger.info(f"Average Loss: {avg_loss:.4f} (±{std_loss:.4f})")  # Fixed encoding issue
    logger.info(f"Total Samples: {successful_samples}")
    logger.info(f"Failed Batches: {failed_batches}")
    logger.info(f"Success Rate: {results['success_rate']:.1f}%")
    
    logger.info("RMSE by Variable:")
    for var_name, rmse in rmse_per_var.items():
        logger.info(f"  {var_name}: {rmse:.4f}")
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 50)

    return results


@hydra.main(config_path="./configs", version_base=None)
def main():
    """Main function for validation script."""
    parser = argparse.ArgumentParser(description='Run model validation')
    parser.add_argument('--config', type=str, default='config_dev.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for validation results')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    
    args = parser.parse_args()

    try:
        # Load configuration
        options = Config().from_args_and_yaml(yaml_path=args.config, args=args)
        print(args.config, options.experiment_name)

        options.validate()
        options.make_directories()
        
        # Set random seed
        set_seed(options.seed)

        # Run validation
        results = run_validation(options, args.checkpoint, args.output_dir)
        
        # Print success message with key metrics
        print("=" * 60)
        print("Validation completed successfully!")
        print(f"Average Loss: {results['average_loss']:.4f}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print("\nRMSE by Variable:")
        for var_name, rmse in results['rmse_per_variable'].items():
            print(f"  {var_name}: {rmse:.4f}")
        print(f"\nDetailed results saved to: {results['output_directory']}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        return 1
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
