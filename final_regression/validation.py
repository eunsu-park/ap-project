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
from pipeline import create_dataloader
from networks import create_model
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
    validator = Validator(
        config=config,
        model=model,
        criterion=criterion,
        device=device,
        logger=logger
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

# import os
# import copy
# import csv
# import argparse
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, Optional, Any, Tuple

# import hydra
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# from utils import set_seed, setup_logger, setup_device, load_model
# from pipeline import create_dataloader
# from networks import create_model


# def calculate_regression_metrics(all_targets: np.ndarray, 
#                                  all_predictions: np.ndarray,
#                                  target_variables: list) -> Dict[str, Dict[str, float]]:
#     """Calculate regression metrics for each target variable.
    
#     Args:
#         all_targets: Array of shape (n_samples, n_groups, n_variables)
#         all_predictions: Array of shape (n_samples, n_groups, n_variables)
#         target_variables: List of target variable names
        
#     Returns:
#         Dictionary containing regression metrics for each variable
#     """
#     n_samples, n_groups, n_variables = all_targets.shape
#     metrics_dict = {}
    
#     for var_idx, var_name in enumerate(target_variables):
#         # Flatten across samples and groups for this variable
#         var_targets = all_targets[:, :, var_idx].flatten()
#         var_predictions = all_predictions[:, :, var_idx].flatten()
        
#         # Calculate regression metrics
#         mae = np.mean(np.abs(var_targets - var_predictions))
#         mse = np.mean((var_targets - var_predictions) ** 2)
#         rmse = np.sqrt(mse)
        
#         # R² Score (coefficient of determination)
#         ss_res = np.sum((var_targets - var_predictions) ** 2)
#         ss_tot = np.sum((var_targets - var_targets.mean()) ** 2)
#         r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
#         # Additional metrics
#         max_error = np.max(np.abs(var_targets - var_predictions))
#         median_ae = np.median(np.abs(var_targets - var_predictions))
        
#         # Mean Absolute Percentage Error (MAPE)
#         # Add small epsilon to avoid division by zero
#         mape = np.mean(np.abs((var_targets - var_predictions) / (np.abs(var_targets) + 1e-8))) * 100
        
#         # Bias (mean error)
#         bias = np.mean(var_predictions - var_targets)
        
#         metrics_dict[var_name] = {
#             'mae': mae,
#             'mse': mse,
#             'rmse': rmse,
#             'r2_score': r2,
#             'max_error': max_error,
#             'median_absolute_error': median_ae,
#             'mape': mape,
#             'bias': bias,
#             'mean_target': var_targets.mean(),
#             'std_target': var_targets.std(),
#             'mean_prediction': var_predictions.mean(),
#             'std_prediction': var_predictions.std(),
#             'min_target': var_targets.min(),
#             'max_target': var_targets.max(),
#             'min_prediction': var_predictions.min(),
#             'max_prediction': var_predictions.max()
#         }
    
#     return metrics_dict


# def validation_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
#                    criterion: torch.nn.Module, device: torch.device,
#                    compute_alignment: bool = False) -> Dict[str, Any]:
#     """Perform a single validation step for regression.
    
#     Args:
#         model: PyTorch model.
#         data_dict: Dictionary containing input data.
#         criterion: Regression loss function (MSE or MAE).
#         device: Device for computation.
#         compute_alignment: If True, compute and return feature alignment metrics.
        
#     Returns:
#         Dictionary containing loss, MAE, RMSE, R², and prediction results.
#         If compute_alignment=True, also includes cosine_sim.
        
#     Raises:
#         RuntimeError: If validation step fails.
#     """
#     model.eval()
#     try:
#         sdo = data_dict["sdo"].to(device)
#         inputs = data_dict["inputs"].to(device)
#         targets = data_dict["targets"].to(device)

#         with torch.no_grad():
#             # Forward pass
#             if compute_alignment:
#                 # Get features for alignment computation
#                 outputs, transformer_features, convlstm_features = model(
#                     inputs, sdo, return_features=True
#                 )
#                 # Calculate feature alignment
#                 cosine_sim = F.cosine_similarity(
#                     transformer_features, convlstm_features, dim=1
#                 ).mean().item()
#             else:
#                 # Standard forward pass without features
#                 outputs = model(inputs, sdo)
#                 cosine_sim = None
            
#             loss = criterion(outputs, targets)
            
#             # Calculate regression metrics: MAE, RMSE, R²
#             mae = F.l1_loss(outputs, targets).item()
#             mse = F.mse_loss(outputs, targets).item()
#             rmse = torch.sqrt(torch.tensor(mse)).item()
            
#             # Calculate R² (coefficient of determination)
#             ss_res = torch.sum((targets - outputs) ** 2).item()
#             ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
#             r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

#         result = {
#             'loss': loss.item(),
#             'mae': mae,
#             'rmse': rmse,
#             'r2_score': r2_score,
#             'targets': targets.cpu().numpy(),
#             'predictions': outputs.cpu().numpy()  # Continuous values
#         }
        
#         # Add cosine similarity if computed
#         if cosine_sim is not None:
#             result['cosine_sim'] = cosine_sim
            
#         return result
        
#     except Exception as e:
#         raise RuntimeError(f"Validation step failed: {e}")


# def save_results_to_csv(all_file_results: list,
#                        output_path: str,
#                        target_variables: list,
#                        logger=None):
#     """Save all validation results to CSV file for regression.

#     Args:
#         all_file_results: List of dictionaries with file_name, targets, predictions
#         output_path: Path to save the CSV file
#         target_variables: List of target variable names
#         logger: Optional logger
#     """
#     try:
#         with open(output_path, 'w', newline='') as csvfile:
#             fieldnames = ['file_name', 'target', 'prediction', 'error', 'absolute_error', 'squared_error']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
#             writer.writeheader()
            
#             # Write each result
#             for result in all_file_results:
#                 file_name = result['file_name']
#                 targets = result['targets']  # Shape: (n_groups, n_variables)
#                 predictions = result['predictions']  # Shape: (n_groups, n_variables)
                
#                 n_groups, n_variables = targets.shape
                
#                 # For each variable
#                 for var_idx, var_name in enumerate(target_variables):
#                     # For each group
#                     for group_idx in range(n_groups):
#                         target_val = float(targets[group_idx, var_idx])
#                         pred_val = float(predictions[group_idx, var_idx])
#                         error = pred_val - target_val
#                         abs_error = abs(error)
#                         sq_error = error ** 2
                        
#                         # Create a descriptive identifier
#                         full_identifier = f"{file_name}_group{group_idx}_{var_name}"
                        
#                         writer.writerow({
#                             'file_name': full_identifier,
#                             'target': target_val,
#                             'prediction': pred_val,
#                             'error': error,
#                             'absolute_error': abs_error,
#                             'squared_error': sq_error
#                         })
        
#         if logger:
#             logger.info(f"Validation results CSV saved to: {output_path}")
#         else:
#             print(f"Validation results CSV saved to: {output_path}")
            
#     except Exception as e:
#         error_msg = f"Failed to save validation results CSV: {e}"
#         if logger:
#             logger.error(error_msg)
#         else:
#             print(f"Error: {error_msg}")


# def save_validation_results(results: Dict[str, Any], output_path: str, logger=None):
#     """Save validation results to a text file for regression.
    
#     Args:
#         results: Dictionary containing validation results
#         output_path: Path to save the results file
#         logger: Optional logger
#     """
#     try:
#         with open(output_path, 'w') as f:
#             f.write("=" * 80 + "\n")
#             f.write("VALIDATION RESULTS (REGRESSION)\n")
#             f.write("=" * 80 + "\n\n")
            
#             # Overall metrics
#             f.write(f"Total Samples: {results['total_samples']}\n")
#             f.write(f"Failed Batches: {results['failed_batches']}\n")
#             f.write(f"Success Rate: {results['success_rate']:.2f}%\n\n")
            
#             f.write(f"Average Loss: {results['average_loss']:.6f}\n")
#             f.write(f"Loss Std Dev: {results['std_loss']:.6f}\n\n")
            
#             f.write(f"Average MAE: {results['average_mae']:.4f}\n")
#             f.write(f"MAE Std Dev: {results['std_mae']:.4f}\n\n")
            
#             f.write(f"Average RMSE: {results['average_rmse']:.4f}\n")
#             f.write(f"RMSE Std Dev: {results['std_rmse']:.4f}\n\n")
            
#             f.write(f"Average R²: {results['average_r2']:.4f}\n")
#             f.write(f"R² Std Dev: {results['std_r2']:.4f}\n\n")
            
#             # Feature alignment if available
#             if 'average_cosine_sim' in results and results['average_cosine_sim'] is not None:
#                 f.write(f"Average Cosine Similarity: {results['average_cosine_sim']:.4f}\n")
#                 f.write(f"Cosine Similarity Std Dev: {results['std_cosine_sim']:.4f}\n\n")
            
#             # Per-variable metrics
#             f.write("=" * 80 + "\n")
#             f.write("METRICS BY VARIABLE\n")
#             f.write("=" * 80 + "\n\n")
            
#             for var_name, metrics in results['metrics_per_variable'].items():
#                 f.write(f"Variable: {var_name}\n")
#                 f.write("=" * 80 + "\n")
                
#                 # Regression Metrics
#                 f.write("Regression Metrics:\n")
#                 f.write("-" * 40 + "\n")
#                 f.write(f"  MAE (Mean Absolute Error):       {metrics['mae']:.4f}\n")
#                 f.write(f"  MSE (Mean Squared Error):        {metrics['mse']:.4f}\n")
#                 f.write(f"  RMSE (Root Mean Squared Error):  {metrics['rmse']:.4f}\n")
#                 f.write(f"  R² Score:                        {metrics['r2_score']:.4f}\n")
#                 f.write(f"  Max Error:                       {metrics['max_error']:.4f}\n")
#                 f.write(f"  Median Absolute Error:           {metrics['median_absolute_error']:.4f}\n")
#                 f.write(f"  MAPE:                            {metrics['mape']:.2f}%\n")
#                 f.write(f"  Bias (Mean Error):               {metrics['bias']:.4f}\n\n")
                
#                 # Distribution Statistics
#                 f.write("Distribution Statistics:\n")
#                 f.write("-" * 40 + "\n")
#                 f.write(f"  Target Mean:       {metrics['mean_target']:.4f}\n")
#                 f.write(f"  Target Std Dev:    {metrics['std_target']:.4f}\n")
#                 f.write(f"  Target Min:        {metrics['min_target']:.4f}\n")
#                 f.write(f"  Target Max:        {metrics['max_target']:.4f}\n\n")
#                 f.write(f"  Prediction Mean:   {metrics['mean_prediction']:.4f}\n")
#                 f.write(f"  Prediction Std:    {metrics['std_prediction']:.4f}\n")
#                 f.write(f"  Prediction Min:    {metrics['min_prediction']:.4f}\n")
#                 f.write(f"  Prediction Max:    {metrics['max_prediction']:.4f}\n")
#                 f.write("\n" + "=" * 80 + "\n\n")
            
#             f.write("=" * 80 + "\n")
#             f.write(f"Results saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write("=" * 80 + "\n")
        
#         if logger:
#             logger.info(f"Validation results saved to: {output_path}")
#         else:
#             print(f"Validation results saved to: {output_path}")
            
#     except Exception as e:
#         error_msg = f"Failed to save validation results: {e}"
#         if logger:
#             logger.error(error_msg)
#         else:
#             print(f"Error: {error_msg}")


# @hydra.main(config_path="./configs", version_base=None)
# def main(config) -> Dict[str, Any]:
#     """Run validation on the dataset for regression.
    
#     Args:
#         config: Configuration object.
        
#     Returns:
#         Dictionary containing validation results including average loss, MAE, RMSE, and R².
        
#     Raises:
#         FileNotFoundError: If checkpoint file doesn't exist.
#         RuntimeError: If validation fails completely.
#     """
#     # Setup logging

#     save_root = config.environment.save_root
#     experiment_name = config.experiment.experiment_name
#     experiment_dir = f"{save_root}/{experiment_name}"
#     checkpoint_dir = f"{experiment_dir}/checkpoint"
#     log_dir = f"{experiment_dir}/log"
#     snapshot_dir = f"{experiment_dir}/snapshot"
#     validation_dir = f"{experiment_dir}/validation"
#     for directory in [experiment_dir, checkpoint_dir, log_dir, snapshot_dir, validation_dir]:
#         if not os.path.exists(directory):
#             os.makedirs(directory)

#     checkpoint_path = config.validation.checkpoint_path
#     output_dir = config.validation.output_dir

#     logger = setup_logger(__name__, log_dir=log_dir)
#     logger.info("Starting validation...")
#     logger.info(f"Checkpoint: {checkpoint_path}")

#     # Setup device
#     device = setup_device(config, logger)

#     # Create validation dataloader
#     validation_config = copy.deepcopy(config)
#     validation_config.experiment.phase = 'validation'
#     validation_config.experiment.batch_size = 1

#     try:
#         validation_dataloader = create_dataloader(validation_config, logger=logger)
#         logger.info(f"Validation dataloader created with {len(validation_dataloader)} batches.")
#     except Exception as e:
#         raise RuntimeError(f"Failed to create validation dataloader: {e}")

#     # Create and load model
#     try:
#         model = create_model(config, logger=logger)
#         model = load_model(model, checkpoint_path, device, logger=logger)
#     except Exception as e:
#         raise RuntimeError(f"Failed to load model: {e}")

#     # Setup loss function - Regression
#     loss_type = config.training.get('loss_type', 'mse').lower()
    
#     if loss_type == 'mae' or loss_type == 'l1':
#         criterion = nn.L1Loss()
#         logger.info("Using L1Loss (MAE) for regression validation")
#     elif loss_type == 'huber':
#         criterion = nn.HuberLoss(delta=10.0)
#         logger.info("Using HuberLoss for regression validation")
#     else:
#         criterion = nn.MSELoss()
#         logger.info("Using MSELoss for regression validation")

#     # Setup output directory
#     if output_dir is None:
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         output_dir = f"{validation_dir}/validation_{timestamp}"
    
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     logger.info(f"Validation results will be saved to: {output_dir}")

#     # Initialize tracking variables
#     validation_losses = []
#     validation_maes = []
#     validation_rmses = []
#     validation_r2_scores = []
#     validation_cosine_sims = []  # Track feature alignment
#     all_targets = []
#     all_predictions = []
#     all_file_results = []  # For CSV export
#     failed_batches = 0
#     successful_samples = 0
    
#     # Check if we should compute alignment (optional, can be controlled by config)
#     compute_alignment = config.validation.get('compute_alignment', True)
#     if compute_alignment:
#         logger.info("Feature alignment computation enabled during validation")

#     logger.info("Running validation...")
    
#     # Run validation loop
#     for i, data_dict in enumerate(validation_dataloader):
#         try:
#             validation_dict = validation_step(
#                 model, data_dict, criterion, device, 
#                 compute_alignment=compute_alignment
#             )
#             validation_losses.append(validation_dict['loss'])
#             validation_maes.append(validation_dict['mae'])
#             validation_rmses.append(validation_dict['rmse'])
#             validation_r2_scores.append(validation_dict['r2_score'])
            
#             # Track cosine similarity if computed
#             if 'cosine_sim' in validation_dict:
#                 validation_cosine_sims.append(validation_dict['cosine_sim'])
            
#             # Accumulate predictions and targets
#             all_targets.append(validation_dict['targets'])
#             all_predictions.append(validation_dict['predictions'])
            
#             # Store file-level results for CSV export
#             targets = validation_dict['targets'][0]  # Remove batch dimension
#             predictions = validation_dict['predictions'][0]
            
#             # Get file_name
#             file_name = f"batch_{i}"
#             if 'file_names' in data_dict:
#                 file_names = data_dict['file_names']
#                 if isinstance(file_names, torch.Tensor):
#                     file_name = str(file_names.tolist()[0]) if len(file_names.tolist()) > 0 else file_name
#                 elif isinstance(file_names, list):
#                     file_name = str(file_names[0]) if len(file_names) > 0 else file_name
#                 else:
#                     file_name = str(file_names)
            
#             all_file_results.append({
#                 'file_name': file_name,
#                 'targets': targets,
#                 'predictions': predictions
#             })
            
#             successful_samples += validation_dict['targets'].shape[0]
            
#             # Log progress periodically
#             if (i + 1) % 50 == 0:
#                 current_avg_loss = np.mean(validation_losses)
#                 current_avg_mae = np.mean(validation_maes)
#                 current_avg_rmse = np.mean(validation_rmses)
#                 current_avg_r2 = np.mean(validation_r2_scores)
#                 log_msg = (
#                     f"Processed {i + 1}/{len(validation_dataloader)} batches | "
#                     f"Avg Loss: {current_avg_loss:.6f} | "
#                     f"Avg MAE: {current_avg_mae:.4f} | "
#                     f"Avg RMSE: {current_avg_rmse:.4f} | "
#                     f"Avg R²: {current_avg_r2:.4f}"
#                 )
#                 if validation_cosine_sims:
#                     current_avg_cosine = np.mean(validation_cosine_sims)
#                     log_msg += f" | Avg Cosine Sim: {current_avg_cosine:.4f}"
#                 logger.info(log_msg)
                
#         except Exception as e:
#             logger.warning(f"Validation failed for batch {i}: {e}")
#             failed_batches += 1
#             continue

#     # Calculate overall statistics if we have successful validations
#     if not validation_losses:
#         logger.error("No successful validation steps completed")
#         raise RuntimeError("Validation failed completely")

#     # Compute validation metrics
#     avg_loss = np.mean(validation_losses)
#     std_loss = np.std(validation_losses)
#     avg_mae = np.mean(validation_maes)
#     std_mae = np.std(validation_maes)
#     avg_rmse = np.mean(validation_rmses)
#     std_rmse = np.std(validation_rmses)
#     avg_r2 = np.mean(validation_r2_scores)
#     std_r2 = np.std(validation_r2_scores)
    
#     # Compute cosine similarity metrics if available
#     if validation_cosine_sims:
#         avg_cosine_sim = np.mean(validation_cosine_sims)
#         std_cosine_sim = np.std(validation_cosine_sims)
#     else:
#         avg_cosine_sim = None
#         std_cosine_sim = None
    
#     # Concatenate all predictions and targets
#     all_targets = np.concatenate(all_targets, axis=0)  # Shape: (n_samples, n_groups, n_vars)
#     all_predictions = np.concatenate(all_predictions, axis=0)
    
#     # Calculate per-variable regression metrics
#     metrics_per_variable = calculate_regression_metrics(
#         all_targets, all_predictions, config.data.target_variables
#     )
    
#     # Log individual variable metrics
#     logger.info("\n" + "=" * 80)
#     logger.info("REGRESSION METRICS BY VARIABLE")
#     logger.info("=" * 80)
#     for var_name, metrics in metrics_per_variable.items():
#         logger.info(f"\n{var_name}:")
#         logger.info(f"  MAE:   {metrics['mae']:.4f}")
#         logger.info(f"  RMSE:  {metrics['rmse']:.4f}")
#         logger.info(f"  R²:    {metrics['r2_score']:.4f}")
#         logger.info(f"  Max Error: {metrics['max_error']:.4f}")
#         logger.info(f"  Median AE: {metrics['median_absolute_error']:.4f}")
#         logger.info(f"  MAPE:  {metrics['mape']:.2f}%")
#         logger.info(f"  Bias:  {metrics['bias']:.4f}")

#     # Compile results
#     results = {
#         'average_loss': avg_loss,
#         'std_loss': std_loss,
#         'average_mae': avg_mae,
#         'std_mae': std_mae,
#         'average_rmse': avg_rmse,
#         'std_rmse': std_rmse,
#         'average_r2': avg_r2,
#         'std_r2': std_r2,
#         'average_cosine_sim': avg_cosine_sim,
#         'std_cosine_sim': std_cosine_sim,
#         'metrics_per_variable': metrics_per_variable,
#         'total_samples': successful_samples,
#         'failed_batches': failed_batches,
#         'success_rate': (len(validation_losses) / len(validation_dataloader)) * 100,
#         'output_directory': str(output_dir)
#     }

#     # Save results to text file
#     results_file_path = output_dir / "validation_results.txt"
#     save_validation_results(results, str(results_file_path), logger)
    
#     # NEW: Save results to CSV file
#     csv_file_path = output_dir / "validation_results.csv"
#     save_results_to_csv(all_file_results, str(csv_file_path), config.data.target_variables, logger)

#     print("=" * 60)
#     print("Validation completed successfully!")
#     print(f"Average Loss: {results['average_loss']:.6f}")
#     print(f"Average MAE:  {results['average_mae']:.4f}")
#     print(f"Average RMSE: {results['average_rmse']:.4f}")
#     print(f"Average R²:   {results['average_r2']:.4f}")
#     if results['average_cosine_sim'] is not None:
#         print(f"Average Cosine Similarity: {results['average_cosine_sim']:.4f}")
#     print(f"Success Rate: {results['success_rate']:.1f}%")
#     print("\nKey Metrics by Variable:")
#     for var_name, metrics in results['metrics_per_variable'].items():
#         print(f"  {var_name}:")
#         print(f"    MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
#                 f"R²: {metrics['r2_score']:.4f}")
#     print(f"\nDetailed results saved to: {results['output_directory']}")
#     print(f"CSV results saved to: {csv_file_path}")
#     print("=" * 60)

#     # Log summary
#     logger.info("\n" + "=" * 80)
#     logger.info("VALIDATION RESULTS SUMMARY (REGRESSION)")
#     logger.info("=" * 80)
#     logger.info(f"Average Loss: {avg_loss:.6f} (±{std_loss:.6f})")
#     logger.info(f"Average MAE:  {avg_mae:.4f} (±{std_mae:.4f})")
#     logger.info(f"Average RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})")
#     logger.info(f"Average R²:   {avg_r2:.4f} (±{std_r2:.4f})")
#     if avg_cosine_sim is not None:
#         logger.info(f"Average Cosine Similarity: {avg_cosine_sim:.4f} (±{std_cosine_sim:.6f})")
#     logger.info(f"Total Samples:  {successful_samples}")
#     logger.info(f"Failed Batches: {failed_batches}")
#     logger.info(f"Success Rate:   {results['success_rate']:.1f}%")
#     logger.info(f"Results saved to: {output_dir}")
#     logger.info(f"CSV saved to: {csv_file_path}")
#     logger.info("=" * 80 + "\n")

#     return results


# if __name__ == '__main__':
#     main()
