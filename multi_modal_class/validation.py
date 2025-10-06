import os
import copy
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from utils import set_seed, setup_logger, setup_device, load_model  # save_plot, calculate_metrics 제거
from config import Config
from pipeline import create_dataloader
from networks import create_model


def validation_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
                   criterion: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """Perform a single validation step for binary classification.
    
    Args:
        model: PyTorch model.
        data_dict: Dictionary containing input data.
        criterion: Loss function.
        device: Device for computation.
        
    Returns:
        Dictionary containing loss, accuracy, and prediction results.
        
    Raises:
        RuntimeError: If validation step fails.
    """
    model.eval()
    try:
        sdo = data_dict["sdo"].to(device)
        inputs = data_dict["inputs"].to(device)
        targets = data_dict["targets"].to(device)

        with torch.no_grad():
            outputs = model(inputs, sdo)  # Logits
            loss = criterion(outputs, targets)
            
            # Convert logits to probabilities and binary predictions
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            # Calculate accuracy
            accuracy = (predictions == targets).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'targets': targets.cpu().numpy(),
            'predictions': predictions.cpu().numpy(),
            'probabilities': probs.cpu().numpy()
        }
    except Exception as e:
        raise RuntimeError(f"Validation step failed: {e}")


def calculate_classification_metrics(all_targets: np.ndarray, 
                                     all_predictions: np.ndarray,
                                     target_variables: list) -> Dict[str, Dict[str, float]]:
    """Calculate classification metrics for each target variable.
    
    Args:
        all_targets: Array of shape (n_samples, n_groups, n_variables)
        all_predictions: Array of shape (n_samples, n_groups, n_variables)
        target_variables: List of target variable names
        
    Returns:
        Dictionary containing metrics for each variable
    """
    n_samples, n_groups, n_variables = all_targets.shape
    metrics_dict = {}
    
    for var_idx, var_name in enumerate(target_variables):
        # Flatten across samples and groups for this variable
        var_targets = all_targets[:, :, var_idx].flatten()
        var_predictions = all_predictions[:, :, var_idx].flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(var_targets, var_predictions)
        
        # Handle cases where there might be only one class
        try:
            precision = precision_score(var_targets, var_predictions, zero_division=0)
            recall = recall_score(var_targets, var_predictions, zero_division=0)
            f1 = f1_score(var_targets, var_predictions, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(var_targets, var_predictions)
        
        metrics_dict[var_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'positive_rate': var_targets.mean(),  # Class balance
            'predicted_positive_rate': var_predictions.mean()
        }
    
    return metrics_dict


def save_validation_results(results: Dict[str, Any], output_path: str, logger=None):
    """Save validation results to a text file.
    
    Args:
        results: Dictionary containing validation results
        output_path: Path to save the results file
        logger: Optional logger
    """
    try:
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VALIDATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall metrics
            f.write(f"Total Samples: {results['total_samples']}\n")
            f.write(f"Failed Batches: {results['failed_batches']}\n")
            f.write(f"Success Rate: {results['success_rate']:.2f}%\n\n")
            
            f.write(f"Average Loss: {results['average_loss']:.6f}\n")
            f.write(f"Loss Std Dev: {results['std_loss']:.6f}\n\n")
            
            f.write(f"Average Accuracy: {results['average_accuracy']:.4f}\n")
            f.write(f"Accuracy Std Dev: {results['std_accuracy']:.4f}\n\n")
            
            # Per-variable metrics
            f.write("=" * 80 + "\n")
            f.write("METRICS BY VARIABLE\n")
            f.write("=" * 80 + "\n\n")
            
            for var_name, metrics in results['metrics_per_variable'].items():
                f.write(f"Variable: {var_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
                f.write(f"  Positive Rate (True):      {metrics['positive_rate']:.4f}\n")
                f.write(f"  Positive Rate (Predicted): {metrics['predicted_positive_rate']:.4f}\n")
                f.write(f"  Confusion Matrix:\n")
                cm = np.array(metrics['confusion_matrix'])
                f.write(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],\n")
                f.write(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Results saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        if logger:
            logger.info(f"Validation results saved to: {output_path}")
        else:
            print(f"Validation results saved to: {output_path}")
            
    except Exception as e:
        error_msg = f"Failed to save validation results: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")


def run_validation(options: Config, checkpoint_path: str, 
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run validation on the dataset for binary classification.
    
    Args:
        options: Configuration object.
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory to save validation results. If None, uses options.validation_dir.
        
    Returns:
        Dictionary containing validation results including average loss and accuracy.
        
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

    # Setup loss function - Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()
    logger.info("Using BCEWithLogitsLoss for binary classification validation")

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"{options.validation_dir}/validation_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Validation results will be saved to: {output_dir}")

    # Initialize tracking variables
    validation_losses = []
    validation_accuracies = []
    all_targets = []
    all_predictions = []
    all_probabilities = []
    failed_batches = 0
    successful_samples = 0

    logger.info("Running validation...")
    
    # Run validation loop
    for i, data_dict in enumerate(validation_dataloader):
        try:
            validation_dict = validation_step(model, data_dict, criterion, device)
            validation_losses.append(validation_dict['loss'])
            validation_accuracies.append(validation_dict['accuracy'])
            
            # Accumulate predictions and targets
            all_targets.append(validation_dict['targets'])
            all_predictions.append(validation_dict['predictions'])
            all_probabilities.append(validation_dict['probabilities'])
            
            successful_samples += validation_dict['targets'].shape[0]
            
            # Log progress periodically
            if (i + 1) % 50 == 0:
                current_avg_loss = np.mean(validation_losses)
                current_avg_acc = np.mean(validation_accuracies)
                logger.info(
                    f"Processed {i + 1}/{len(validation_dataloader)} batches | "
                    f"Avg Loss: {current_avg_loss:.6f} | Avg Accuracy: {current_avg_acc:.4f}"
                )
                
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
    avg_accuracy = np.mean(validation_accuracies)
    std_accuracy = np.std(validation_accuracies)
    
    # Concatenate all predictions and targets
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (n_samples, n_groups, n_vars)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    
    # Calculate per-variable classification metrics
    metrics_per_variable = calculate_classification_metrics(
        all_targets, all_predictions, options.target_variables
    )
    
    # Log individual variable metrics
    logger.info("\n" + "=" * 50)
    logger.info("METRICS BY VARIABLE")
    logger.info("=" * 50)
    for var_name, metrics in metrics_per_variable.items():
        logger.info(f"\n{var_name}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")

    # Compile results
    results = {
        'average_loss': avg_loss,
        'std_loss': std_loss,
        'average_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'metrics_per_variable': metrics_per_variable,
        'total_samples': successful_samples,
        'failed_batches': failed_batches,
        'success_rate': (len(validation_losses) / len(validation_dataloader)) * 100,
        'output_directory': str(output_dir)
    }

    # Save results to text file
    results_file_path = output_dir / "validation_results.txt"
    save_validation_results(results, str(results_file_path), logger)

    # Log summary
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Average Loss:     {avg_loss:.6f} (±{std_loss:.6f})")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    logger.info(f"Total Samples:    {successful_samples}")
    logger.info(f"Failed Batches:   {failed_batches}")
    logger.info(f"Success Rate:     {results['success_rate']:.1f}%")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 50 + "\n")

    return results


def main():
    """Main function for validation script."""
    parser = argparse.ArgumentParser(description='Run model validation')
    parser.add_argument('--config', default='configs/config_dev.yaml', type=str,
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

        print(options)

        options.validate()
        options.make_directories()
        
        # Set random seed
        set_seed(options.seed)

        # Run validation
        results = run_validation(options, args.checkpoint, args.output_dir)
        
        # Print success message with key metrics
        print("=" * 60)
        print("Validation completed successfully!")
        print(f"Average Loss:     {results['average_loss']:.6f}")
        print(f"Average Accuracy: {results['average_accuracy']:.4f}")
        print(f"Success Rate:     {results['success_rate']:.1f}%")
        print("\nMetrics by Variable:")
        for var_name, metrics in results['metrics_per_variable'].items():
            print(f"  {var_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
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