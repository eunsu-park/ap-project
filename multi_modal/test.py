import os
import copy
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np

from utils import set_seed, setup_logger, setup_device, load_model, save_plot, calculate_metrics
from config import Config
from pipeline import create_dataloader
from networks import create_model


def prediction_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
                   device: torch.device) -> Dict[str, Any]:
    """Perform a single prediction step.
    
    Args:
        model: PyTorch model.
        data_dict: Dictionary containing input data.
        device: Device for computation.
        
    Returns:
        Dictionary containing prediction results.
        
    Raises:
        RuntimeError: If prediction step fails.
    """
    model.eval()
    try:
        sdo = data_dict["sdo"].to(device)
        inputs = data_dict["inputs"].to(device)

        with torch.no_grad():
            outputs = model(inputs, sdo)

        return {
            'outputs': outputs.cpu().detach().numpy()
        }
    except Exception as e:
        raise RuntimeError(f"Prediction step failed: {e}")


def create_test_dataloader(options: Config, test_list_path: Optional[str] = None) -> torch.utils.data.DataLoader:
    """Create test dataloader with appropriate configuration.
    
    Args:
        options: Configuration object.
        test_list_path: Path to test data list. If None, uses validation data.
        
    Returns:
        DataLoader: Test data loader.
        
    Raises:
        RuntimeError: If dataloader creation fails.
    """
    test_options = copy.deepcopy(options)
    
    # Configure for test phase
    if test_list_path and os.path.exists(test_list_path):
        test_options.validation_list_path = test_list_path
    
    test_options.phase = 'validation'  # Use validation phase for test data loading
    test_options.batch_size = 1  # Process one sample at a time for testing
    
    try:
        return create_dataloader(test_options)
    except Exception as e:
        raise RuntimeError(f"Failed to create test dataloader: {e}")


def save_predictions_summary(all_predictions: np.ndarray, prediction_metadata: List[Dict],
                           target_variables: List[str], output_dir: Path,
                           all_targets: Optional[np.ndarray] = None,
                           test_metrics: Optional[Dict] = None) -> str:
    """Save comprehensive predictions summary to file.
    
    Args:
        all_predictions: Array of all predictions.
        prediction_metadata: List of metadata for each prediction.
        target_variables: List of target variable names.
        output_dir: Output directory.
        all_targets: Optional array of target values.
        test_metrics: Optional test metrics.
        
    Returns:
        Path to saved summary file.
    """
    predictions_file = f"{output_dir}/all_predictions.npz"
    
    save_data = {
        'predictions': all_predictions,
        'prediction_metadata': prediction_metadata,
        'target_variables': target_variables,
        'prediction_means': np.mean(all_predictions, axis=0),
        'prediction_stds': np.std(all_predictions, axis=0),
        'timestamp': datetime.now().isoformat()
    }
    
    if all_targets is not None:
        save_data['targets'] = all_targets
        
    if test_metrics is not None:
        save_data['test_metrics'] = test_metrics
    
    np.savez_compressed(predictions_file, **save_data)
    return predictions_file


def run_test(options: Config, checkpoint_path: str, output_dir: Optional[str] = None, 
            test_list_path: Optional[str] = None) -> Dict[str, Any]:
    """Run test inference on the dataset.
    
    Args:
        options: Configuration object.
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory to save test results. If None, uses options.test_dir.
        test_list_path: Path to test data list. If None, uses validation data.
        
    Returns:
        Dictionary containing test results including predictions and statistics.
        
    Raises:
        FileNotFoundError: If checkpoint or test data files don't exist.
        RuntimeError: If test process fails.
    """
    # Setup logging
    logger = setup_logger(__name__, log_dir=options.log_dir)
    logger.info("Starting test inference...")
    logger.info(f"Checkpoint: {checkpoint_path}")
    if test_list_path:
        logger.info(f"Test data list: {test_list_path}")

    # Setup device
    device = setup_device(options, logger)

    # Create test dataloader
    try:
        test_dataloader = create_test_dataloader(options, test_list_path)
        logger.info(f"Test dataloader created with {len(test_dataloader)} batches.")
    except Exception as e:
        raise RuntimeError(f"Failed to create test dataloader: {e}")

    # Create and load model
    try:
        model = create_model(options, logger)
        model = load_model(model, checkpoint_path, device, logger)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"{options.test_dir}/test_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Test results will be saved to: {output_dir}")

    # Initialize tracking variables
    all_predictions = []
    all_targets = []
    prediction_metadata = []
    failed_batches = 0
    successful_predictions = 0

    logger.info("Running test inference...")
    
    # Run test inference loop
    for i, data_dict in enumerate(test_dataloader):
        try:
            prediction_dict = prediction_step(model, data_dict, device)
            file_names = data_dict["file_names"]
            
            # Process each sample in the batch
            for j in range(len(file_names)):
                file_name = os.path.splitext(file_names[j])[0]
                
                # Store predictions and metadata
                predictions = prediction_dict['outputs'][j]
                all_predictions.append(predictions)
                prediction_metadata.append({
                    'file_name': file_name,
                    'batch_index': i,
                    'sample_index': j,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Handle targets if available (for test sets with ground truth)
                has_targets = 'targets' in data_dict
                if has_targets:
                    targets = data_dict['targets'].cpu().numpy()[j]
                    all_targets.append(targets)
                    
                    # Create comparison plot
                    plot_path = f"{output_dir}/{file_name}_comparison"
                    plot_title = f'Test Prediction vs Ground Truth - {file_name}'
                    
                    try:
                        save_plot(
                            targets=targets, 
                            outputs=predictions,
                            target_variables=options.target_variables, 
                            stat_dict=test_dataloader.dataset.stat_dict,
                            plot_path=plot_path, 
                            plot_title=plot_title, 
                            logger=logger
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save comparison plot for {file_name}: {e}")
                else:
                    # Create prediction-only plot
                    plot_path = f"{output_dir}/{file_name}_prediction"
                    plot_title = f'Test Prediction - {file_name}'
                    
                    # For prediction-only plots, create dummy targets array
                    dummy_targets = np.zeros_like(predictions)
                    try:
                        save_plot(
                            targets=dummy_targets, 
                            outputs=predictions,
                            target_variables=options.target_variables, 
                            stat_dict=test_dataloader.dataset.stat_dict,
                            plot_path=plot_path, 
                            plot_title=plot_title, 
                            logger=logger
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save prediction plot for {file_name}: {e}")
                
                successful_predictions += 1
            
            # Log progress periodically
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(test_dataloader)} batches")
                
        except Exception as e:
            logger.warning(f"Test inference failed for batch {i}: {e}")
            failed_batches += 1
            continue

    # Validate that we have predictions
    if not all_predictions:
        logger.error("No successful test predictions completed")
        raise RuntimeError("Test inference failed completely")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)  # Shape: (n_samples, seq_len, n_vars)
    has_targets = len(all_targets) > 0
    
    if has_targets:
        all_targets = np.array(all_targets)

    # Calculate prediction statistics
    pred_means = np.mean(all_predictions, axis=0)  # Shape: (seq_len, n_vars)
    pred_stds = np.std(all_predictions, axis=0)
    
    # Calculate test metrics if targets are available
    test_metrics = None
    if has_targets:
        try:
            # Reshape for calculate_metrics function
            targets_reshaped = all_targets[..., np.newaxis, :].transpose(0, 2, 1)
            predictions_reshaped = all_predictions[..., np.newaxis, :].transpose(0, 2, 1)
            
            test_metrics = calculate_metrics(
                targets_reshaped, predictions_reshaped, options.target_variables
            )
            
            # Log metrics
            logger.info("Test Metrics:")
            for var_name, metrics in test_metrics.items():
                logger.info(f"  {var_name}:")
                logger.info(f"    RMSE: {metrics['rmse']:.4f}")
                logger.info(f"    MAE: {metrics['mae']:.4f}")
                logger.info(f"    Correlation: {metrics['correlation']:.4f}")
                logger.info(f"    R²: {metrics['r_squared']:.4f}")
            
            # Create overall test comparison plot
            try:
                overall_plot_path = f"{output_dir}/overall_test_results"
                overall_plot_title = 'Overall Test Results - Predictions vs Ground Truth'
                
                # Use mean across all samples for overall plot
                mean_targets = np.mean(all_targets, axis=0)
                mean_predictions = np.mean(all_predictions, axis=0)
                
                save_plot(
                    targets=mean_targets, 
                    outputs=mean_predictions,
                    target_variables=options.target_variables, 
                    stat_dict=test_dataloader.dataset.stat_dict,
                    plot_path=overall_plot_path, 
                    plot_title=overall_plot_title, 
                    logger=logger
                )
            except Exception as e:
                logger.warning(f"Failed to create overall test plot: {e}")
        except Exception as e:
            logger.warning(f"Failed to calculate test metrics: {e}")
    else:
        logger.info("No ground truth available - generating predictions only")

    # Save all predictions and metadata
    try:
        predictions_file = save_predictions_summary(
            all_predictions, prediction_metadata, options.target_variables,
            output_dir, all_targets if has_targets else None, test_metrics
        )
        logger.info(f"All predictions saved to: {predictions_file}")
    except Exception as e:
        logger.warning(f"Failed to save predictions summary: {e}")

    # Compile final results
    results = {
        'total_predictions': len(all_predictions),
        'failed_batches': failed_batches,
        'success_rate': (successful_predictions / len(test_dataloader)) * 100,
        'prediction_means': pred_means,
        'prediction_stds': pred_stds,
        'output_directory': str(output_dir),
        'has_ground_truth': has_targets
    }
    
    if test_metrics:
        results['test_metrics'] = test_metrics

    # Log final summary
    logger.info("=" * 50)
    logger.info("Test Results Summary")
    logger.info(f"Total Predictions: {len(all_predictions)}")
    logger.info(f"Failed Batches: {failed_batches}")
    logger.info(f"Success Rate: {results['success_rate']:.1f}%")
    logger.info(f"Ground Truth Available: {'Yes' if has_targets else 'No'}")
    
    if test_metrics:
        logger.info("Test Metrics Summary:")
        for var_name, metrics in test_metrics.items():
            logger.info(f"  {var_name} RMSE: {metrics['rmse']:.4f}")
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 50)

    return results


def main():
    """Main function for test script."""
    parser = argparse.ArgumentParser(description='Run model test inference')
    parser.add_argument('--config', type=str, default='config_dev.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_list', type=str, default=None,
                       help='Path to test data list file (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for test results')
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

        # Run test
        results = run_test(options, args.checkpoint, args.output_dir, args.test_list)
        
        # Print success summary
        print("=" * 60)
        print("Test inference completed successfully!")
        print(f"Total Predictions: {results['total_predictions']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Ground Truth Available: {'Yes' if results['has_ground_truth'] else 'No'}")
        
        if 'test_metrics' in results:
            print("\nTest Metrics Summary:")
            for var_name, metrics in results['test_metrics'].items():
                print(f"  {var_name}:")
                print(f"    RMSE: {metrics['rmse']:.4f}")
                print(f"    MAE: {metrics['mae']:.4f}")
                print(f"    R²: {metrics['r_squared']:.4f}")
        
        print(f"\nDetailed results saved to: {results['output_directory']}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
