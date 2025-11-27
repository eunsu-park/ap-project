import os
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import hydra
import torch
import torch.nn as nn
import numpy as np

from utils import setup_logger, setup_device, load_model
from pipeline import create_dataloader
from networks import create_model


def validation_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
                   device: torch.device) -> Dict[str, Any]:
    """Perform a single validation step and return prediction results.
    
    Args:
        model: PyTorch model.
        data_dict: Dictionary containing input data.
        device: Device for computation.
        
    Returns:
        Dictionary containing targets, predictions, and metadata.
    """
    model.eval()
    
    sdo = data_dict["sdo"].to(device)
    inputs = data_dict["inputs"].to(device)
    targets = data_dict["labels"].to(device)

    with torch.no_grad():
        outputs = model(inputs, sdo)  # Logits
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).float()

    return {
        'targets': targets.cpu().numpy(),
        'predictions': predictions.cpu().numpy(),
        'probabilities': probs.cpu().numpy()
    }


def check_incorrect_samples(targets: np.ndarray, predictions: np.ndarray, 
                           target_variables: list) -> Dict[str, List[int]]:
    """Check which samples have incorrect predictions for each variable.
    
    Args:
        targets: Array of shape (n_groups, n_variables)
        predictions: Array of shape (n_groups, n_variables)
        target_variables: List of target variable names
        
    Returns:
        Dictionary mapping variable names to lists of incorrect group indices
    """
    n_groups, n_variables = targets.shape
    incorrect_by_variable = {}
    
    for var_idx, var_name in enumerate(target_variables):
        incorrect_groups = []
        for group_idx in range(n_groups):
            if targets[group_idx, var_idx] != predictions[group_idx, var_idx]:
                incorrect_groups.append(group_idx)
        
        if incorrect_groups:
            incorrect_by_variable[var_name] = incorrect_groups
    
    return incorrect_by_variable


def save_incorrect_files_list(incorrect_files: List[Dict[str, Any]], 
                              output_path: str, 
                              target_variables: list,
                              logger=None):
    """Save list of files with incorrect predictions.
    
    Args:
        incorrect_files: List of dictionaries containing file info and errors
        output_path: Path to save the results file
        target_variables: List of target variable names
        logger: Optional logger
    """
    try:
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FILES WITH INCORRECT PREDICTIONS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files with errors: {len(incorrect_files)}\n\n")
            
            # Summary by variable
            f.write("=" * 80 + "\n")
            f.write("SUMMARY BY VARIABLE\n")
            f.write("=" * 80 + "\n\n")
            
            error_counts = {var: 0 for var in target_variables}
            for file_info in incorrect_files:
                for var_name in file_info['incorrect_variables']:
                    error_counts[var_name] += len(file_info['incorrect_variables'][var_name])
            
            for var_name, count in error_counts.items():
                f.write(f"{var_name}: {count} incorrect predictions\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED FILE LIST\n")
            f.write("=" * 80 + "\n\n")
            
            # Detailed list
            for idx, file_info in enumerate(incorrect_files, 1):
                f.write(f"{idx}. Batch Index: {file_info['batch_idx']}\n")
                
                # Print file_names (can be a list)
                if 'file_names' in file_info and file_info['file_names'] is not None:
                    if isinstance(file_info['file_names'], list):
                        f.write(f"   File Names:\n")
                        for fname in file_info['file_names']:
                            f.write(f"      - {fname}\n")
                    else:
                        f.write(f"   File Name: {file_info['file_names']}\n")
                
                f.write(f"   Incorrect Variables:\n")
                for var_name, group_indices in file_info['incorrect_variables'].items():
                    f.write(f"      - {var_name}: groups {group_indices}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        if logger:
            logger.info(f"Incorrect files list saved to: {output_path}")
        else:
            print(f"Incorrect files list saved to: {output_path}")
            
    except Exception as e:
        error_msg = f"Failed to save incorrect files list: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")


@hydra.main(config_path="./configs", version_base=None)
def main(config) -> Dict[str, Any]:
    """List files with incorrect predictions.
    
    Args:
        config: Hydra configuration object.
        
    Returns:
        Dictionary containing list of incorrect files.
    """
    # Setup directories
    save_root = config.environment.save_root
    experiment_name = config.experiment.experiment_name
    experiment_dir = f"{save_root}/{experiment_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoint"
    log_dir = f"{experiment_dir}/log"
    validation_dir = f"{experiment_dir}/validation"
    
    for directory in [experiment_dir, checkpoint_dir, log_dir, validation_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    checkpoint_path = config.validation.checkpoint_path
    output_dir = config.validation.output_dir

    # Setup logging
    logger = setup_logger(__name__, log_dir=log_dir)
    logger.info("Starting incorrect predictions listing...")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Setup device
    device = setup_device(config, logger)

    # Create validation dataloader
    validation_config = copy.deepcopy(config)
    validation_config.experiment.phase = 'validation'
    validation_config.experiment.batch_size = 1

    try:
        validation_dataloader = create_dataloader(validation_config, logger=logger)
        logger.info(f"Validation dataloader created with {len(validation_dataloader)} batches.")
    except Exception as e:
        raise RuntimeError(f"Failed to create validation dataloader: {e}")

    # Create and load model
    try:
        model = create_model(config, logger=logger)
        model = load_model(model, checkpoint_path, device, logger=logger)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"{validation_dir}/incorrect_files_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # Initialize tracking
    incorrect_files = []
    total_processed = 0
    total_with_errors = 0

    logger.info("Checking predictions...")
    
    # Run validation loop
    for i, data_dict in enumerate(validation_dataloader):
        try:
            validation_dict = validation_step(model, data_dict, device)
            
            targets = validation_dict['targets']  # Shape: (1, n_groups, n_vars)
            predictions = validation_dict['predictions']  # Shape: (1, n_groups, n_vars)
            
            # Remove batch dimension
            targets = targets[0]  # Shape: (n_groups, n_vars)
            predictions = predictions[0]  # Shape: (n_groups, n_vars)
            
            # Check for incorrect predictions
            incorrect_vars = check_incorrect_samples(
                targets, predictions, config.data.target_variables
            )
            
            # If there are any incorrect predictions, record this file
            if incorrect_vars:
                file_info = {
                    'batch_idx': i,
                    'incorrect_variables': incorrect_vars
                }
                
                # Get file_names from data_dict
                if 'file_names' in data_dict:
                    # file_names might be a list or tensor, handle both cases
                    file_names = data_dict['file_names']
                    if isinstance(file_names, torch.Tensor):
                        file_info['file_names'] = file_names.tolist()
                    elif isinstance(file_names, list):
                        file_info['file_names'] = file_names
                    else:
                        file_info['file_names'] = str(file_names)
                else:
                    file_info['file_names'] = None
                
                incorrect_files.append(file_info)
                total_with_errors += 1
            
            total_processed += 1
            
            # Log progress periodically
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(validation_dataloader)} batches | "
                    f"Files with errors: {total_with_errors}"
                )
                
        except Exception as e:
            logger.warning(f"Failed to process batch {i}: {e}")
            continue

    # Summary statistics
    error_rate = (total_with_errors / total_processed * 100) if total_processed > 0 else 0
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {total_processed}")
    logger.info(f"Files with incorrect predictions: {total_with_errors}")
    logger.info(f"Error rate: {error_rate:.2f}%")
    logger.info("=" * 80 + "\n")

    # Save results
    results_file_path = output_dir / "incorrect_files_list.txt"
    save_incorrect_files_list(
        incorrect_files, 
        str(results_file_path), 
        config.data.target_variables,
        logger
    )

    # Print summary to console
    print("=" * 60)
    print("Incorrect Predictions Listing Completed!")
    print(f"Total files processed: {total_processed}")
    print(f"Files with errors: {total_with_errors} ({error_rate:.2f}%)")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)

    # Compile results
    results = {
        'incorrect_files': incorrect_files,
        'total_processed': total_processed,
        'total_with_errors': total_with_errors,
        'error_rate': error_rate,
        'output_directory': str(output_dir)
    }

    return results


if __name__ == '__main__':
    main()