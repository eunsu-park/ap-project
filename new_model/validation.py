import os
import copy
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import hydra
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from utils import set_seed, setup_logger, setup_device, load_model
from pipeline import create_dataloader
from networks import create_model


def calculate_brier_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate Brier Score for probability predictions.
    
    Args:
        targets: Binary target values (0 or 1)
        probabilities: Predicted probabilities (between 0 and 1)
        
    Returns:
        Brier Score (lower is better, ranges from 0 to 1)
    """
    return np.mean((probabilities - targets) ** 2)


def calculate_brier_skill_score(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    """Calculate Brier Skill Score and related metrics.
    
    The Brier Skill Score (BSS) measures the improvement over a reference forecast
    (typically climatology - the sample mean).
    
    BSS = 1 - (BS / BS_ref)
    where BS_ref is the Brier Score of the reference forecast.
    
    Args:
        targets: Binary target values (0 or 1)
        probabilities: Predicted probabilities (between 0 and 1)
        
    Returns:
        Dictionary containing:
        - Brier_Score: The Brier Score of the model
        - Brier_Score_Reference: Brier Score of climatology forecast
        - Brier_Skill_Score: The skill score (BSS)
    """
    # Calculate Brier Score for the model
    brier_score = calculate_brier_score(targets, probabilities)
    
    # Calculate reference Brier Score (climatology: always predict the mean)
    climatology = np.mean(targets)
    brier_score_ref = calculate_brier_score(targets, np.full_like(probabilities, climatology))
    
    # Calculate Brier Skill Score
    # BSS = 1 means perfect forecast
    # BSS = 0 means no improvement over climatology
    # BSS < 0 means worse than climatology
    if brier_score_ref > 0:
        brier_skill_score = 1 - (brier_score / brier_score_ref)
    else:
        brier_skill_score = 0.0
    
    return {
        'Brier_Score': brier_score,
        'Brier_Score_Reference': brier_score_ref,
        'Brier_Skill_Score': brier_skill_score
    }


def calculate_contingency_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Calculate contingency table based metrics for binary classification.
    
    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        tn: True Negatives
        
    Returns:
        Dictionary containing various contingency-based metrics
    """
    metrics = {}
    
    # Basic counts
    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['FN'] = fn
    metrics['TN'] = tn
    
    # POD (Probability of Detection) = Hit Rate = Sensitivity = Recall
    metrics['POD'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # FAR (False Alarm Ratio)
    metrics['FAR'] = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # POFD (Probability of False Detection) = Fall-out
    metrics['POFD'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # TSS (True Skill Statistic) = Hanssen-Kuipers Score
    metrics['TSS'] = metrics['POD'] - metrics['POFD']
    
    # CSI (Critical Success Index) = Threat Score
    metrics['CSI'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Bias Score
    metrics['Bias'] = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # HSS (Heidke Skill Score)
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    metrics['HSS'] = numerator / denominator if denominator > 0 else 0.0
    
    # Accuracy
    metrics['Accuracy'] = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    # Precision (1 - FAR)
    metrics['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1 Score
    if metrics['Precision'] + metrics['POD'] > 0:
        metrics['F1'] = 2 * metrics['Precision'] * metrics['POD'] / (metrics['Precision'] + metrics['POD'])
    else:
        metrics['F1'] = 0.0
    
    # Specificity (True Negative Rate)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


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
        # targets = data_dict["targets"].to(device)
        targets = data_dict["labels"].to(device)  # For binary classification

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
                                     all_probabilities: np.ndarray,
                                     target_variables: list) -> Dict[str, Dict[str, float]]:
    """Calculate classification metrics for each target variable.
    
    Args:
        all_targets: Array of shape (n_samples, n_groups, n_variables)
        all_predictions: Array of shape (n_samples, n_groups, n_variables)
        all_probabilities: Array of shape (n_samples, n_groups, n_variables)
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
        var_probabilities = all_probabilities[:, :, var_idx].flatten()
        
        # Calculate basic sklearn metrics
        accuracy = accuracy_score(var_targets, var_predictions)
        
        # Handle cases where there might be only one class
        try:
            precision = precision_score(var_targets, var_predictions, zero_division=0)
            recall = recall_score(var_targets, var_predictions, zero_division=0)
            f1 = f1_score(var_targets, var_predictions, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(var_targets, var_predictions, labels=[0, 1])
        
        # Extract TP, FP, FN, TN from confusion matrix
        # cm[0,0] = TN, cm[0,1] = FP, cm[1,0] = FN, cm[1,1] = TP
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        
        # Calculate contingency table based metrics
        contingency_metrics = calculate_contingency_metrics(
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)
        )
        
        # Calculate Brier Skill Score
        brier_metrics = calculate_brier_skill_score(var_targets, var_probabilities)
        
        # Combine all metrics
        metrics_dict[var_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'positive_rate': var_targets.mean(),  # Class balance
            'predicted_positive_rate': var_predictions.mean(),
            # Add contingency metrics
            **contingency_metrics,
            # Add Brier metrics
            **brier_metrics
        }
    
    return metrics_dict


def save_results_to_csv(all_file_results: list,
                       output_path: str,
                       target_variables: list,
                       logger=None):
    """Save all validation results to CSV file.

    Args:
        all_file_results: List of dictionaries with file_name, targets, predictions
        output_path: Path to save the CSV file
        target_variables: List of target variable names
        logger: Optional logger
    """
    try:
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['file_name', 'target', 'output', 'output_prob']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Write each result
            for result in all_file_results:
                file_name = result['file_name']
                targets = result['targets']  # Shape: (n_groups, n_variables)
                predictions = result['predictions']  # Shape: (n_groups, n_variables)
                probabilities = result['probabilities']
                
                n_groups, n_variables = targets.shape
                
                # For each variable
                for var_idx, var_name in enumerate(target_variables):
                    # For each group
                    for group_idx in range(n_groups):
                        target_val = int(targets[group_idx, var_idx])
                        pred_val = int(predictions[group_idx, var_idx])
                        pred_prob = probabilities[group_idx, var_idx]
                        
                        # Create a descriptive identifier
                        full_identifier = f"{file_name}_group{group_idx}_{var_name}"
                        
                        writer.writerow({
                            'file_name': full_identifier,
                            'target': target_val,
                            'output': pred_val,
                            'output_prob' : pred_prob
                        })
        
        if logger:
            logger.info(f"Validation results CSV saved to: {output_path}")
        else:
            print(f"Validation results CSV saved to: {output_path}")
            
    except Exception as e:
        error_msg = f"Failed to save validation results CSV: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")


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
                f.write("=" * 80 + "\n")
                
                # Contingency Table
                f.write("Contingency Table:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  True Positives  (TP): {metrics['TP']}\n")
                f.write(f"  False Positives (FP): {metrics['FP']}\n")
                f.write(f"  False Negatives (FN): {metrics['FN']}\n")
                f.write(f"  True Negatives  (TN): {metrics['TN']}\n\n")
                
                # Confusion Matrix visualization
                f.write("Confusion Matrix:\n")
                f.write("-" * 40 + "\n")
                f.write("                Predicted\n")
                f.write("                Neg    Pos\n")
                f.write(f"Actual  Neg   [{metrics['TN']:6d} {metrics['FP']:6d}]\n")
                f.write(f"        Pos   [{metrics['FN']:6d} {metrics['TP']:6d}]\n\n")
                
                # Basic Metrics
                f.write("Basic Classification Metrics:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n\n")
                
                # Contingency-based Metrics
                f.write("Contingency-based Metrics:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  POD (Probability of Detection):  {metrics['POD']:.4f}\n")
                f.write(f"  FAR (False Alarm Ratio):         {metrics['FAR']:.4f}\n")
                f.write(f"  POFD (Prob. of False Detection): {metrics['POFD']:.4f}\n")
                f.write(f"  TSS (True Skill Statistic):      {metrics['TSS']:.4f}\n")
                f.write(f"  CSI (Critical Success Index):    {metrics['CSI']:.4f}\n")
                f.write(f"  Bias Score:                      {metrics['Bias']:.4f}\n")
                f.write(f"  HSS (Heidke Skill Score):        {metrics['HSS']:.4f}\n")
                f.write(f"  Specificity:                     {metrics['Specificity']:.4f}\n\n")
                
                # Brier Score Metrics
                f.write("Probabilistic Metrics:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Brier Score:                     {metrics['Brier_Score']:.4f}\n")
                f.write(f"  Brier Score (Reference):         {metrics['Brier_Score_Reference']:.4f}\n")
                f.write(f"  Brier Skill Score:               {metrics['Brier_Skill_Score']:.4f}\n\n")
                
                # Class distribution
                f.write("Class Distribution:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  True Positive Rate:      {metrics['positive_rate']:.4f}\n")
                f.write(f"  Predicted Positive Rate: {metrics['predicted_positive_rate']:.4f}\n")
                f.write("\n" + "=" * 80 + "\n\n")
            
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


@hydra.main(config_path="./configs", version_base=None)
def main(config) -> Dict[str, Any]:
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

    save_root = config.environment.save_root
    experiment_name = config.experiment.experiment_name
    experiment_dir = f"{save_root}/{experiment_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoint"
    log_dir = f"{experiment_dir}/log"
    snapshot_dir = f"{experiment_dir}/snapshot"
    validation_dir = f"{experiment_dir}/validation"
    for directory in [experiment_dir, checkpoint_dir, log_dir, snapshot_dir, validation_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    checkpoint_path = config.validation.checkpoint_path
    output_dir = config.validation.output_dir

    logger = setup_logger(__name__, log_dir=log_dir)
    logger.info("Starting validation...")
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

    # Setup loss function - Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()
    logger.info("Using BCEWithLogitsLoss for binary classification validation")

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"{validation_dir}/validation_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Validation results will be saved to: {output_dir}")

    # Initialize tracking variables
    validation_losses = []
    validation_accuracies = []
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_file_results = []  # NEW: For CSV export
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
            
            # NEW: Store file-level results for CSV export
            targets = validation_dict['targets'][0]  # Remove batch dimension
            predictions = validation_dict['predictions'][0]
            
            # Get file_name
            file_name = f"batch_{i}"
            if 'file_names' in data_dict:
                file_names = data_dict['file_names']
                if isinstance(file_names, torch.Tensor):
                    file_name = str(file_names.tolist()[0]) if len(file_names.tolist()) > 0 else file_name
                elif isinstance(file_names, list):
                    file_name = str(file_names[0]) if len(file_names) > 0 else file_name
                else:
                    file_name = str(file_names)
            
            all_file_results.append({
                'file_name': file_name,
                'targets': targets,
                'predictions': predictions
            })
            
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
    
    # Calculate per-variable classification metrics (including contingency and Brier metrics)
    metrics_per_variable = calculate_classification_metrics(
        all_targets, all_predictions, all_probabilities, config.data.target_variables
    )
    
    # Log individual variable metrics
    logger.info("\n" + "=" * 80)
    logger.info("METRICS BY VARIABLE")
    logger.info("=" * 80)
    for var_name, metrics in metrics_per_variable.items():
        logger.info(f"\n{var_name}:")
        logger.info(f"  Contingency Table: TP={metrics['TP']}, FP={metrics['FP']}, "
                   f"FN={metrics['FN']}, TN={metrics['TN']}")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  POD:       {metrics['POD']:.4f}")
        logger.info(f"  FAR:       {metrics['FAR']:.4f}")
        logger.info(f"  TSS:       {metrics['TSS']:.4f}")
        logger.info(f"  CSI:       {metrics['CSI']:.4f}")
        logger.info(f"  Brier Score: {metrics['Brier_Score']:.4f}")
        logger.info(f"  Brier Skill Score: {metrics['Brier_Skill_Score']:.4f}")

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
    
    # NEW: Save results to CSV file
    csv_file_path = output_dir / "validation_results.csv"
    save_results_to_csv(all_file_results, str(csv_file_path), config.data.target_variables, logger)

    print("=" * 60)
    print("Validation completed successfully!")
    print(f"Average Loss:     {results['average_loss']:.6f}")
    print(f"Average Accuracy: {results['average_accuracy']:.4f}")
    print(f"Success Rate:     {results['success_rate']:.1f}%")
    print("\nKey Metrics by Variable:")
    for var_name, metrics in results['metrics_per_variable'].items():
        print(f"  {var_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}, TSS: {metrics['TSS']:.4f}, "
                f"CSI: {metrics['CSI']:.4f}, BSS: {metrics['Brier_Skill_Score']:.4f}")
    print(f"\nDetailed results saved to: {results['output_directory']}")
    print(f"CSV results saved to: {csv_file_path}")
    print("=" * 60)

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Average Loss:     {avg_loss:.6f} (±{std_loss:.6f})")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    logger.info(f"Total Samples:    {successful_samples}")
    logger.info(f"Failed Batches:   {failed_batches}")
    logger.info(f"Success Rate:     {results['success_rate']:.1f}%")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"CSV saved to: {csv_file_path}")
    logger.info("=" * 80 + "\n")

    return results


if __name__ == '__main__':
    main()