import os
import time
from multiprocessing import freeze_support
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from utils import set_seed, setup_logger, TrainingLogger
from config import Config
from pipeline import create_dataloader
from networks import create_model


def denormalize_predictions(predictions, targets, stat_dict, variable_name):
    """Denormalize predictions and targets using statistics.
    
    Args:
        predictions (np.ndarray): Normalized predictions.
        targets (np.ndarray): Normalized targets.
        stat_dict (dict): Statistics dictionary containing mean and std.
        variable_name (str): Variable name to denormalize.
        
    Returns:
        tuple: Denormalized (predictions, targets).
        
    Raises:
        KeyError: If variable_name not found in stat_dict.
        ValueError: If statistics are invalid.
    """
    if variable_name not in stat_dict:
        raise KeyError(f"Variable '{variable_name}' not found in statistics dictionary")
    
    if 'mean' not in stat_dict[variable_name] or 'std' not in stat_dict[variable_name]:
        raise ValueError(f"Missing mean or std for variable '{variable_name}'")
    
    mean = stat_dict[variable_name]['mean']
    std = stat_dict[variable_name]['std']
    
    pred_denorm = (predictions * std) + mean
    true_denorm = (targets * std) + mean
    
    # Clip to non-negative values
    pred_denorm = np.clip(pred_denorm, 0, None)
    true_denorm = np.clip(true_denorm, 0, None)
    
    return pred_denorm, true_denorm


def train_one_epoch(model, dataloader, criterion, optimizer, device, options, 
                   training_logger, logger, epoch, iterations, target_variable, target_idx):
    """Train model for one epoch.
    
    Args:
        model: PyTorch model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Training device.
        options: Configuration options.
        training_logger: Training logger instance.
        logger: Logger instance.
        epoch (int): Current epoch number.
        iterations (int): Current iteration count.
        target_variable (str): Name of target variable to visualize.
        target_idx (int): Index of target variable in output tensor.
        
    Returns:
        int: Updated iteration count.
        
    Raises:
        RuntimeError: If training fails.
    """
    if len(dataloader) == 0:
        logger.warning("Empty training dataloader")
        return iterations
    
    running_loss = 0.0
    t0 = time.time()

    try:
        for i, data_dict in enumerate(dataloader):
            sdo = data_dict["sdo"].to(device)
            inputs = data_dict["inputs"].to(device)
            targets = data_dict["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, sdo)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            iterations += 1
            running_loss += loss.item()

            if iterations % options.report_freq == 0:
                avg_loss = running_loss / options.report_freq
                elapsed_time = time.time() - t0
                
                training_logger.log_progress(
                    epoch + 1, i + 1, iterations, avg_loss, elapsed_time
                )

                # Get first sample for visualization
                target = targets[0].cpu().detach().numpy()
                output = outputs[0].cpu().detach().numpy()

                # Denormalize target variable predictions
                try:
                    pred_values, true_values = denormalize_predictions(
                        output[:, target_idx], target[:, target_idx],   
                        dataloader.dataset.stat_dict, target_variable
                    )

                    # Save prediction plot
                    plot_path = f"{options.snapshot_dir}/epoch{epoch + 1}_batch{i + 1}.png"
                    save_prediction_plot(
                        true_values, pred_values, epoch + 1, i + 1, iterations, avg_loss, 
                        plot_path, logger, target_variable
                    )
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to create training plot: {e}")

                running_loss = 0.0
                t0 = time.time()
    except Exception as e:
        logger.error(f"Error during training epoch {epoch + 1}: {e}")
        raise RuntimeError(f"Training failed at epoch {epoch + 1}: {e}")
    
    return iterations

def save_prediction_plot(true_values, pred_values, epoch, batch_idx, iteration, loss, 
                        save_path, logger, variable_name):
    """Save prediction comparison plot.
    
    Args:
        true_values (np.ndarray): True values.
        pred_values (np.ndarray): Predicted values.
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
        iteration (int): Current iteration number.
        loss (float): Current loss value.
        save_path (str): Path to save the plot.
        logger: Logger instance.
        variable_name (str): Name of the variable being plotted.
    """
    fig = None
    try:
        if len(true_values) == 0 or len(pred_values) == 0:
            logger.warning("Empty prediction arrays, skipping plot save")
            return
            
        fig = plt.figure(figsize=(10, 5))
        plt.title(f'Epoch {epoch}, Batch {batch_idx}, Iteration {iteration}, Loss: {loss:.3f}')
        plt.subplot(1, 1, 1)
        plt.plot(true_values, label=f'True {variable_name}')
        plt.plot(pred_values, label=f'Predicted {variable_name}')
        plt.legend()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.debug(f"Training plot saved: {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save training plot: {e}")
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close('all')  # Fallback to close any open figures


def save_test_results(true_values, pred_values, epoch, batch_loss, save_path, 
                     logger, variable_name, file_name, batch_idx=None):
    """Save test results plot.
    
    Args:
        true_values (np.ndarray): True values.
        pred_values (np.ndarray): Predicted values.
        epoch (int): Current epoch number.
        batch_loss (float): Batch loss value.
        save_path (str): Path to save the plot.
        logger: Logger instance.
        variable_name (str): Name of the variable being plotted.
        batch_idx (int, optional): Batch index for title.
    """
    fig = None
    try:
        if len(true_values) == 0 or len(pred_values) == 0:
            logger.warning("Empty test result arrays, skipping plot save")
            return
            
        fig = plt.figure(figsize=(10, 5))
        if batch_idx is not None:
            plt.title(f'Test Results - Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss:.3f}, File: {file_name}')
        else:
            plt.title(f'Test Results - Epoch {epoch}, Loss: {batch_loss:.3f}, File: {file_name}')
        plt.subplot(1, 1, 1)
        plt.plot(true_values, label=f'True {variable_name}')
        plt.plot(pred_values, label=f'Predicted {variable_name}')
        plt.legend()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.debug(f"Test results saved: {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save test results: {e}")
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close('all')  # Fallback to close any open figures


def evaluate_model(model, dataloader, criterion, device, stat_dict, target_variable, 
                  target_idx, logger, test_dir, epoch):
    """Evaluate model on given dataloader and save all test plots.
    
    Args:
        model: PyTorch model.
        dataloader: Evaluation dataloader.
        criterion: Loss function.
        device: Evaluation device.
        stat_dict (dict): Statistics dictionary for denormalization.
        target_variable (str): Name of target variable to visualize.
        target_idx (int): Index of target variable in output tensor.
        logger: Logger instance.
        test_dir (str): Base test directory.
        epoch (int): Current epoch number.
        
    Returns:
        float: Average loss across all batches.
        
    Raises:
        RuntimeError: If evaluation fails.
    """
    if len(dataloader) == 0:
        logger.warning("Empty test dataloader")
        return 0.0
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Create epoch-specific directory
    epoch_dir = Path(test_dir) / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created test directory: {epoch_dir}")
    
    try:
        with torch.no_grad():
            for i, data_dict in enumerate(dataloader):
                sdo = data_dict["sdo"].to(device)
                inputs = data_dict["inputs"].to(device)
                targets = data_dict["targets"].to(device)
                file_names = data_dict["file_names"]
                
                outputs = model(inputs, sdo)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Process first sample in batch for visualization
                target = targets[0].cpu().numpy()
                output = outputs[0].cpu().numpy()
                file_name = file_names[0]
                
                try:
                    batch_pred, batch_true  = denormalize_predictions(
                        output[:, target_idx], target[:, target_idx],
                        stat_dict, target_variable
                    )
                    
                    # Save plot for this batch
                    plot_path = epoch_dir / f"test_batch_{i+1:04d}.png"
                    save_test_results(
                        batch_true, batch_pred, epoch, loss.item(), 
                        str(plot_path), logger, target_variable, file_name, i + 1
                    )
                    
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to denormalize test predictions for batch {i+1}: {e}")
                    
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise RuntimeError(f"Model evaluation failed: {e}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Saved {num_batches} test plots in {epoch_dir}")
    return avg_loss
    

if __name__ == '__main__':
    # For Windows compatibility
    freeze_support()

    options = Config()
    options = options.from_args_and_yaml(yaml_path="config_wulver.yaml", args=None)
    options.validate()
    options.make_directories()
    set_seed(options.seed)    

    # Setup logging
    logger = setup_logger(__name__, log_dir=options.log_dir)
    training_logger = TrainingLogger(logger, report_freq=options.report_freq)
    logger.info(f"Configuration: {options}")

    device = 'cuda' if torch.cuda.is_available() and options.device == 'cuda' else options.device
    if options.device == 'mps' and not torch.backends.mps.is_available():
        logger.error("MPS device is not available. Please check your PyTorch installation or use a different device.")
        raise RuntimeError("MPS device is not available. Please check your PyTorch installation or use a different device.")
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    dataloader = create_dataloader(options)
    logger.info(f"Dataloader created with {len(dataloader)} batches.")

    batch_size_train = options.batch_size    
    options.phase = 'test'
    options.batch_size = 1  # Use batch size of 1 for testing
    test_dataloader = create_dataloader(options)
    logger.info(f"Test dataloader created with {len(test_dataloader)} batches.")
    options.phase = 'train'
    options.batch_size = batch_size_train  # Restore original batch size

    model = create_model(options)
    model.to(device)
    logger.info(f"Model: {model}")

    if options.phase == 'train':
        model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)

        training_logger.start_training(options.num_epochs, len(dataloader))

        # Get target variable for visualization (use first target variable)
        if not options.target_variables:
            logger.error("No target variables specified in configuration")
            raise ValueError("No target variables specified in configuration")
            
        target_variable = options.target_variables[0]
        target_idx = 0  # Index in the output tensor
        logger.info(f"Using '{target_variable}' for visualization")

        iterations = 0
        
        try:
            for epoch in range(options.num_epochs):
                iterations = train_one_epoch(
                    model, dataloader, criterion, optimizer, device, 
                    options, training_logger, logger, epoch, iterations,
                    target_variable, target_idx
                )

                training_logger.log_epoch_complete(epoch + 1)

                # Run test evaluation
                if (epoch + 1) % options.test_freq == 0:
                    logger.info(f"Running test evaluation at epoch {epoch + 1}")
                    try:
                        test_loss, test_true, test_pred = evaluate_model(
                            model, test_dataloader, criterion, device, 
                            dataloader.dataset.stat_dict, target_variable, target_idx, logger,
                            options.test_dir, epoch
                        )

                        logger.info(f"Test loss at epoch {epoch + 1}: {test_loss:.3f}")
                        
                        # Save test results plot
                        # if test_true is not None and test_pred is not None:
                        #     test_plot_path = f"{options.test_dir}/test_epoch{epoch + 1}.png"
                        #     save_test_results(
                        #         test_true, test_pred, epoch + 1, test_loss, 
                        #         test_plot_path, logger, target_variable
                        #     )
                    except Exception as e:
                        logger.error(f"Test evaluation failed at epoch {epoch + 1}: {e}")
                    finally:
                        # Restore training 
                        model.train()

                # Save model checkpoint
                if (epoch + 1) % options.save_freq == 0:
                    try:
                        model_path = f"{options.checkpoint_dir}/model_epoch{epoch + 1}.pth"
                        torch.save(model.state_dict(), model_path)
                        training_logger.log_model_save(model_path, epoch + 1)
                        logger.info(f"Model checkpoint saved: {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to save model at epoch {epoch + 1}: {e}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            training_logger.finish_training()
        
        # Save final model
        try:
            final_model_path = f"{options.checkpoint_dir}/model.pth"
            torch.save(model.state_dict(), final_model_path)
            training_logger.log_model_save(final_model_path)
            logger.info(f"Final model saved: {final_model_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
            
    else:
        logger.info(f"Phase '{options.phase}' not implemented in this script")
