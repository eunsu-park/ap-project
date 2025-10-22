import os
import sys
import time
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import hydra

from utils import set_seed, setup_logger, setup_device, save_plot
from pipeline import create_dataloader
from networks import create_model
from losses import create_loss


def train_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
               criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, Any]:
    """Perform a single training step.
    
    Args:
        model: PyTorch model.
        data_dict: Dictionary containing input data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Training device.
        
    Returns:
        Dictionary containing loss and prediction results.

    Raises:
        RuntimeError: If training step fails.
    """
    model.train()
    try:
        sdo = data_dict["sdo"].to(device)
        inputs = data_dict["inputs"].to(device)
        targets = data_dict["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs, sdo)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        return {
            'loss': loss.item(),
            'targets': targets.cpu().detach().numpy(),
            'outputs': outputs.cpu().detach().numpy()
        }

    except Exception as e:
        raise RuntimeError(f"Training step failed: {e}")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, filepath: str,
                   logger: Optional = None) -> None:
    """Save model checkpoint with optimizer state.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch number.
        loss: Current loss value.
        filepath: Path to save checkpoint.
        logger: Optional logger for output.
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        
        message = f"Checkpoint saved: {filepath}"
        if logger:
            logger.info(message)
        else:
            print(message)
            
    except Exception as e:
        error_msg = f"Failed to save checkpoint {filepath}: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")


def train(config) -> None:
    """Run training process with comprehensive monitoring and error handling.
    
    Args:
        config: Configuration object containing training parameters.
        
    Raises:
        RuntimeError: If training setup or execution fails.
    """
    # Setup logging
    logger = setup_logger(__name__, log_dir=config.experiment.log_dir)
    logger.info(f"Training configuration:\n{config}")

    # Set seed and device
    set_seed(config.experiment.seed, logger=logger)
    device = setup_device(config, logger=logger)
    
    # Create dataloader
    try:
        dataloader = create_dataloader(config, logger=logger)
    except Exception as e:
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
        raise RuntimeError(f"Failed to create model: {e}")
        
    # Create loss function and optimizer
    try:
        criterion = create_loss(config, logger=logger)
        optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
        
        # Optional: Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup training components: {e}")

    # Training initialization
    start_time = datetime.now()
    logger.info("=" * 50)
    logger.info("Training Started")
    logger.info(f"Total epochs: {config.training.num_epochs}")
    logger.info(f"Batches per epoch: {len(dataloader)}")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    # Training tracking variables
    iterations = 0
    epochs = 0
    best_loss = float('inf')
    epoch_losses = []
    training_history = []

    try:
        while epochs < config.training.num_epochs:
            epoch_start_time = time.time()
            running_loss = 0.0
            batch_count = 0
            epoch_loss_sum = 0.0
            
            logger.info(f"Starting epoch {epochs + 1}/{config.training.num_epochs}")

            for i, data_dict in enumerate(dataloader):
                try:
                    # Training step
                    train_dict = train_step(model, data_dict, criterion, optimizer, device)
                    current_loss = train_dict['loss']
                    
                    iterations += 1
                    running_loss += current_loss
                    epoch_loss_sum += current_loss
                    batch_count += 1

                    # Progress reporting
                    if iterations % config.training.report_freq == 0:
                        avg_loss = running_loss / config.training.report_freq
                        elapsed_time = time.time() - epoch_start_time
                        progress = (i + 1) / len(dataloader) * 100
                        
                        logger.info(
                            f"[Epoch {epochs + 1}, Batch {i + 1}/{len(dataloader)}, "
                            f"Iteration {iterations}] loss: {avg_loss:.6f} | "
                            f"Time: {elapsed_time:.2f}s | Progress: {progress:.1f}%"
                        )

                        # Save training snapshot
                        try:
                            plot_path = f"{config.experiment.snapshot_dir}/iteration_{iterations}_epoch_{epochs+1}"
                            plot_title = f'Training Progress - Iteration {iterations}, Epoch {epochs + 1}'
                            save_plot(
                                targets=train_dict['targets'][0], 
                                outputs=train_dict['outputs'][0],
                                target_variables=config.data.target_variables, 
                                stat_dict=dataloader.dataset.stat_dict,
                                plot_path=plot_path, 
                                plot_title=plot_title, 
                                logger=logger
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save training snapshot: {e}")

                        running_loss = 0.0
                        
                except Exception as e:
                    logger.warning(f"Training step failed for batch {i}: {e}")
                    continue

            # End of epoch processing
            epochs += 1
            
            if batch_count > 0:
                epoch_avg_loss = epoch_loss_sum / batch_count
                epoch_losses.append(epoch_avg_loss)
                
                # Learning rate scheduling
                scheduler.step(epoch_avg_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                epoch_duration = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epochs} completed in {epoch_duration:.2f}s | "
                    f"Average loss: {epoch_avg_loss:.6f} | LR: {current_lr:.8f}"
                )
                
                # Save training history
                training_history.append({
                    'epoch': epochs,
                    'loss': epoch_avg_loss,
                    'learning_rate': current_lr,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Save best model
                if epoch_avg_loss < best_loss:
                    best_loss = epoch_avg_loss
                    best_model_path = f"{config.experiment.checkpoint_dir}/best_model.pth"
                    save_checkpoint(model, optimizer, epochs, epoch_avg_loss, best_model_path, logger)
                    logger.info(f"New best model saved with loss: {best_loss:.6f}")

            # Periodic model saving
            if epochs % config.training.model_save_freq == 0:
                model_path = f"{config.experiment.checkpoint_dir}/model_epoch{epochs}.pth"
                save_checkpoint(model, optimizer, epochs, epoch_avg_loss, model_path, logger)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Final model save
    final_model_path = f"{config.experiment.checkpoint_dir}/model_final.pth"
    final_loss = epoch_losses[-1] if epoch_losses else float('inf')
    save_checkpoint(model, optimizer, epochs, final_loss, final_model_path, logger)
    
    # Training completion
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    logger.info("=" * 50)
    logger.info("Training Completed")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Total iterations: {iterations}")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Final loss: {final_loss:.6f}")
    logger.info(f"Total training time: {total_duration}")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    # Save training history
    try:
        import json
        history_path = f"{config.experiment.log_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    # Plot training curve
    try:
        if epoch_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=2)
            plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            curve_path = f"{config.experiment.log_dir}/training_curve.png"
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Training curve saved: {curve_path}")
    except Exception as e:
        logger.warning(f"Failed to save training curve: {e}")


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    try :

        save_root = config.environment.save_root
        experiment_name = config.experiment.experiment_name
        experiment_dir = f"{save_root}/{experiment_name}"
        chechpoint_dir = f"{experiment_dir}/checkpoint"
        log_dir = f"{experiment_dir}/log"
        validation_dir = f"{experiment_dir}/validation"
        snapshot_dir = f"{experiment_dir}/snapshot"

        dir_paths = [
            experiment_dir, chechpoint_dir,
            log_dir, validation_dir, snapshot_dir
        ]
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

        config.experiment.experiment_dir = experiment_dir
        config.experiment.checkpoint_dir = chechpoint_dir
        config.experiment.log_dir = log_dir
        config.experiment.validation_dir = validation_dir
        config.experiment.snapshot_dir = snapshot_dir

        # Run training
        train(config)
        
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Results saved to: {config.experiment.experiment_dir}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    freeze_support()
    exit(main())
