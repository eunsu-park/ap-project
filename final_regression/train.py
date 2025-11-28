"""
Refactored training script for multimodal solar wind prediction.

Major improvements:
- Modular design with Trainer class
- Separated concerns (training, metrics, checkpoints)
- Cleaner main function
- Better error handling
"""

import os
from multiprocessing import freeze_support
import hydra
import torch
import torch.optim as optim

from utils import set_seed, setup_logger, setup_device
from pipeline import create_dataloader
from networks import create_model
from losses import create_loss_functions
from trainers import Trainer, save_training_history, plot_training_curves


def setup_training_environment(config):
    """Setup logging, seeding, and device.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (logger, device)
    """
    # Create directories
    save_root = config.environment.save_root
    experiment_name = config.experiment.experiment_name
    experiment_dir = f"{save_root}/{experiment_name}"
    
    directories = [
        f"{experiment_dir}/checkpoint",
        f"{experiment_dir}/log",
        f"{experiment_dir}/snapshot",
        f"{experiment_dir}/validation"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(__name__, log_dir=f"{experiment_dir}/log")
    logger.info(f"Training configuration:\n{config}")
    
    # Set seed and device
    set_seed(config.environment.seed, logger=logger)
    device = setup_device(config, logger=logger)
    
    return logger, device


def create_training_components(config, model, logger=None):
    """Create optimizer, scheduler, and loss functions.
    
    Args:
        config: Configuration object.
        model: PyTorch model.
        logger: Optional logger for output.
        
    Returns:
        Tuple of (optimizer, scheduler, criterion, contrastive_criterion)
    """
    # Optimizer
    if config.training.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate
        )
        opt_name = "Adam"
    elif config.training.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9
        )
        opt_name = "SGD"
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate
        )
        opt_name = "Adam (default)"
        if logger:
            logger.warning(f"Unknown optimizer '{config.training.optimizer}', using Adam")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Loss functions
    criterion, contrastive_criterion = create_loss_functions(config, logger)
    
    if logger:
        logger.info(f"Optimizer: {opt_name}, LR: {config.training.learning_rate}")
        logger.info(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    return optimizer, scheduler, criterion, contrastive_criterion


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    """Run training process.
    
    Args:
        config: Configuration object containing training parameters.
        
    Raises:
        RuntimeError: If training setup or execution fails.
    """
    # Setup environment
    logger, device = setup_training_environment(config)
    
    # Create dataloader
    try:
        dataloader = create_dataloader(config, logger=logger)
        logger.info(f"Dataloader created: {len(dataloader.dataset)} samples, "
                   f"{len(dataloader)} batches")
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
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
        logger.error(f"Failed to create model: {e}")
        raise RuntimeError(f"Failed to create model: {e}")
    
    # Create training components
    try:
        optimizer, scheduler, criterion, contrastive_criterion = create_training_components(
            config, model, logger
        )
    except Exception as e:
        logger.error(f"Failed to create training components: {e}")
        raise RuntimeError(f"Failed to create training components: {e}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        contrastive_criterion=contrastive_criterion,
        device=device,
        logger=logger
    )
    
    # Train
    try:
        history = trainer.fit(dataloader, config.training.num_epochs)
        
        # Save results
        save_training_history(history, config, logger)
        plot_training_curves(history, config, logger)
        
        logger.info("Training completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    freeze_support()
    main()

# import os
# import sys
# import time
# import argparse
# from datetime import datetime
# from multiprocessing import freeze_support
# from pathlib import Path
# from typing import Dict, Any, Optional

# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# import hydra

# from utils import set_seed, setup_logger, setup_device, save_plot
# from pipeline import create_dataloader
# from networks import create_model, MultiModalContrastiveLoss, MultiModalMSELoss
# from losses import create_loss


# def train_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
#                criterion: torch.nn.Module, 
#                contrastive_criterion: torch.nn.Module,
#                lambda_contrastive: float,
#                optimizer: torch.optim.Optimizer,
#                device: torch.device) -> Dict[str, Any]:
#     """Perform a single training step with contrastive loss.
    
#     Args:
#         model: PyTorch model.
#         data_dict: Dictionary containing input data.
#         criterion: Regression loss function (MSE or MAE).
#         contrastive_criterion: Contrastive loss function for multimodal alignment.
#         lambda_contrastive: Weight for contrastive loss component.
#         optimizer: Optimizer.
#         device: Training device.
        
#     Returns:
#         Dictionary containing losses, MAE, RMSE, alignment metrics, and predictions.

#     Raises:
#         RuntimeError: If training step fails.
#     """
#     model.train()
#     try:
#         sdo = data_dict["sdo"].to(device)
#         inputs = data_dict["inputs"].to(device)
#         targets = data_dict["targets"].to(device)
#         # targets = data_dict["targets"].to(device)

#         optimizer.zero_grad()
        
#         # Forward pass with feature extraction
#         outputs, transformer_features, convlstm_features = model(
#             inputs, sdo, return_features=True
#         )
        
#         # Regression loss
#         reg_loss = criterion(outputs, targets)
        
#         # Contrastive loss for multimodal alignment
#         cont_loss = contrastive_criterion(transformer_features, convlstm_features)
        
#         # Combined loss
#         total_loss = reg_loss + lambda_contrastive * cont_loss
        
#         total_loss.backward()
        
#         # Gradient clipping to prevent exploding gradients
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()

#         # Calculate regression metrics and alignment
#         with torch.no_grad():
#             # Calculate regression metrics: MAE and RMSE
#             mae = F.l1_loss(outputs, targets).item()
#             mse = F.mse_loss(outputs, targets).item()
#             rmse = torch.sqrt(torch.tensor(mse)).item()
            
#             # Calculate feature alignment (cosine similarity)
#             cosine_sim = F.cosine_similarity(
#                 transformer_features, convlstm_features, dim=1
#             ).mean().item()

#         return {
#             'loss': total_loss.item(),
#             'reg_loss': reg_loss.item(),
#             'cont_loss': cont_loss.item(),
#             'cosine_sim': cosine_sim,
#             'mae': mae,
#             'rmse': rmse,
#             'targets': targets.cpu().detach().numpy(),
#             'outputs': outputs.cpu().detach().numpy()
#         }

#     except Exception as e:
#         raise RuntimeError(f"Training step failed: {e}")


# def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
#                    epoch: int, loss: float, filepath: str,
#                    logger: Optional = None) -> None:
#     """Save model checkpoint with optimizer state.
    
#     Args:
#         model: PyTorch model.
#         optimizer: Optimizer.
#         epoch: Current epoch number.
#         loss: Current loss value.
#         filepath: Path to save checkpoint.
#         logger: Optional logger for output.
#     """
#     try:
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             'timestamp': datetime.now().isoformat()
#         }
        
#         # Ensure directory exists
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
#         torch.save(checkpoint, filepath)
        
#         message = f"Checkpoint saved: {filepath}"
#         if logger:
#             logger.info(message)
#         else:
#             print(message)
            
#     except Exception as e:
#         error_msg = f"Failed to save checkpoint {filepath}: {e}"
#         if logger:
#             logger.error(error_msg)
#         else:
#             print(f"Error: {error_msg}")


# @hydra.main(config_path="./configs", version_base=None)
# def main(config):
#     """Run training process with comprehensive monitoring and error handling.
    
#     Args:
#         config: Configuration object containing training parameters.
        
#     Raises:
#         RuntimeError: If training setup or execution fails.
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

#     logger = setup_logger(__name__, log_dir=log_dir)
#     logger.info(f"Training configuration:\n{config}")

#     # Set seed and device
#     set_seed(config.environment.seed, logger=logger)
#     device = setup_device(config, logger=logger)
    
#     # Create dataloader
#     try:
#         dataloader = create_dataloader(config, logger=logger)
#     except Exception as e:
#         raise RuntimeError(f"Failed to create dataloader: {e}")
    
#     # Create model
#     try:
#         model = create_model(config, logger=logger)
#         model.to(device)
        
#         # Log model parameters
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
#     except Exception as e:
#         raise RuntimeError(f"Failed to create model: {e}")
    
#     # Create loss function and optimizer
#     try:
#         # Commented out: Original config-based loss creation
#         # criterion = create_loss(config, logger=logger)
        
#         # Regression loss based on config
#         loss_type = config.training.get('loss_type', 'mse').lower()
        
#         if loss_type == 'mae' or loss_type == 'l1':
#             criterion = nn.L1Loss()
#             logger.info("Loss function: L1Loss (MAE) for regression")
#         elif loss_type == 'huber':
#             criterion = nn.HuberLoss(delta=10.0)
#             logger.info("Loss function: HuberLoss (delta=10.0, robust to outliers) for regression")
#         else:  # Default: MSE
#             criterion = nn.MSELoss()
#             logger.info("Loss function: MSELoss for regression")

#         logger.info("Model outputs continuous values for regression task")
        
#         # Contrastive loss for multimodal alignment
#         contrastive_type = config.training.get('contrastive_type', 'infonce')  # 'infonce' or 'mse'
#         lambda_contrastive = config.training.get('lambda_contrastive', 0.1)
        
#         if contrastive_type.lower() == 'mse':
#             # MSE-based consistency loss (better for small batch sizes)
#             contrastive_criterion = MultiModalMSELoss(reduction='mean').to(device)
#             logger.info(f"Contrastive Loss: MultiModalMSELoss (MSE-based consistency)")
#             logger.info(f"  - Direct feature alignment (no negative samples)")
#             logger.info(f"  - Suitable for small batch sizes")
#         else:  # infonce (default)
#             # InfoNCE contrastive loss (requires sufficient batch size)
#             contrastive_temperature = config.training.get('contrastive_temperature', 0.3)
#             contrastive_criterion = MultiModalContrastiveLoss(
#                 temperature=contrastive_temperature,
#                 normalize=True
#             ).to(device)
#             logger.info(f"Contrastive Loss: MultiModalContrastiveLoss (InfoNCE)")
#             logger.info(f"  - Temperature: {contrastive_temperature}")
#             logger.info(f"  - Uses in-batch negative samples")
#             logger.info(f"  - Batch size: {config.experiment.batch_size}")
        
#         logger.info(f"Lambda contrastive: {lambda_contrastive}")
        
#         optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
        
#         # Optional: Learning rate scheduler
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=10
#         )
        
#     except Exception as e:
#         raise RuntimeError(f"Failed to setup training components: {e}")

#     # Training initialization
#     start_time = datetime.now()
#     logger.info("=" * 50)
#     logger.info("Training Started")
#     logger.info(f"Total epochs: {config.training.num_epochs}")
#     logger.info(f"Batches per epoch: {len(dataloader)}")
#     logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     logger.info("=" * 50)
    
#     # Training tracking variables
#     iterations = 0
#     epochs = 0
#     best_loss = float('inf')
#     epoch_losses = []
#     epoch_reg_losses = []
#     epoch_cont_losses = []
#     epoch_cosine_sims = []
#     training_history = []

#     try:
#         while epochs < config.training.num_epochs:
#             epoch_start_time = time.time()
#             running_loss = 0.0
#             running_reg_loss = 0.0
#             running_cont_loss = 0.0
#             running_cosine_sim = 0.0
#             running_mae = 0.0
#             running_rmse = 0.0
#             batch_count = 0
#             epoch_loss_sum = 0.0
#             epoch_reg_loss_sum = 0.0
#             epoch_cont_loss_sum = 0.0
#             epoch_cosine_sim_sum = 0.0
#             epoch_mae_sum = 0.0
#             epoch_rmse_sum = 0.0

#             logger.info(f"Starting epoch {epochs + 1}/{config.training.num_epochs}")

#             for i, data_dict in enumerate(dataloader):
#                 try:
#                     # Training step
#                     train_dict = train_step(
#                         model, data_dict, criterion, 
#                         contrastive_criterion, lambda_contrastive,
#                         optimizer, device
#                     )
#                     current_loss = train_dict['loss']
#                     current_reg_loss = train_dict['reg_loss']
#                     current_cont_loss = train_dict['cont_loss']
#                     current_cosine_sim = train_dict['cosine_sim']
#                     current_mae = train_dict['mae']
#                     current_rmse = train_dict['rmse']
                    
#                     iterations += 1
#                     running_loss += current_loss
#                     running_reg_loss += current_reg_loss
#                     running_cont_loss += current_cont_loss
#                     running_cosine_sim += current_cosine_sim
#                     running_mae += current_mae
#                     running_rmse += current_rmse
#                     epoch_loss_sum += current_loss
#                     epoch_reg_loss_sum += current_reg_loss
#                     epoch_cont_loss_sum += current_cont_loss
#                     epoch_cosine_sim_sum += current_cosine_sim
#                     epoch_mae_sum += current_mae
#                     epoch_rmse_sum += current_rmse
#                     batch_count += 1

#                     # Progress reporting
#                     if iterations % config.training.report_freq == 0:
#                         avg_loss = running_loss / config.training.report_freq
#                         avg_reg_loss = running_reg_loss / config.training.report_freq
#                         avg_cont_loss = running_cont_loss / config.training.report_freq
#                         avg_cosine_sim = running_cosine_sim / config.training.report_freq
#                         avg_mae = running_mae / config.training.report_freq
#                         avg_rmse = running_rmse / config.training.report_freq
#                         elapsed_time = time.time() - epoch_start_time
#                         progress = (i + 1) / len(dataloader) * 100
                        
#                         logger.info(
#                             f"[Epoch {epochs + 1}, Batch {i + 1}/{len(dataloader)}, "
#                             f"Iteration {iterations}] "
#                             f"total_loss: {avg_loss:.6f} | "
#                             f"reg_loss: {avg_reg_loss:.6f} | "
#                             f"cont_loss: {avg_cont_loss:.6f} | "
#                             f"cosine_sim: {avg_cosine_sim:.4f} | "
#                             f"MAE: {avg_mae:.4f} | "
#                             f"RMSE: {avg_rmse:.4f} | "
#                             f"Time: {elapsed_time:.2f}s | Progress: {progress:.1f}%"
#                         )

#                         # Commented out: Save training snapshot plot
#                         """
#                         try:
#                             plot_path = f"{config.snapshot_dir}/iteration_{iterations}_epoch_{epochs+1}"
#                             plot_title = f'Training Progress - Iteration {iterations}, Epoch {epochs + 1}'
#                             save_plot(
#                                 targets=train_dict['targets'][0], 
#                                 outputs=train_dict['outputs'][0],
#                                 target_variables=config.target_variables, 
#                                 stat_dict=dataloader.dataset.stat_dict,
#                                 plot_path=plot_path, 
#                                 plot_title=plot_title, 
#                                 logger=logger
#                             )
#                         except Exception as e:
#                             logger.warning(f"Failed to save training snapshot: {e}")
#                         """

#                         running_loss = 0.0
#                         running_reg_loss = 0.0
#                         running_cont_loss = 0.0
#                         running_cosine_sim = 0.0
#                         running_mae = 0.0
#                         running_rmse = 0.0
                        
#                 except Exception as e:
#                     logger.warning(f"Training step failed for batch {i}: {e}")
#                     continue

#             # End of epoch processing
#             epochs += 1
            
#             if batch_count > 0:
#                 epoch_avg_loss = epoch_loss_sum / batch_count
#                 epoch_avg_reg_loss = epoch_reg_loss_sum / batch_count
#                 epoch_avg_cont_loss = epoch_cont_loss_sum / batch_count
#                 epoch_avg_cosine_sim = epoch_cosine_sim_sum / batch_count
#                 epoch_avg_mae = epoch_mae_sum / batch_count
#                 epoch_avg_rmse = epoch_rmse_sum / batch_count
                
#                 epoch_losses.append(epoch_avg_loss)
#                 epoch_reg_losses.append(epoch_avg_reg_loss)
#                 epoch_cont_losses.append(epoch_avg_cont_loss)
#                 epoch_cosine_sims.append(epoch_avg_cosine_sim)
                
#                 # Learning rate scheduling
#                 scheduler.step(epoch_avg_loss)
#                 current_lr = optimizer.param_groups[0]['lr']
                
#                 epoch_duration = time.time() - epoch_start_time
#                 logger.info(
#                     f"Epoch {epochs} completed in {epoch_duration:.2f}s | "
#                     f"Total loss: {epoch_avg_loss:.6f} | "
#                     f"Reg loss: {epoch_avg_reg_loss:.6f} | "
#                     f"Cont loss: {epoch_avg_cont_loss:.6f} | "
#                     f"Cosine sim: {epoch_avg_cosine_sim:.4f} | "
#                     f"MAE: {epoch_avg_mae:.4f} | "
#                     f"RMSE: {epoch_avg_rmse:.4f} | "
#                     f"LR: {current_lr:.8f}"
#                 )
                
#                 # Save training history
#                 training_history.append({
#                     'epoch': epochs,
#                     'loss': epoch_avg_loss,
#                     'reg_loss': epoch_avg_reg_loss,
#                     'cont_loss': epoch_avg_cont_loss,
#                     'cosine_sim': epoch_avg_cosine_sim,
#                     'mae': epoch_avg_mae,
#                     'rmse': epoch_avg_rmse,
#                     'learning_rate': current_lr,
#                     'timestamp': datetime.now().isoformat()
#                 })
                
#                 # Save best model (loss 기준)
#                 if epoch_avg_loss < best_loss:
#                     best_loss = epoch_avg_loss
#                     best_model_path = f"{checkpoint_dir}/best_model.pth"
#                     save_checkpoint(model, optimizer, epochs, epoch_avg_loss, best_model_path, logger)
#                     logger.info(f"New best model saved with loss: {best_loss:.6f}")

#             # Periodic model saving
#             if epochs % config.training.model_save_freq == 0:
#                 model_path = f"{checkpoint_dir}/model_epoch{epochs}.pth"
#                 save_checkpoint(model, optimizer, epochs, epoch_avg_loss, model_path, logger)

#     except KeyboardInterrupt:
#         logger.info("Training interrupted by user")
#     except Exception as e:
#         logger.error(f"Training failed: {e}")
#         raise

#     # Final model save
#     final_model_path = f"{checkpoint_dir}/model_final.pth"
#     final_loss = epoch_losses[-1] if epoch_losses else float('inf')
#     save_checkpoint(model, optimizer, epochs, final_loss, final_model_path, logger)
    
#     # Training completion
#     end_time = datetime.now()
#     total_duration = end_time - start_time
    
#     logger.info("=" * 50)
#     logger.info("Training Completed")
#     logger.info(f"Total epochs: {epochs}")
#     logger.info(f"Total iterations: {iterations}")
#     logger.info(f"Best loss: {best_loss:.6f}")
#     logger.info(f"Final loss: {final_loss:.6f}")
#     logger.info(f"Total training time: {total_duration}")
#     logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     logger.info("=" * 50)

#     # Save training history
#     try:
#         import json
#         history_path = f"{log_dir}/training_history.json"
#         with open(history_path, 'w') as f:
#             json.dump(training_history, f, indent=2)
#         logger.info(f"Training history saved: {history_path}")
#     except Exception as e:
#         logger.warning(f"Failed to save training history: {e}")

#     # Plot training curves
#     try:
#         if epoch_losses:
#             fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
#             # Total loss
#             axes[0, 0].plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=2)
#             axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
#             axes[0, 0].set_xlabel('Epoch')
#             axes[0, 0].set_ylabel('Loss')
#             axes[0, 0].grid(True, alpha=0.3)
            
#             # Classification vs Contrastive Loss
#             axes[0, 1].plot(range(1, len(epoch_reg_losses) + 1), epoch_reg_losses, 'r-', 
#                            linewidth=2, label='Regression Loss')
#             axes[0, 1].plot(range(1, len(epoch_cont_losses) + 1), epoch_cont_losses, 'g-', 
#                            linewidth=2, label='Contrastive Loss')
#             axes[0, 1].set_title('Loss Components', fontsize=12, fontweight='bold')
#             axes[0, 1].set_xlabel('Epoch')
#             axes[0, 1].set_ylabel('Loss')
#             axes[0, 1].legend()
#             axes[0, 1].grid(True, alpha=0.3)
            
#             # Cosine Similarity
#             axes[1, 0].plot(range(1, len(epoch_cosine_sims) + 1), epoch_cosine_sims, 
#                            'm-', linewidth=2)
#             axes[1, 0].set_title('Feature Alignment (Cosine Similarity)', 
#                                 fontsize=12, fontweight='bold')
#             axes[1, 0].set_xlabel('Epoch')
#             axes[1, 0].set_ylabel('Cosine Similarity')
#             axes[1, 0].grid(True, alpha=0.3)
#             axes[1, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
#             axes[1, 0].legend()
            
#             # MAE and RMSE
#             mae_history = [h['mae'] for h in training_history]
#             rmse_history = [h['rmse'] for h in training_history]
#             axes[1, 1].plot(range(1, len(mae_history) + 1), mae_history, 
#                            'c-', linewidth=2, label='MAE')
#             axes[1, 1].plot(range(1, len(rmse_history) + 1), rmse_history, 
#                            'orange', linewidth=2, label='RMSE')
#             axes[1, 1].set_title('Regression Metrics', fontsize=12, fontweight='bold')
#             axes[1, 1].set_xlabel('Epoch')
#             axes[1, 1].set_ylabel('Error')
#             axes[1, 1].legend()
#             axes[1, 1].grid(True, alpha=0.3)
            
#             plt.tight_layout()
#             curve_path = f"{log_dir}/training_curves.png"
#             plt.savefig(curve_path, dpi=150, bbox_inches='tight')
#             plt.close()
#             logger.info(f"Training curves saved: {curve_path}")
#     except Exception as e:
#         logger.warning(f"Failed to save training curves: {e}")


# if __name__ == '__main__':
#     freeze_support()
#     main()
