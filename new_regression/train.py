import os
import sys
import time
import argparse
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import hydra

from utils import set_seed, setup_logger, setup_device, save_plot
from pipeline import create_dataloader
from networks import create_model, MultiModalContrastiveLoss, MultiModalMSELoss
# from losses import create_loss  # Commented out - using BCEWithLogitsLoss instead


def train_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
               criterion: torch.nn.Module, 
               contrastive_criterion: torch.nn.Module,
               lambda_contrastive: float,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, Any]:
    """Perform a single training step with contrastive loss.
    
    Args:
        model: PyTorch model.
        data_dict: Dictionary containing input data.
        criterion: Classification loss function.
        contrastive_criterion: Contrastive loss function for multimodal alignment.
        lambda_contrastive: Weight for contrastive loss component.
        optimizer: Optimizer.
        device: Training device.
        
    Returns:
        Dictionary containing losses, accuracy, alignment metrics, and predictions.

    Raises:
        RuntimeError: If training step fails.
    """
    model.train()
    try:
        sdo = data_dict["sdo"].to(device)
        inputs = data_dict["inputs"].to(device)
        targets = data_dict["labels"].to(device)
        # targets = data_dict["targets"].to(device)

        optimizer.zero_grad()
        
        # Forward pass with feature extraction
        outputs, transformer_features, convlstm_features = model(
            inputs, sdo, return_features=True
        )
        
        # Classification loss
        cls_loss = criterion(outputs, targets)
        
        # Contrastive loss for multimodal alignment
        cont_loss = contrastive_criterion(transformer_features, convlstm_features)
        
        # Combined loss
        total_loss = cls_loss + lambda_contrastive * cont_loss
        
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Calculate accuracy and alignment metrics
        with torch.no_grad():
            # Convert logits to probabilities and then to binary predictions
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            # Calculate accuracy (correct predictions / total predictions)
            accuracy = (predictions == targets).float().mean().item()
            
            # Calculate feature alignment (cosine similarity)
            cosine_sim = F.cosine_similarity(
                transformer_features, convlstm_features, dim=1
            ).mean().item()

        return {
            'loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'cont_loss': cont_loss.item(),
            'cosine_sim': cosine_sim,
            'accuracy': accuracy,
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


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    """Run training process with comprehensive monitoring and error handling.
    
    Args:
        config: Configuration object containing training parameters.
        
    Raises:
        RuntimeError: If training setup or execution fails.
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

    logger = setup_logger(__name__, log_dir=log_dir)
    logger.info(f"Training configuration:\n{config}")

    # Set seed and device
    set_seed(config.environment.seed, logger=logger)
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
        # Commented out: Original config-based loss creation
        # criterion = create_loss(config, logger=logger)
        
        # New: Binary Cross Entropy with Logits Loss for multi-label classification
        # This loss combines sigmoid activation and BCE loss for numerical stability
        if config.experiment.apply_pos_weight is True :
            pos_weight = torch.tensor(dataloader.dataset.pos_weight)
            criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
            logger.info(f"Loss function: BCEWithLogitsLoss(pos_weight={pos_weight}) (for binary classification)")
        else :
            criterion = nn.BCEWithLogitsLoss()
            logger.info("Loss function: BCEWithLogitsLoss (for binary classification)")

        logger.info("Model outputs logits - sigmoid will be applied during loss computation")
        
        # Contrastive loss for multimodal alignment
        contrastive_type = config.training.get('contrastive_type', 'infonce')  # 'infonce' or 'mse'
        lambda_contrastive = config.training.get('lambda_contrastive', 0.1)
        
        if contrastive_type.lower() == 'mse':
            # MSE-based consistency loss (better for small batch sizes)
            contrastive_criterion = MultiModalMSELoss(reduction='mean').to(device)
            logger.info(f"Contrastive Loss: MultiModalMSELoss (MSE-based consistency)")
            logger.info(f"  - Direct feature alignment (no negative samples)")
            logger.info(f"  - Suitable for small batch sizes")
        else:  # infonce (default)
            # InfoNCE contrastive loss (requires sufficient batch size)
            contrastive_temperature = config.training.get('contrastive_temperature', 0.3)
            contrastive_criterion = MultiModalContrastiveLoss(
                temperature=contrastive_temperature,
                normalize=True
            ).to(device)
            logger.info(f"Contrastive Loss: MultiModalContrastiveLoss (InfoNCE)")
            logger.info(f"  - Temperature: {contrastive_temperature}")
            logger.info(f"  - Uses in-batch negative samples")
            logger.info(f"  - Batch size: {config.experiment.batch_size}")
        
        logger.info(f"Lambda contrastive: {lambda_contrastive}")
        
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
    epoch_cls_losses = []
    epoch_cont_losses = []
    epoch_cosine_sims = []
    training_history = []

    try:
        while epochs < config.training.num_epochs:
            epoch_start_time = time.time()
            running_loss = 0.0
            running_cls_loss = 0.0
            running_cont_loss = 0.0
            running_cosine_sim = 0.0
            running_accuracy = 0.0  # 추가: accuracy 누적
            batch_count = 0
            epoch_loss_sum = 0.0
            epoch_cls_loss_sum = 0.0
            epoch_cont_loss_sum = 0.0
            epoch_cosine_sim_sum = 0.0
            epoch_accuracy_sum = 0.0  # 추가: epoch accuracy 누적

            logger.info(f"Starting epoch {epochs + 1}/{config.training.num_epochs}")

            for i, data_dict in enumerate(dataloader):
                try:
                    # Training step
                    train_dict = train_step(
                        model, data_dict, criterion, 
                        contrastive_criterion, lambda_contrastive,
                        optimizer, device
                    )
                    current_loss = train_dict['loss']
                    current_cls_loss = train_dict['cls_loss']
                    current_cont_loss = train_dict['cont_loss']
                    current_cosine_sim = train_dict['cosine_sim']
                    current_accuracy = train_dict['accuracy']  # 추가
                    
                    iterations += 1
                    running_loss += current_loss
                    running_cls_loss += current_cls_loss
                    running_cont_loss += current_cont_loss
                    running_cosine_sim += current_cosine_sim
                    running_accuracy += current_accuracy  # 추가
                    epoch_loss_sum += current_loss
                    epoch_cls_loss_sum += current_cls_loss
                    epoch_cont_loss_sum += current_cont_loss
                    epoch_cosine_sim_sum += current_cosine_sim
                    epoch_accuracy_sum += current_accuracy  # 추가
                    batch_count += 1

                    # Progress reporting
                    if iterations % config.training.report_freq == 0:
                        avg_loss = running_loss / config.training.report_freq
                        avg_cls_loss = running_cls_loss / config.training.report_freq
                        avg_cont_loss = running_cont_loss / config.training.report_freq
                        avg_cosine_sim = running_cosine_sim / config.training.report_freq
                        avg_accuracy = running_accuracy / config.training.report_freq  # 추가
                        elapsed_time = time.time() - epoch_start_time
                        progress = (i + 1) / len(dataloader) * 100
                        
                        logger.info(
                            f"[Epoch {epochs + 1}, Batch {i + 1}/{len(dataloader)}, "
                            f"Iteration {iterations}] "
                            f"total_loss: {avg_loss:.6f} | "
                            f"cls_loss: {avg_cls_loss:.6f} | "
                            f"cont_loss: {avg_cont_loss:.6f} | "
                            f"cosine_sim: {avg_cosine_sim:.4f} | "
                            f"accuracy: {avg_accuracy:.4f} | "
                            f"Time: {elapsed_time:.2f}s | Progress: {progress:.1f}%"
                        )

                        # Commented out: Save training snapshot plot
                        """
                        try:
                            plot_path = f"{config.snapshot_dir}/iteration_{iterations}_epoch_{epochs+1}"
                            plot_title = f'Training Progress - Iteration {iterations}, Epoch {epochs + 1}'
                            save_plot(
                                targets=train_dict['targets'][0], 
                                outputs=train_dict['outputs'][0],
                                target_variables=config.target_variables, 
                                stat_dict=dataloader.dataset.stat_dict,
                                plot_path=plot_path, 
                                plot_title=plot_title, 
                                logger=logger
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save training snapshot: {e}")
                        """

                        running_loss = 0.0
                        running_cls_loss = 0.0
                        running_cont_loss = 0.0
                        running_cosine_sim = 0.0
                        running_accuracy = 0.0  # 추가: 리셋
                        
                except Exception as e:
                    logger.warning(f"Training step failed for batch {i}: {e}")
                    continue

            # End of epoch processing
            epochs += 1
            
            if batch_count > 0:
                epoch_avg_loss = epoch_loss_sum / batch_count
                epoch_avg_cls_loss = epoch_cls_loss_sum / batch_count
                epoch_avg_cont_loss = epoch_cont_loss_sum / batch_count
                epoch_avg_cosine_sim = epoch_cosine_sim_sum / batch_count
                epoch_avg_accuracy = epoch_accuracy_sum / batch_count  # 추가
                
                epoch_losses.append(epoch_avg_loss)
                epoch_cls_losses.append(epoch_avg_cls_loss)
                epoch_cont_losses.append(epoch_avg_cont_loss)
                epoch_cosine_sims.append(epoch_avg_cosine_sim)
                
                # Learning rate scheduling
                scheduler.step(epoch_avg_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                epoch_duration = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epochs} completed in {epoch_duration:.2f}s | "
                    f"Total loss: {epoch_avg_loss:.6f} | "
                    f"Cls loss: {epoch_avg_cls_loss:.6f} | "
                    f"Cont loss: {epoch_avg_cont_loss:.6f} | "
                    f"Cosine sim: {epoch_avg_cosine_sim:.4f} | "
                    f"Accuracy: {epoch_avg_accuracy:.4f} | "
                    f"LR: {current_lr:.8f}"
                )
                
                # Save training history
                training_history.append({
                    'epoch': epochs,
                    'loss': epoch_avg_loss,
                    'cls_loss': epoch_avg_cls_loss,
                    'cont_loss': epoch_avg_cont_loss,
                    'cosine_sim': epoch_avg_cosine_sim,
                    'accuracy': epoch_avg_accuracy,  # 추가
                    'learning_rate': current_lr,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Save best model (loss 기준)
                if epoch_avg_loss < best_loss:
                    best_loss = epoch_avg_loss
                    best_model_path = f"{checkpoint_dir}/best_model.pth"
                    save_checkpoint(model, optimizer, epochs, epoch_avg_loss, best_model_path, logger)
                    logger.info(f"New best model saved with loss: {best_loss:.6f}")

            # Periodic model saving
            if epochs % config.training.model_save_freq == 0:
                model_path = f"{checkpoint_dir}/model_epoch{epochs}.pth"
                save_checkpoint(model, optimizer, epochs, epoch_avg_loss, model_path, logger)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Final model save
    final_model_path = f"{checkpoint_dir}/model_final.pth"
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
        history_path = f"{log_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    # Plot training curves
    try:
        if epoch_losses:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Total loss
            axes[0, 0].plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=2)
            axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Classification vs Contrastive Loss
            axes[0, 1].plot(range(1, len(epoch_cls_losses) + 1), epoch_cls_losses, 'r-', 
                           linewidth=2, label='Classification Loss')
            axes[0, 1].plot(range(1, len(epoch_cont_losses) + 1), epoch_cont_losses, 'g-', 
                           linewidth=2, label='Contrastive Loss')
            axes[0, 1].set_title('Loss Components', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Cosine Similarity
            axes[1, 0].plot(range(1, len(epoch_cosine_sims) + 1), epoch_cosine_sims, 
                           'm-', linewidth=2)
            axes[1, 0].set_title('Feature Alignment (Cosine Similarity)', 
                                fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Cosine Similarity')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
            axes[1, 0].legend()
            
            # Accuracy
            accuracy_history = [h['accuracy'] for h in training_history]
            axes[1, 1].plot(range(1, len(accuracy_history) + 1), accuracy_history, 
                           'c-', linewidth=2)
            axes[1, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            curve_path = f"{log_dir}/training_curves.png"
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Training curves saved: {curve_path}")
    except Exception as e:
        logger.warning(f"Failed to save training curves: {e}")


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
# from networks import create_model, MultiModalContrastiveLoss
# # from losses import create_loss  # Commented out - using BCEWithLogitsLoss instead


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
#         criterion: Classification loss function.
#         contrastive_criterion: Contrastive loss function for multimodal alignment.
#         lambda_contrastive: Weight for contrastive loss component.
#         optimizer: Optimizer.
#         device: Training device.
        
#     Returns:
#         Dictionary containing losses, accuracy, alignment metrics, and predictions.

#     Raises:
#         RuntimeError: If training step fails.
#     """
#     model.train()
#     try:
#         sdo = data_dict["sdo"].to(device)
#         inputs = data_dict["inputs"].to(device)
#         targets = data_dict["labels"].to(device)
#         # targets = data_dict["targets"].to(device)

#         optimizer.zero_grad()
        
#         # Forward pass with feature extraction
#         outputs, transformer_features, convlstm_features = model(
#             inputs, sdo, return_features=True
#         )
        
#         # Classification loss
#         cls_loss = criterion(outputs, targets)
        
#         # Contrastive loss for multimodal alignment
#         cont_loss = contrastive_criterion(transformer_features, convlstm_features)
        
#         # Combined loss
#         total_loss = cls_loss + lambda_contrastive * cont_loss
        
#         total_loss.backward()
        
#         # Gradient clipping to prevent exploding gradients
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()

#         # Calculate accuracy and alignment metrics
#         with torch.no_grad():
#             # Convert logits to probabilities and then to binary predictions
#             probs = torch.sigmoid(outputs)
#             predictions = (probs > 0.5).float()
            
#             # Calculate accuracy (correct predictions / total predictions)
#             accuracy = (predictions == targets).float().mean().item()
            
#             # Calculate feature alignment (cosine similarity)
#             cosine_sim = F.cosine_similarity(
#                 transformer_features, convlstm_features, dim=1
#             ).mean().item()

#         return {
#             'loss': total_loss.item(),
#             'cls_loss': cls_loss.item(),
#             'cont_loss': cont_loss.item(),
#             'cosine_sim': cosine_sim,
#             'accuracy': accuracy,
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
        
#         # New: Binary Cross Entropy with Logits Loss for multi-label classification
#         # This loss combines sigmoid activation and BCE loss for numerical stability
#         if config.experiment.apply_pos_weight is True :
#             pos_weight = torch.tensor(dataloader.dataset.pos_weight)
#             criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
#             logger.info(f"Loss function: BCEWithLogitsLoss(pos_weight={pos_weight}) (for binary classification)")
#         else :
#             criterion = nn.BCEWithLogitsLoss()
#             logger.info("Loss function: BCEWithLogitsLoss (for binary classification)")

#         logger.info("Model outputs logits - sigmoid will be applied during loss computation")
        
#         # Contrastive loss for multimodal alignment
#         contrastive_temperature = config.training.get('contrastive_temperature', 0.3)
#         lambda_contrastive = config.training.get('lambda_contrastive', 0.1)
        
#         contrastive_criterion = MultiModalContrastiveLoss(
#             temperature=contrastive_temperature,
#             normalize=True
#         ).to(device)
        
#         logger.info(f"Contrastive Loss: MultiModalContrastiveLoss(temperature={contrastive_temperature}, normalize=True)")
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
#     epoch_cls_losses = []
#     epoch_cont_losses = []
#     epoch_cosine_sims = []
#     training_history = []

#     try:
#         while epochs < config.training.num_epochs:
#             epoch_start_time = time.time()
#             running_loss = 0.0
#             running_cls_loss = 0.0
#             running_cont_loss = 0.0
#             running_cosine_sim = 0.0
#             running_accuracy = 0.0  # 추가: accuracy 누적
#             batch_count = 0
#             epoch_loss_sum = 0.0
#             epoch_cls_loss_sum = 0.0
#             epoch_cont_loss_sum = 0.0
#             epoch_cosine_sim_sum = 0.0
#             epoch_accuracy_sum = 0.0  # 추가: epoch accuracy 누적

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
#                     current_cls_loss = train_dict['cls_loss']
#                     current_cont_loss = train_dict['cont_loss']
#                     current_cosine_sim = train_dict['cosine_sim']
#                     current_accuracy = train_dict['accuracy']  # 추가
                    
#                     iterations += 1
#                     running_loss += current_loss
#                     running_cls_loss += current_cls_loss
#                     running_cont_loss += current_cont_loss
#                     running_cosine_sim += current_cosine_sim
#                     running_accuracy += current_accuracy  # 추가
#                     epoch_loss_sum += current_loss
#                     epoch_cls_loss_sum += current_cls_loss
#                     epoch_cont_loss_sum += current_cont_loss
#                     epoch_cosine_sim_sum += current_cosine_sim
#                     epoch_accuracy_sum += current_accuracy  # 추가
#                     batch_count += 1

#                     # Progress reporting
#                     if iterations % config.training.report_freq == 0:
#                         avg_loss = running_loss / config.training.report_freq
#                         avg_cls_loss = running_cls_loss / config.training.report_freq
#                         avg_cont_loss = running_cont_loss / config.training.report_freq
#                         avg_cosine_sim = running_cosine_sim / config.training.report_freq
#                         avg_accuracy = running_accuracy / config.training.report_freq  # 추가
#                         elapsed_time = time.time() - epoch_start_time
#                         progress = (i + 1) / len(dataloader) * 100
                        
#                         logger.info(
#                             f"[Epoch {epochs + 1}, Batch {i + 1}/{len(dataloader)}, "
#                             f"Iteration {iterations}] "
#                             f"total_loss: {avg_loss:.6f} | "
#                             f"cls_loss: {avg_cls_loss:.6f} | "
#                             f"cont_loss: {avg_cont_loss:.6f} | "
#                             f"cosine_sim: {avg_cosine_sim:.4f} | "
#                             f"accuracy: {avg_accuracy:.4f} | "
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
#                         running_cls_loss = 0.0
#                         running_cont_loss = 0.0
#                         running_cosine_sim = 0.0
#                         running_accuracy = 0.0  # 추가: 리셋
                        
#                 except Exception as e:
#                     logger.warning(f"Training step failed for batch {i}: {e}")
#                     continue

#             # End of epoch processing
#             epochs += 1
            
#             if batch_count > 0:
#                 epoch_avg_loss = epoch_loss_sum / batch_count
#                 epoch_avg_cls_loss = epoch_cls_loss_sum / batch_count
#                 epoch_avg_cont_loss = epoch_cont_loss_sum / batch_count
#                 epoch_avg_cosine_sim = epoch_cosine_sim_sum / batch_count
#                 epoch_avg_accuracy = epoch_accuracy_sum / batch_count  # 추가
                
#                 epoch_losses.append(epoch_avg_loss)
#                 epoch_cls_losses.append(epoch_avg_cls_loss)
#                 epoch_cont_losses.append(epoch_avg_cont_loss)
#                 epoch_cosine_sims.append(epoch_avg_cosine_sim)
                
#                 # Learning rate scheduling
#                 scheduler.step(epoch_avg_loss)
#                 current_lr = optimizer.param_groups[0]['lr']
                
#                 epoch_duration = time.time() - epoch_start_time
#                 logger.info(
#                     f"Epoch {epochs} completed in {epoch_duration:.2f}s | "
#                     f"Total loss: {epoch_avg_loss:.6f} | "
#                     f"Cls loss: {epoch_avg_cls_loss:.6f} | "
#                     f"Cont loss: {epoch_avg_cont_loss:.6f} | "
#                     f"Cosine sim: {epoch_avg_cosine_sim:.4f} | "
#                     f"Accuracy: {epoch_avg_accuracy:.4f} | "
#                     f"LR: {current_lr:.8f}"
#                 )
                
#                 # Save training history
#                 training_history.append({
#                     'epoch': epochs,
#                     'loss': epoch_avg_loss,
#                     'cls_loss': epoch_avg_cls_loss,
#                     'cont_loss': epoch_avg_cont_loss,
#                     'cosine_sim': epoch_avg_cosine_sim,
#                     'accuracy': epoch_avg_accuracy,  # 추가
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
#             axes[0, 1].plot(range(1, len(epoch_cls_losses) + 1), epoch_cls_losses, 'r-', 
#                            linewidth=2, label='Classification Loss')
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
            
#             # Accuracy
#             accuracy_history = [h['accuracy'] for h in training_history]
#             axes[1, 1].plot(range(1, len(accuracy_history) + 1), accuracy_history, 
#                            'c-', linewidth=2)
#             axes[1, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
#             axes[1, 1].set_xlabel('Epoch')
#             axes[1, 1].set_ylabel('Accuracy')
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


# # import os
# # import sys
# # import time
# # import argparse
# # from datetime import datetime
# # from multiprocessing import freeze_support
# # from pathlib import Path
# # from typing import Dict, Any, Optional

# # import torch
# # import torch.optim as optim
# # import torch.nn as nn
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import hydra

# # from utils import set_seed, setup_logger, setup_device, save_plot
# # from pipeline import create_dataloader
# # from networks import create_model
# # # from losses import create_loss  # Commented out - using BCEWithLogitsLoss instead


# # def train_step(model: torch.nn.Module, data_dict: Dict[str, torch.Tensor], 
# #                criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
# #                device: torch.device) -> Dict[str, Any]:
# #     """Perform a single training step.
    
# #     Args:
# #         model: PyTorch model.
# #         data_dict: Dictionary containing input data.
# #         criterion: Loss function.
# #         optimizer: Optimizer.
# #         device: Training device.
        
# #     Returns:
# #         Dictionary containing loss, accuracy, and prediction results.

# #     Raises:
# #         RuntimeError: If training step fails.
# #     """
# #     model.train()
# #     try:
# #         sdo = data_dict["sdo"].to(device)
# #         inputs = data_dict["inputs"].to(device)
# #         targets = data_dict["labels"].to(device)
# #         # targets = data_dict["targets"].to(device)

# #         optimizer.zero_grad()
# #         outputs = model(inputs, sdo)
# #         loss = criterion(outputs, targets)
# #         loss.backward()
        
# #         # Gradient clipping to prevent exploding gradients
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
# #         optimizer.step()

# #         # Calculate accuracy for binary classification
# #         with torch.no_grad():
# #             # Convert logits to probabilities and then to binary predictions
# #             probs = torch.sigmoid(outputs)
# #             predictions = (probs > 0.5).float()
            
# #             # Calculate accuracy (correct predictions / total predictions)
# #             accuracy = (predictions == targets).float().mean().item()

# #         return {
# #             'loss': loss.item(),
# #             'accuracy': accuracy,
# #             'targets': targets.cpu().detach().numpy(),
# #             'outputs': outputs.cpu().detach().numpy()
# #         }

# #     except Exception as e:
# #         raise RuntimeError(f"Training step failed: {e}")


# # def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
# #                    epoch: int, loss: float, filepath: str,
# #                    logger: Optional = None) -> None:
# #     """Save model checkpoint with optimizer state.
    
# #     Args:
# #         model: PyTorch model.
# #         optimizer: Optimizer.
# #         epoch: Current epoch number.
# #         loss: Current loss value.
# #         filepath: Path to save checkpoint.
# #         logger: Optional logger for output.
# #     """
# #     try:
# #         checkpoint = {
# #             'epoch': epoch,
# #             'model_state_dict': model.state_dict(),
# #             'optimizer_state_dict': optimizer.state_dict(),
# #             'loss': loss,
# #             'timestamp': datetime.now().isoformat()
# #         }
        
# #         # Ensure directory exists
# #         os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
# #         torch.save(checkpoint, filepath)
        
# #         message = f"Checkpoint saved: {filepath}"
# #         if logger:
# #             logger.info(message)
# #         else:
# #             print(message)
            
# #     except Exception as e:
# #         error_msg = f"Failed to save checkpoint {filepath}: {e}"
# #         if logger:
# #             logger.error(error_msg)
# #         else:
# #             print(f"Error: {error_msg}")


# # @hydra.main(config_path="./configs", version_base=None)
# # def main(config):
# #     """Run training process with comprehensive monitoring and error handling.
    
# #     Args:
# #         config: Configuration object containing training parameters.
        
# #     Raises:
# #         RuntimeError: If training setup or execution fails.
# #     """
# #     # Setup logging

# #     save_root = config.environment.save_root
# #     experiment_name = config.experiment.experiment_name
# #     experiment_dir = f"{save_root}/{experiment_name}"
# #     checkpoint_dir = f"{experiment_dir}/checkpoint"
# #     log_dir = f"{experiment_dir}/log"
# #     snapshot_dir = f"{experiment_dir}/snapshot"
# #     validation_dir = f"{experiment_dir}/validation"
# #     for directory in [experiment_dir, checkpoint_dir, log_dir, snapshot_dir, validation_dir]:
# #         if not os.path.exists(directory):
# #             os.makedirs(directory)

# #     logger = setup_logger(__name__, log_dir=log_dir)
# #     logger.info(f"Training configuration:\n{config}")

# #     # Set seed and device
# #     set_seed(config.environment.seed, logger=logger)
# #     device = setup_device(config, logger=logger)
    
# #     # Create dataloader
# #     try:
# #         dataloader = create_dataloader(config, logger=logger)
# #     except Exception as e:
# #         raise RuntimeError(f"Failed to create dataloader: {e}")
    
# #     # Create model
# #     try:
# #         model = create_model(config, logger=logger)
# #         model.to(device)
        
# #         # Log model parameters
# #         total_params = sum(p.numel() for p in model.parameters())
# #         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #         logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
# #     except Exception as e:
# #         raise RuntimeError(f"Failed to create model: {e}")
    
# #     # Create loss function and optimizer
# #     try:
# #         # Commented out: Original config-based loss creation
# #         # criterion = create_loss(config, logger=logger)
        
# #         # New: Binary Cross Entropy with Logits Loss for multi-label classification
# #         # This loss combines sigmoid activation and BCE loss for numerical stability
# #         if config.experiment.apply_pos_weight is True :
# #             pos_weight = torch.tensor(dataloader.dataset.pos_weight)
# #             criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
# #             logger.info(f"Loss function: BCEWithLogitsLoss(pos_weight={pos_weight}) (for binary classification)")
# #         else :
# #             criterion = nn.BCEWithLogitsLoss()
# #             logger.info("Loss function: BCEWithLogitsLoss (for binary classification)")

# #         logger.info("Model outputs logits - sigmoid will be applied during loss computation")
        
# #         optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
        
# #         # Optional: Learning rate scheduler
# #         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# #             optimizer, mode='min', factor=0.5, patience=10
# #         )
        
# #     except Exception as e:
# #         raise RuntimeError(f"Failed to setup training components: {e}")

# #     # Training initialization
# #     start_time = datetime.now()
# #     logger.info("=" * 50)
# #     logger.info("Training Started")
# #     logger.info(f"Total epochs: {config.training.num_epochs}")
# #     logger.info(f"Batches per epoch: {len(dataloader)}")
# #     logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
# #     logger.info("=" * 50)
    
# #     # Training tracking variables
# #     iterations = 0
# #     epochs = 0
# #     best_loss = float('inf')
# #     epoch_losses = []
# #     training_history = []

# #     try:
# #         while epochs < config.training.num_epochs:
# #             epoch_start_time = time.time()
# #             running_loss = 0.0
# #             running_accuracy = 0.0  # 추가: accuracy 누적
# #             batch_count = 0
# #             epoch_loss_sum = 0.0
# #             epoch_accuracy_sum = 0.0  # 추가: epoch accuracy 누적

# #             logger.info(f"Starting epoch {epochs + 1}/{config.training.num_epochs}")

# #             for i, data_dict in enumerate(dataloader):
# #                 try:
# #                     # Training step
# #                     train_dict = train_step(model, data_dict, criterion, optimizer, device)
# #                     current_loss = train_dict['loss']
# #                     current_accuracy = train_dict['accuracy']  # 추가
                    
# #                     iterations += 1
# #                     running_loss += current_loss
# #                     running_accuracy += current_accuracy  # 추가
# #                     epoch_loss_sum += current_loss
# #                     epoch_accuracy_sum += current_accuracy  # 추가
# #                     batch_count += 1

# #                     # Progress reporting
# #                     if iterations % config.training.report_freq == 0:
# #                         avg_loss = running_loss / config.training.report_freq
# #                         avg_accuracy = running_accuracy / config.training.report_freq  # 추가
# #                         elapsed_time = time.time() - epoch_start_time
# #                         progress = (i + 1) / len(dataloader) * 100
                        
# #                         logger.info(
# #                             f"[Epoch {epochs + 1}, Batch {i + 1}/{len(dataloader)}, "
# #                             f"Iteration {iterations}] loss: {avg_loss:.6f} | "
# #                             f"accuracy: {avg_accuracy:.4f} | "  # 추가
# #                             f"Time: {elapsed_time:.2f}s | Progress: {progress:.1f}%"
# #                         )

# #                         # Commented out: Save training snapshot plot
# #                         """
# #                         try:
# #                             plot_path = f"{config.snapshot_dir}/iteration_{iterations}_epoch_{epochs+1}"
# #                             plot_title = f'Training Progress - Iteration {iterations}, Epoch {epochs + 1}'
# #                             save_plot(
# #                                 targets=train_dict['targets'][0], 
# #                                 outputs=train_dict['outputs'][0],
# #                                 target_variables=config.target_variables, 
# #                                 stat_dict=dataloader.dataset.stat_dict,
# #                                 plot_path=plot_path, 
# #                                 plot_title=plot_title, 
# #                                 logger=logger
# #                             )
# #                         except Exception as e:
# #                             logger.warning(f"Failed to save training snapshot: {e}")
# #                         """

# #                         running_loss = 0.0
# #                         running_accuracy = 0.0  # 추가: 리셋
                        
# #                 except Exception as e:
# #                     logger.warning(f"Training step failed for batch {i}: {e}")
# #                     continue

# #             # End of epoch processing
# #             epochs += 1
            
# #             if batch_count > 0:
# #                 epoch_avg_loss = epoch_loss_sum / batch_count
# #                 epoch_avg_accuracy = epoch_accuracy_sum / batch_count  # 추가
# #                 epoch_losses.append(epoch_avg_loss)
                
# #                 # Learning rate scheduling
# #                 scheduler.step(epoch_avg_loss)
# #                 current_lr = optimizer.param_groups[0]['lr']
                
# #                 epoch_duration = time.time() - epoch_start_time
# #                 logger.info(
# #                     f"Epoch {epochs} completed in {epoch_duration:.2f}s | "
# #                     f"Average loss: {epoch_avg_loss:.6f} | "
# #                     f"Average accuracy: {epoch_avg_accuracy:.4f} | "  # 추가
# #                     f"LR: {current_lr:.8f}"
# #                 )
                
# #                 # Save training history
# #                 training_history.append({
# #                     'epoch': epochs,
# #                     'loss': epoch_avg_loss,
# #                     'accuracy': epoch_avg_accuracy,  # 추가
# #                     'learning_rate': current_lr,
# #                     'timestamp': datetime.now().isoformat()
# #                 })
                
# #                 # Save best model (loss 기준)
# #                 if epoch_avg_loss < best_loss:
# #                     best_loss = epoch_avg_loss
# #                     best_model_path = f"{checkpoint_dir}/best_model.pth"
# #                     save_checkpoint(model, optimizer, epochs, epoch_avg_loss, best_model_path, logger)
# #                     logger.info(f"New best model saved with loss: {best_loss:.6f}")

# #             # Periodic model saving
# #             if epochs % config.training.model_save_freq == 0:
# #                 model_path = f"{checkpoint_dir}/model_epoch{epochs}.pth"
# #                 save_checkpoint(model, optimizer, epochs, epoch_avg_loss, model_path, logger)

# #     except KeyboardInterrupt:
# #         logger.info("Training interrupted by user")
# #     except Exception as e:
# #         logger.error(f"Training failed: {e}")
# #         raise

# #     # Final model save
# #     final_model_path = f"{checkpoint_dir}/model_final.pth"
# #     final_loss = epoch_losses[-1] if epoch_losses else float('inf')
# #     save_checkpoint(model, optimizer, epochs, final_loss, final_model_path, logger)
    
# #     # Training completion
# #     end_time = datetime.now()
# #     total_duration = end_time - start_time
    
# #     logger.info("=" * 50)
# #     logger.info("Training Completed")
# #     logger.info(f"Total epochs: {epochs}")
# #     logger.info(f"Total iterations: {iterations}")
# #     logger.info(f"Best loss: {best_loss:.6f}")
# #     logger.info(f"Final loss: {final_loss:.6f}")
# #     logger.info(f"Total training time: {total_duration}")
# #     logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
# #     logger.info("=" * 50)

# #     # Save training history
# #     try:
# #         import json
# #         history_path = f"{log_dir}/training_history.json"
# #         with open(history_path, 'w') as f:
# #             json.dump(training_history, f, indent=2)
# #         logger.info(f"Training history saved: {history_path}")
# #     except Exception as e:
# #         logger.warning(f"Failed to save training history: {e}")

# #     # Plot training curve
# #     try:
# #         if epoch_losses:
# #             plt.figure(figsize=(10, 6))
# #             plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=2)
# #             plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
# #             plt.xlabel('Epoch', fontsize=12)
# #             plt.ylabel('Loss', fontsize=12)
# #             plt.grid(True, alpha=0.3)
# #             plt.tight_layout()
            
# #             curve_path = f"{log_dir}/training_curve.png"
# #             plt.savefig(curve_path, dpi=150, bbox_inches='tight')
# #             plt.close()
# #             logger.info(f"Training curve saved: {curve_path}")
# #     except Exception as e:
# #         logger.warning(f"Failed to save training curve: {e}")


# # if __name__ == '__main__':
# #     freeze_support()
# #     main()
