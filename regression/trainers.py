"""
Training components for solar wind prediction model.

Contains Trainer, MetricsTracker, and CheckpointManager classes.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class MetricsTracker:
    """Track and aggregate training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for new epoch."""
        self.metrics = {
            'loss': [],
            'reg_loss': [],
            'cont_loss': [],
            'mae': [],
            'rmse': [],
            'cosine_sim': []
        }
    
    def update(self, batch_metrics: Dict[str, float]):
        """Update metrics with batch results.
        
        Args:
            batch_metrics: Dictionary containing metric values from a batch.
        """
        for key in ['loss', 'reg_loss', 'cont_loss', 'mae', 'rmse', 'cosine_sim']:
            if key in batch_metrics:
                self.metrics[key].append(batch_metrics[key])
    
    def get_running_average(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get average of last N samples.
        
        Args:
            last_n: Number of recent samples to average. If None, use all.
            
        Returns:
            Dictionary of averaged metrics.
        """
        avg = {}
        for key, values in self.metrics.items():
            if values:
                data = values[-last_n:] if last_n else values
                avg[key] = float(np.mean(data))
        return avg
    
    def get_epoch_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for the epoch.
        
        Returns:
            Dictionary of statistics (mean, std, min, max) for each metric.
        """
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        return summary


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, logger=None, save_freq: int = 1):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints.
            logger: Optional logger for output.
            save_freq: Frequency (in epochs) to save periodic checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.save_freq = save_freq
        self.best_loss = float('inf')
    
    def save(self, model: nn.Module, optimizer: optim.Optimizer, 
             epoch: int, loss: float, filename: Optional[str] = None):
        """Save checkpoint.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch number.
            loss: Current loss value.
            filename: Optional filename. If None, uses 'model_epoch{epoch}.pth'.
        """
        if filename is None:
            filename = f"model_epoch{epoch}.pth"
        
        filepath = self.checkpoint_dir / filename
        
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, filepath)
            
            message = f"Checkpoint saved: {filepath}"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)
        
        except Exception as e:
            error_msg = f"Failed to save checkpoint {filepath}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"Error: {error_msg}")
    
    def save_if_best(self, model: nn.Module, optimizer: optim.Optimizer,
                     epoch: int, loss: float):
        """Save checkpoint if current loss is best.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch number.
            loss: Current loss value.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.save(model, optimizer, epoch, loss, 'best_model.pth')
            
            message = f"New best model saved with loss: {loss:.6f}"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)
    
    def save_periodic(self, model: nn.Module, optimizer: optim.Optimizer,
                      epoch: int, loss: float):
        """Save periodic checkpoint based on save frequency.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch number.
            loss: Current loss value.
        """
        if epoch % self.save_freq == 0:
            self.save(model, optimizer, epoch, loss)


class Trainer:
    """Trainer for multimodal solar wind prediction model."""
    
    def __init__(
        self,
        config,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        contrastive_criterion: nn.Module,
        device: torch.device,
        logger=None
    ):
        """
        Args:
            config: Configuration object.
            model: PyTorch model.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            criterion: Regression loss function.
            contrastive_criterion: Contrastive loss function.
            device: Device for computation.
            logger: Optional logger for output.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.contrastive_criterion = contrastive_criterion
        self.device = device
        self.logger = logger
        
        # Components
        self.metrics_tracker = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=f"{config.environment.save_root}/{config.experiment.experiment_name}/checkpoint",
            logger=logger,
            save_freq=config.training.model_save_freq
        )
        
        # Training state
        self.current_epoch = 0
        self.total_iterations = 0
        self.training_history = []
        
        # Lambda for contrastive loss
        self.lambda_contrastive = config.training.lambda_contrastive
    
    def train_step(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform a single training step.
        
        Args:
            data_dict: Dictionary containing input data.
            
        Returns:
            Dictionary containing losses and metrics.
        """
        self.model.train()
        
        # Move data to device
        sdo = data_dict["sdo"].to(self.device)
        inputs = data_dict["inputs"].to(self.device)
        targets = data_dict["targets"].to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with feature extraction
        outputs, transformer_features, convlstm_features = self.model(
            inputs, sdo, return_features=True
        )
        
        # Compute losses
        reg_loss = self.criterion(outputs, targets)
        cont_loss = self.contrastive_criterion(transformer_features, convlstm_features)
        total_loss = reg_loss + self.lambda_contrastive * cont_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            mae = F.l1_loss(outputs, targets).item()
            mse = F.mse_loss(outputs, targets).item()
            rmse = np.sqrt(mse)
            
            cosine_sim = F.cosine_similarity(
                transformer_features, convlstm_features, dim=1
            ).mean().item()
        
        return {
            'loss': total_loss.item(),
            'reg_loss': reg_loss.item(),
            'cont_loss': cont_loss.item(),
            'mae': mae,
            'rmse': rmse,
            'cosine_sim': cosine_sim
        }
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train one epoch.
        
        Args:
            dataloader: Training data loader.
            
        Returns:
            Dictionary of epoch-averaged metrics.
        """
        self.metrics_tracker.reset()
        epoch_start_time = time.time()
        
        for batch_idx, data_dict in enumerate(dataloader):
            try:
                # Train step
                batch_metrics = self.train_step(data_dict)
                self.metrics_tracker.update(batch_metrics)
                self.total_iterations += 1
                
                # Log progress
                if (batch_idx + 1) % self.config.training.report_freq == 0:
                    self.log_progress(batch_idx, len(dataloader), epoch_start_time)
            
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Training step failed for batch {batch_idx}: {e}")
                else:
                    print(f"Warning: Batch {batch_idx} failed: {e}")
                continue
        
        # Get epoch summary
        epoch_summary = self.metrics_tracker.get_epoch_summary()
        epoch_metrics = {key: stats['mean'] for key, stats in epoch_summary.items()}
        
        # Learning rate scheduling
        if self.scheduler:
            self.scheduler.step(epoch_metrics.get('loss', 0))
        
        # Log epoch summary
        self.log_epoch_summary(epoch_metrics, time.time() - epoch_start_time)
        
        return epoch_metrics
    
    def log_progress(self, batch_idx: int, total_batches: int, epoch_start_time: float):
        """Log training progress.
        
        Args:
            batch_idx: Current batch index.
            total_batches: Total number of batches.
            epoch_start_time: Start time of epoch.
        """
        avg_metrics = self.metrics_tracker.get_running_average(
            last_n=self.config.training.report_freq
        )
        
        elapsed_time = time.time() - epoch_start_time
        progress = (batch_idx + 1) / total_batches * 100
        
        message = (
            f"[Epoch {self.current_epoch + 1}, "
            f"Batch {batch_idx + 1}/{total_batches}, "
            f"Iter {self.total_iterations}] "
            f"total_loss: {avg_metrics.get('loss', 0):.6f} | "
            f"reg_loss: {avg_metrics.get('reg_loss', 0):.6f} | "
            f"cont_loss: {avg_metrics.get('cont_loss', 0):.6f} | "
            f"cosine_sim: {avg_metrics.get('cosine_sim', 0):.4f} | "
            f"MAE: {avg_metrics.get('mae', 0):.4f} | "
            f"RMSE: {avg_metrics.get('rmse', 0):.4f} | "
            f"Time: {elapsed_time:.2f}s | Progress: {progress:.1f}%"
        )
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def log_epoch_summary(self, metrics: Dict[str, float], duration: float):
        """Log epoch summary.
        
        Args:
            metrics: Dictionary of epoch metrics.
            duration: Epoch duration in seconds.
        """
        current_lr = self.optimizer.param_groups[0]['lr']
        
        message = (
            f"Epoch {self.current_epoch + 1} completed in {duration:.2f}s | "
            f"Total loss: {metrics.get('loss', 0):.6f} | "
            f"Reg loss: {metrics.get('reg_loss', 0):.6f} | "
            f"Cont loss: {metrics.get('cont_loss', 0):.6f} | "
            f"Cosine sim: {metrics.get('cosine_sim', 0):.4f} | "
            f"MAE: {metrics.get('mae', 0):.4f} | "
            f"RMSE: {metrics.get('rmse', 0):.4f} | "
            f"LR: {current_lr:.8f}"
        )
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def fit(self, dataloader, num_epochs: int) -> List[Dict[str, Any]]:
        """Full training loop.
        
        Args:
            dataloader: Training data loader.
            num_epochs: Number of epochs to train.
            
        Returns:
            Training history (list of epoch metrics).
        """
        if self.logger:
            self.logger.info("=" * 50)
            self.logger.info("Starting Training")
            self.logger.info("=" * 50)
        else:
            print("=" * 50)
            print("Starting Training")
            print("=" * 50)
        
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch(dataloader)
            
            # Save to history
            self.training_history.append({
                'epoch': epoch + 1,
                **epoch_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Save best model
            self.checkpoint_manager.save_if_best(
                self.model, self.optimizer, epoch + 1, epoch_metrics.get('loss', 0)
            )
            
            # Save periodic checkpoint
            self.checkpoint_manager.save_periodic(
                self.model, self.optimizer, epoch + 1, epoch_metrics.get('loss', 0)
            )
        
        # Final model save
        self.checkpoint_manager.save(
            self.model, self.optimizer, num_epochs,
            self.training_history[-1].get('loss', 0) if self.training_history else 0,
            'model_final.pth'
        )
        
        # Training completion
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        if self.logger:
            self.logger.info("=" * 50)
            self.logger.info("Training Completed")
            self.logger.info(f"Total epochs: {num_epochs}")
            self.logger.info(f"Total iterations: {self.total_iterations}")
            self.logger.info(f"Best loss: {self.checkpoint_manager.best_loss:.6f}")
            self.logger.info(f"Total training time: {total_duration}")
            self.logger.info("=" * 50)
        else:
            print("=" * 50)
            print("Training Completed")
            print(f"Total epochs: {num_epochs}")
            print(f"Total iterations: {self.total_iterations}")
            print(f"Best loss: {self.checkpoint_manager.best_loss:.6f}")
            print(f"Total training time: {total_duration}")
            print("=" * 50)
        
        return self.training_history


def save_training_history(history: List[Dict[str, Any]], config, logger=None):
    """Save training history to JSON file.
    
    Args:
        history: List of epoch metrics.
        config: Configuration object.
        logger: Optional logger for output.
    """
    log_dir = f"{config.environment.save_root}/{config.experiment.experiment_name}/log"
    os.makedirs(log_dir, exist_ok=True)
    
    history_path = f"{log_dir}/training_history.json"
    
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        message = f"Training history saved: {history_path}"
        if logger:
            logger.info(message)
        else:
            print(message)
    
    except Exception as e:
        error_msg = f"Failed to save training history: {e}"
        if logger:
            logger.warning(error_msg)
        else:
            print(f"Warning: {error_msg}")


def plot_training_curves(history: List[Dict[str, Any]], config, logger=None):
    """Plot and save training curves.
    
    Args:
        history: List of epoch metrics.
        config: Configuration object.
        logger: Optional logger for output.
    """
    if not history:
        return
    
    log_dir = f"{config.environment.save_root}/{config.experiment.experiment_name}/log"
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Extract metrics
        epochs = [h['epoch'] for h in history]
        losses = [h.get('loss', 0) for h in history]
        reg_losses = [h.get('reg_loss', 0) for h in history]
        cont_losses = [h.get('cont_loss', 0) for h in history]
        cosine_sims = [h.get('cosine_sim', 0) for h in history]
        maes = [h.get('mae', 0) for h in history]
        rmses = [h.get('rmse', 0) for h in history]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total loss
        axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Regression vs Contrastive Loss
        axes[0, 1].plot(epochs, reg_losses, 'r-', linewidth=2, label='Regression Loss')
        axes[0, 1].plot(epochs, cont_losses, 'g-', linewidth=2, label='Contrastive Loss')
        axes[0, 1].set_title('Loss Components', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cosine Similarity
        axes[1, 0].plot(epochs, cosine_sims, 'm-', linewidth=2)
        axes[1, 0].set_title('Feature Alignment (Cosine Similarity)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
        axes[1, 0].legend()
        
        # MAE and RMSE
        axes[1, 1].plot(epochs, maes, 'c-', linewidth=2, label='MAE')
        axes[1, 1].plot(epochs, rmses, 'orange', linewidth=2, label='RMSE')
        axes[1, 1].set_title('Regression Metrics', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        curve_path = f"{log_dir}/training_curves.png"
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        message = f"Training curves saved: {curve_path}"
        if logger:
            logger.info(message)
        else:
            print(message)
    
    except Exception as e:
        error_msg = f"Failed to save training curves: {e}"
        if logger:
            logger.warning(error_msg)
        else:
            print(f"Warning: {error_msg}")