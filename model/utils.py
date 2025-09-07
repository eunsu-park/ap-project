import os
import random
import logging
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=250104):
    random.seed(seed) # For built-in random module
    np.random.seed(seed) # For numpy
    torch.manual_seed(seed) # For CPU
    if torch.cuda.is_available(): # For GPUs
        torch.cuda.manual_seed(seed)  # For single GPU
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True # Ensure reproducibility
    torch.backends.cudnn.benchmark = False # Disable to ensure reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed) # For hash-based operations
    print(f"Random seed set to: {seed}")


def setup_logger(name: str, log_dir: str = None, level: int = logging.INFO):
    """Setup logger with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory to save log files. If None, only console output.
        level: Logging level (default: INFO)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir is specified)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created: {log_file}")
    
    return logger


class TrainingLogger:
    """Simple training progress logger."""
    
    def __init__(self, logger, report_freq: int = 100):
        """Initialize training logger.
        
        Args:
            logger: Logger instance
            report_freq: Frequency of progress reporting
        """
        self.logger = logger
        self.report_freq = report_freq
        self.start_time = None
        
    def start_training(self, total_epochs: int, batches_per_epoch: int):
        """Log training start information."""
        self.start_time = datetime.now()
        self.logger.info("="*50)
        self.logger.info("Training Started")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Batches per epoch: {batches_per_epoch}")
        self.logger.info("="*50)
        
    def log_progress(self, epoch: int, batch: int, iteration: int, 
                    loss: float, elapsed_time: float):
        """Log training progress."""
        self.logger.info(
            f"[Epoch {epoch}, Batch {batch}, Iteration {iteration}] "
            f"loss: {loss:.3f} | Time: {elapsed_time:.2f}s"
        )
        
    def log_epoch_complete(self, epoch: int, avg_loss: float = None):
        """Log epoch completion."""
        msg = f"Epoch {epoch} completed"
        if avg_loss is not None:
            msg += f" | Average loss: {avg_loss:.3f}"
        self.logger.info(msg)
        
    def log_model_save(self, path: str, epoch: int = None):
        """Log model saving."""
        if epoch:
            self.logger.info(f"Model saved at epoch {epoch}: {path}")
        else:
            self.logger.info(f"Final model saved: {path}")
            
    def finish_training(self):
        """Log training completion."""
        if self.start_time:
            total_time = datetime.now() - self.start_time
            self.logger.info("="*50)
            self.logger.info("Training Completed")
            self.logger.info(f"Total training time: {total_time}")
            self.logger.info("="*50)
