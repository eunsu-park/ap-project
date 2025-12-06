"""
Logging utilities for setting up loggers.

This module provides functions to configure loggers with both
console and file handlers.
"""

import os
import logging
from datetime import datetime
from typing import Optional


def setup_logger(name: str, log_dir: Optional[str] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """Setup logger with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__).
        log_dir: Directory to save log files. If None, only console output.
        level: Logging level (default: INFO).
        
    Returns:
        Configured logger instance.
        
    Raises:
        OSError: If log directory cannot be created.
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
        try:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'training_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Log file created: {log_file}")
        except OSError as e:
            logger.warning(f"Failed to create log file: {e}")
    
    return logger


def log_message(logger: Optional[logging.Logger], message: str, 
                level: int = logging.INFO) -> None:
    """Helper function for consistent logging.
    
    Args:
        logger: Optional logger instance.
        message: Message to log.
        level: Logging level.
    """
    if logger:
        logger.log(level, message)
    else:
        print(message)
