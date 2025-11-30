import os
import time
import random
import logging
from datetime import datetime
from typing import Optional, Union, List

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 250104, logger: Optional[logging.Logger] = None) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
        logger: Optional logger for output.
    """
    random.seed(seed)  # For built-in random module
    np.random.seed(seed)  # For numpy
    torch.manual_seed(seed)  # For CPU
    
    if torch.cuda.is_available():  # For GPUs
        torch.cuda.manual_seed(seed)  # For single GPU
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  # For hash-based operations
    
    message = f"Random seed set to: {seed}"
    _log_message(logger, message, logging.INFO)


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


def setup_device(options, logger: Optional[logging.Logger] = None) -> torch.device:
    """Setup computation device with fallback options.
    
    Args:
        options: Configuration object containing device preference.
        logger: Optional logger for output.
        
    Returns:
        Configured torch.device.
    """
    requested_device = options.environment.device
    
    # Check CUDA availability
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            message = f"Using CUDA device: {gpu_name}"
        else:
            device = torch.device('cpu')
            message = "CUDA requested but not available, falling back to CPU"
            _log_message(logger, message, logging.WARNING)
    
    # Check MPS availability (Apple Silicon)
    elif requested_device == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            message = "Using MPS device (Apple Silicon)"
        else:
            device = torch.device('cpu')
            message = "MPS requested but not available, falling back to CPU"
            _log_message(logger, message, logging.WARNING)
    
    # CPU device
    elif requested_device == 'cpu':
        device = torch.device('cpu')
        message = "Using CPU device"
    
    else:
        device = torch.device('cpu')
        message = f"Unknown device '{requested_device}', falling back to CPU"
        _log_message(logger, message, logging.WARNING)
    
    _log_message(logger, f"Device configured: {device}", logging.INFO)
    return device


def load_model(model: torch.nn.Module, checkpoint_path: str, 
              device: torch.device, logger: Optional[logging.Logger] = None) -> torch.nn.Module:
    """Load model from checkpoint with comprehensive error handling.
    
    Args:
        model: PyTorch model instance.
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.
        logger: Optional logger for output.
        
    Returns:
        Loaded model.
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If model loading fails.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict into model
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        message = f"Model loaded successfully from: {checkpoint_path}"
        _log_message(logger, message, logging.INFO)
        
        return model
        
    except Exception as e:
        error_msg = f"Failed to load model from {checkpoint_path}: {e}"
        _log_message(logger, error_msg, logging.ERROR)
        raise RuntimeError(error_msg)


def save_plot(targets: np.ndarray, outputs: np.ndarray, 
              target_variables: List[str], stat_dict: dict,
              plot_path: str, plot_title: str, 
              logger: Optional[logging.Logger] = None) -> None:
    """Save comparison plot and data with improved error handling.

    Args:
        targets: Ground truth values of shape (seq_len, n_vars).
        outputs: Model predictions of shape (seq_len, n_vars).
        target_variables: List of target variable names.
        stat_dict: Dictionary containing statistics for denormalization.
        plot_path: Path to save the plot (without extension).
        plot_title: Title of the plot.
        logger: Optional logger for output.
        
    Raises:
        ValueError: If input shapes don't match or are invalid.
        OSError: If file saving fails.
    """
    # Validate inputs
    if targets.shape != outputs.shape:
        raise ValueError(f"Shape mismatch: targets {targets.shape} != outputs {outputs.shape}")
    
    if targets.shape[1] != len(target_variables):
        raise ValueError(f"Variable count mismatch: got {targets.shape[1]}, expected {len(target_variables)}")
    
    try:
        # Denormalize data
        targets_denorm, outputs_denorm = _denormalize_predictions(
            targets, outputs, target_variables, stat_dict
        )
        
        # Create and save plot
        _create_comparison_plot(
            targets_denorm, outputs_denorm, target_variables, 
            plot_title, f"{plot_path}.png"
        )
        
        # Save data as HDF5
        _save_data_h5(targets_denorm, outputs_denorm, f"{plot_path}.h5")
        
        message = f"Plot and data saved: {plot_path}"
        _log_message(logger, message, logging.DEBUG)
        
    except Exception as e:
        error_msg = f"Failed to save plot {plot_path}: {e}"
        _log_message(logger, error_msg, logging.ERROR)
        raise OSError(error_msg)


def _denormalize_predictions(targets: np.ndarray, outputs: np.ndarray,
                           target_variables: List[str], stat_dict: dict) -> tuple:
    """Denormalize predictions using statistics.
    
    Args:
        targets: Normalized target values.
        outputs: Normalized prediction values.
        target_variables: List of variable names.
        stat_dict: Statistics dictionary.
        
    Returns:
        Tuple of (denormalized_targets, denormalized_outputs).
    """
    zero_clip_variables = {"ap_index"}  # Variables that should be clipped to >= 0
    
    targets_denorm_list = []
    outputs_denorm_list = []

    for idx, variable in enumerate(target_variables):
        # Process targets
        target_var = targets[:, idx:idx+1]
        if variable in stat_dict:
            mean = stat_dict[variable]['mean']
            std = stat_dict[variable]['std']
            target_denorm = (target_var * std) + mean
        else:
            target_denorm = target_var
        
        # Clip if necessary
        if variable in zero_clip_variables:
            target_denorm = np.clip(target_denorm, 0, None)
        targets_denorm_list.append(target_denorm)

        # Process outputs
        output_var = outputs[:, idx:idx+1]
        if variable in stat_dict:
            output_denorm = (output_var * std) + mean
        else:
            output_denorm = output_var
            
        # Clip if necessary
        if variable in zero_clip_variables:
            output_denorm = np.clip(output_denorm, 0, None)
        outputs_denorm_list.append(output_denorm)

    return (np.concatenate(targets_denorm_list, axis=1), 
            np.concatenate(outputs_denorm_list, axis=1))


def _create_comparison_plot(targets: np.ndarray, outputs: np.ndarray,
                          target_variables: List[str], title: str, 
                          save_path: str) -> None:
    """Create and save comparison plot.
    
    Args:
        targets: Denormalized target values.
        outputs: Denormalized output values.
        target_variables: List of variable names.
        title: Plot title.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_variables)))
    
    for idx, (variable, color) in enumerate(zip(target_variables, colors)):
        ax.plot(targets[:, idx], label=f'True {variable}', 
               color=color, linewidth=2, alpha=0.8)
        ax.plot(outputs[:, idx], label=f'Predicted {variable}', 
               color=color, linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _save_data_h5(targets: np.ndarray, outputs: np.ndarray, save_path: str) -> None:
    """Save denormalized data to HDF5 file.
    
    Args:
        targets: Denormalized target values.
        outputs: Denormalized output values.
        save_path: Path to save the HDF5 file.
    """
    with h5py.File(save_path, 'w') as f:
        f.create_dataset("targets", data=targets, compression='gzip')
        f.create_dataset("outputs", data=outputs, compression='gzip')


def _log_message(logger: Optional[logging.Logger], message: str, 
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


def calculate_metrics(targets: np.ndarray, predictions: np.ndarray, 
                     variable_names: List[str]) -> dict:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        targets: Ground truth values of shape (n_samples, seq_len, n_vars).
        predictions: Predicted values of shape (n_samples, seq_len, n_vars).
        variable_names: List of variable names.
        
    Returns:
        Dictionary containing metrics for each variable.
        
    Raises:
        ValueError: If input shapes don't match.
    """
    if targets.shape != predictions.shape:
        raise ValueError(f"Shape mismatch: targets {targets.shape} != predictions {predictions.shape}")
    
    metrics = {}
    
    for var_idx, var_name in enumerate(variable_names):
        target_var = targets[..., var_idx]
        pred_var = predictions[..., var_idx]
        
        # Calculate metrics
        mse = np.mean((target_var - pred_var) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(target_var - pred_var))
        
        # Calculate correlation coefficient
        target_flat = target_var.flatten()
        pred_flat = pred_var.flatten()
        correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
        
        # Calculate R-squared
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics[var_name] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'r_squared': float(r_squared)
        }
    
    return metrics


class WulverSubmitter:
    def __init__(self, config):
        lines = [
            "#!/bin/bash -l",
            ""
        ]
        lines += [f"#SBATCH --output={config["OUT_DIR"]}/%x.%j.out"]
        lines += [f"#SBATCH --error={config["ERR_DIR"]}/%x.%j.err"]
        lines += [f"#SBATCH --partition={config["PARTITION"]}"]
        lines += [f"#SBATCH --nodes={config["NUM_NODE"]}"]
        lines += [f"#SBATCH --ntasks-per-node={config["NUM_CPU_CORE"]}"]
        if config["MIG"] :
            lines += [f"#SBATCH --gres=gpu:a100_10g:{config["NUM_GPU"]}"]
        else :
            lines += [f"#SBATCH --gres=gpu:{config["NUM_GPU"]}"]
        lines += [f"#SBATCH --mem={config["MEM"]:d}M"]
        if config["QOS"] not in ("standard", f"high_{config["PI"]}", "low"):
            raise NameError
        lines += [f"#SBATCH --qos={config["QOS"]}"]
        lines += [f"#SBATCH --account={config["PI"]}"]
        lines += [f"#SBATCH --time={config["TIME"]}"]
        lines += [""]
        lines += ["module purge > /dev/null 2>&1"]
        lines += ["module load wulver # Load slurm, easybuild"]
        lines += ["conda activate ap"]
        self.lines = lines

    def submit(self, job_name, commands, script_path, dry_run=True):
        lines = self.lines.copy()
        lines.insert(2, f"#SBATCH --job-name={job_name}")

        if isinstance(commands, str):
            lines.append(commands)
        elif isinstance(commands, list):
            for command in commands :
                lines.append(command)
        else :
            raise TypeError
        with open(script_path, "w") as f :
            f.write("\n".join(lines))
        time.sleep(1)
        if dry_run :
            return
        else :
            os.system(f"sbatch {script_path}")
            return
    
