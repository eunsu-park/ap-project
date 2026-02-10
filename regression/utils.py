# python standard library
import os
import random
import logging
from typing import List, Optional

# third-party library
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py

# custom library


def setup_seed(seed: int = 250104) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed: {seed}")


def setup_device(requested_device) -> torch.device:
    # CUDA
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            try:
                # Test actual GPU availability (driver may report available but GPU busy)
                torch.zeros(1, device='cuda')
                device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Using CUDA: {gpu_name}")
            except (RuntimeError, torch.cuda.CudaError):
                device = torch.device('cpu')
                print("CUDA device busy/unavailable, falling back to CPU")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    
    # MPS (Apple Silicon)
    elif requested_device == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("MPS not available, using CPU")
    
    # CPU
    elif requested_device == 'cpu':
        device = torch.device('cpu')
        print("Using CPU")
    
    # Unknown
    else:
        device = torch.device('cpu')
        print(f"Unknown device '{requested_device}', using CPU")
    
    return device


def setup_experiment(config):
    setup_seed(config.experiment.seed)
    device = setup_device(config.environment.device)
    return device


def load_model(model: torch.nn.Module, checkpoint_path: str, 
               device: torch.device) -> torch.nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
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
    
    # Load into model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {checkpoint_path}")
    return model


def save_model(model: torch.nn.Module, save_path: str, 
               epoch: Optional[int] = None, optimizer: Optional[torch.optim.Optimizer] = None):

    checkpoint = {
        'model_state_dict': model.state_dict()
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Model saved: {save_path}")


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
        targets_denorm, outputs_denorm = denormalize_predictions(
            targets, outputs, target_variables, stat_dict
        )
        
        # Create and save plot
        create_comparison_plot(
            targets_denorm, outputs_denorm, target_variables, 
            plot_title, f"{plot_path}.png"
        )
        
        # Save data as HDF5
        save_data_h5(targets_denorm, outputs_denorm, f"{plot_path}.h5")
        
        message = f"Plot and data saved: {plot_path}"
        _log_message(logger, message, logging.DEBUG)
        
    except Exception as e:
        error_msg = f"Failed to save plot {plot_path}: {e}"
        _log_message(logger, error_msg, logging.ERROR)
        raise OSError(error_msg)


def denormalize_predictions(targets: np.ndarray, outputs: np.ndarray,
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


def create_comparison_plot(targets: np.ndarray, outputs: np.ndarray,
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


def save_data_h5(targets: np.ndarray, outputs: np.ndarray, save_path: str) -> None:
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
    """Helper function for logging."""
    if logger:
        logger.log(level, message)
    else:
        print(message)
