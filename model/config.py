import argparse
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    """
    Configuration class for multi-modal deep learning model training and evaluation.
    
    This class contains all configuration parameters needed for training and evaluating
    a multi-modal model that combines linear processing of solar wind data with 
    Inception-LSTM processing of image sequences.
    
    Attributes:
        seed (int): Random seed for reproducibility. Defaults to 42.
        device (str): Device to use for training ('cuda', 'cpu', or 'mps'). Defaults to 'cuda'.
        experiment_name (str): Name of the experiment for organizing outputs. Defaults to 'default'.
        
        data_root (str): Root directory containing the dataset. Defaults to './data/dataset'.
        save_root (str): Root directory for saving results. Defaults to './results'.
        stat_file_path (str): Path to statistics file. Defaults to './data/statistics.pkl'.
        train_list_path (str): Path to training data list CSV file. Defaults to './data/train_list.csv'.
        test_list_path (str): Path to testing data list CSV file. Defaults to './data/test_list.csv'.
        
        input_variables (List[str]): List of input variable names for solar wind data.
        input_sequence_length (int): Length of input sequences. Defaults to 40.
        target_variables (List[str]): List of target variable names to predict.
        target_sequence_length (int): Length of target sequences. Defaults to 24.
        num_linear_output (int): Number of output units per linear block. Defaults to 256.
        
        inception_in_channels (int): Number of input channels for inception model. Defaults to 2.
        inception_out_channels (int): Number of output channels for inception model. Defaults to 96.
        inception_in_image_size (int): Input image size (square). Defaults to 64.
        inception_in_image_frames (int): Number of input image frames. Defaults to 20.
        lstm_hidden_size (int): Hidden size of LSTM layer. Defaults to 512.
        
        batch_size (int): Training batch size. Defaults to 4.
        num_workers (int): Number of data loading workers. Defaults to 4.
        phase (str): Training phase ('train', 'val', 'test'). Defaults to 'train'.
        num_epochs (int): Number of training epochs. Defaults to 100.
        
        learning_rate (float): Learning rate for optimizer. Defaults to 2e-4.
        
        report_freq (int): Frequency of progress reporting. Defaults to 100.
        save_freq (int): Frequency of model saving. Defaults to 5.
        test_freq (int): Frequency of testing. Defaults to 1.
    """
    
    # Basic settings
    seed: int = 42
    device: str = 'cuda'
    experiment_name: str = 'default'
    
    # Data paths
    data_root: str = './data/dataset'
    save_root: str = './results'
    stat_file_path: str = './data/statistics.pkl'
    train_list_path: str = './data/train_list.csv'
    test_list_path: str = './data/test_list.csv'
    
    # Model architecture
    ## Linear Model
    input_variables: List[str] = field(default_factory=lambda: [
        'Bx_GSE', 'By_GSM', 'Bz_GSM', 'B_magnitude', 'Flow_speed',
        'Proton_density', 'Temperature', 'Kp_index', 'ap_index',
        'DST_index', 'f107_index', 'R_sunspot'
    ])
    input_sequence_length: int = 40
    target_variables: List[str] = field(default_factory=lambda: ['ap_index', 'DST_index'])
    target_sequence_length: int = 24
    num_linear_output: int = 256

    ## Inception-LSTM Model
    inception_in_channels: int = 2
    inception_out_channels: int = 96
    inception_in_image_size: int = 64
    inception_in_image_frames: int = 20
    lstm_hidden_size: int = 512

    # Training settings
    batch_size: int = 4
    num_workers: int = 4
    phase: str = 'train'
    num_epochs: int = 100
    
    # Optimization
    learning_rate: float = 2e-4
    
    # Logging and saving
    report_freq: int = 100
    save_freq: int = 5
    test_freq: int = 5
        
    # Computed properties (calculated in __post_init__)
    num_input_variables: int = field(init=False)
    num_target_variables: int = field(init=False)

    # Post-defined necessary directories
    checkpoint_dir: str = field(init=False)
    log_dir: str = field(init=False)
    snapshot_dir: str = field(init=False)
    validation_dir: str = field(init=False)
    test_dir: str = field(init=False)
    tensorboard_dir: str = field(init=False)
    
    def __post_init__(self):
        """
        Compute derived parameters and create directory paths after initialization.
        
        This method is automatically called after the dataclass is initialized.
        It computes the number of input and target variables from the provided lists
        and sets up the directory structure for saving results.
        """
        # Compute derived parameters
        self.num_input_variables = len(self.input_variables)
        self.num_target_variables = len(self.target_variables)

        # Define necessary directories
        self.checkpoint_dir = f"{self.save_root}/{self.experiment_name}/checkpoint"
        self.log_dir = f"{self.save_root}/{self.experiment_name}/log"
        self.snapshot_dir = f"{self.save_root}/{self.experiment_name}/snapshot"
        self.validation_dir = f"{self.save_root}/{self.experiment_name}/validation"
        self.test_dir = f"{self.save_root}/{self.experiment_name}/test"
        self.tensorboard_dir = f"{self.save_root}/{self.experiment_name}/tensorboard"

    def make_directories(self):
        """
        Create all necessary directories for saving experiment results.
        
        Creates the directory structure needed for storing checkpoints, logs,
        snapshots, validation results, test results, and tensorboard logs.
        All directories are created recursively if they don't exist.
        """
        # Create necessary directories
        Path(self.save_root).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        Path(self.validation_dir).mkdir(parents=True, exist_ok=True)
        Path(self.test_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path (str): Path to the YAML configuration file.
            
        Returns:
            Config: Configuration instance loaded from YAML file.
            
        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file cannot be parsed.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_args_and_yaml(cls, yaml_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> 'Config':
        """
        Load configuration from YAML file and override with command line arguments.
        
        Args:
            yaml_path (Optional[str]): Path to YAML configuration file. If None or file 
                                     doesn't exist, uses default configuration.
            args (Optional[argparse.Namespace]): Command line arguments namespace. 
                                               Non-None values override YAML settings.
            
        Returns:
            Config: Configuration instance with settings from YAML and command line overrides.
        """
        # Start with default config
        if yaml_path and Path(yaml_path).exists():
            config = cls.from_yaml(yaml_path)
        else:
            config = cls()
        
        # Override with command line arguments
        if args:
            for key, value in vars(args).items():
                if hasattr(config, key) and value is not None:
                    setattr(config, key, value)
        
        return config
    
    def save_yaml(self, path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            path (str): Path where the YAML file should be saved.
            
        Note:
            Only saves fields that are part of the dataclass initialization
            (excludes computed fields like directory paths).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary (exclude computed fields)
        config_dict = {}
        for field_info in self.__dataclass_fields__.values():
            if field_info.init:  # Only include fields that are part of __init__
                config_dict[field_info.name] = getattr(self, field_info.name)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """
        Validate configuration parameters for correctness and consistency.
        
        Checks that all configuration parameters are valid and consistent with
        each other. This includes validating data types, ranges, and dependencies
        between parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
            FileNotFoundError: If required directories cannot be created.
        """
        # Variables validation - ensure input and target variables are properly configured
        if not self.input_variables or not self.target_variables:
            raise ValueError("Input and target variables lists cannot be empty")
        
        if self.num_input_variables <= 0 or self.num_target_variables <= 0:
            raise ValueError("Number of input and target variables must be positive")
        
        # Sequence length validation
        if self.input_sequence_length <= 0 or self.target_sequence_length <= 0:
            raise ValueError("Input and target sequence lengths must be positive")
        
        # Training parameters validation
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError("Device must be one of: cpu, cuda, mps")
        
        # Path validation - only validate paths that are critical and should exist
        # Note: data_root validation removed as it may not exist at config validation time
        stat_file_parent = Path(self.stat_file_path).parent
        if not stat_file_parent.exists():
            # Try to create the directory instead of failing
            try:
                stat_file_parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise FileNotFoundError(f"Cannot create statistics file directory: {stat_file_parent}") from e
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary format.
        
        Returns:
            dict: Dictionary containing all configuration parameters including
                 computed fields like directory paths.
        """
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
    
    def __str__(self) -> str:
        """
        Pretty print configuration for debugging and logging.
        
        Returns:
            str: Formatted string representation of all configuration parameters,
                organized in a human-readable format suitable for logging.
        """
        lines = ["Configuration:"]
        for field_info in self.__dataclass_fields__.values():
            value = getattr(self, field_info.name)
            lines.append(f"  {field_info.name}: {value}")
        return "\n".join(lines)