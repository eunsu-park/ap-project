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
    a multi-modal model that combines transformer processing of solar wind data with 
    ConvLSTM processing of image sequences.
    
    Attributes:
        seed (int): Random seed for reproducibility. Defaults to 42.
        device (str): Device to use for training ('cuda', 'cpu', or 'mps'). Defaults to 'cuda'.
        experiment_name (str): Name of the experiment for organizing outputs. Defaults to 'default'.
        
        data_root (str): Root directory containing the dataset. Defaults to './data/dataset'.
        save_root (str): Root directory for saving results. Defaults to './results'.
        stat_file_path (str): Path to statistics file. Defaults to './data/statistics.pkl'.
        train_list_path (str): Path to training data list CSV file. Defaults to './data/train_list.csv'.
        validation_list_path (str): Path to validation data list CSV file. Defaults to './data/validation_list.csv'.
        
        input_variables (List[str]): List of input variable names for solar wind data.
        input_sequence_length (int): Length of input sequences. Defaults to 40.
        target_variables (List[str]): List of target variable names to predict.
        target_sequence_length (int): Length of target sequences. Defaults to 24.
        
        # Transformer Model Settings
        transformer_d_model (int): Transformer model dimension. Defaults to 256.
        transformer_nhead (int): Number of attention heads. Defaults to 8.
        transformer_num_layers (int): Number of transformer encoder layers. Defaults to 3.
        transformer_dim_feedforward (int): Dimension of feedforward network. Defaults to 512.
        transformer_dropout (float): Dropout rate for transformer. Defaults to 0.1.
        
        # ConvLSTM Model Settings
        convlstm_input_channels (int): Number of input channels for ConvLSTM. Defaults to 2.
        convlstm_hidden_channels (int): Hidden channels for ConvLSTM. Defaults to 64.
        convlstm_kernel_size (int): Kernel size for ConvLSTM. Defaults to 3.
        convlstm_num_layers (int): Number of ConvLSTM layers. Defaults to 2.
        
        # Cross-Modal Fusion Settings
        fusion_num_heads (int): Number of attention heads for cross-modal fusion. Defaults to 4.
        fusion_dropout (float): Dropout rate for fusion module. Defaults to 0.1.
        
        # Image Settings
        image_size (int): Input image size (square). Defaults to 64.
        
        batch_size (int): Training batch size. Defaults to 4.
        num_workers (int): Number of data loading workers. Defaults to 4.
        phase (str): Training phase ('train', 'val', 'test'). Defaults to 'train'.
        num_epochs (int): Number of training epochs. Defaults to 100.
        
        learning_rate (float): Learning rate for optimizer. Defaults to 2e-4.
        
        report_freq (int): Frequency of progress reporting. Defaults to 100.
        model_save_freq (int): Frequency of model saving. Defaults to 5.
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
    validation_list_path: str = './data/validation_list.csv'
    
    # Model architecture
    ## Input/Output Settings
    input_variables: List[str] = field(default_factory=lambda: [
        'Bx_GSE', 'By_GSM', 'Bz_GSM', 'B_magnitude', 'Flow_speed',
        'Proton_density', 'Temperature', 'Kp_index', 'ap_index',
        'DST_index', 'f107_index', 'R_sunspot'
    ])
    input_sequence_length: int = 40
    target_variables: List[str] = field(default_factory=lambda: ['ap_index', 'DST_index'])
    target_sequence_length: int = 24

    ## Transformer Model Settings
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_num_layers: int = 3
    transformer_dim_feedforward: int = 512
    transformer_dropout: float = 0.1

    ## ConvLSTM Model Settings  
    convlstm_input_channels: int = 2
    convlstm_input_image_frames: int = 20
    convlstm_hidden_channels: int = 64
    convlstm_kernel_size: int = 3
    convlstm_num_layers: int = 2

    ## Cross-Modal Fusion Settings
    fusion_num_heads: int = 4
    fusion_dropout: float = 0.1

    ## Image Settings
    image_size: int = 64

    # Training settings
    batch_size: int = 4
    num_workers: int = 4
    phase: str = 'train'
    num_epochs: int = 100
    
    # Optimization
    loss_type: str = 'mse'
    learning_rate: float = 2e-4
    
    # Logging and saving
    report_freq: int = 100
    model_save_freq: int = 5
        
    # Computed properties (calculated in __post_init__)
    num_input_variables: int = field(init=False)
    num_target_variables: int = field(init=False)

    # Post-defined necessary directories
    checkpoint_dir: str = field(init=False)
    log_dir: str = field(init=False)
    snapshot_dir: str = field(init=False)
    validation_dir: str = field(init=False)
    
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
        self.experiment_dir = f"{self.save_root}/{self.experiment_name}"
        self.checkpoint_dir = f"{self.experiment_dir}/checkpoint"
        self.log_dir = f"{self.experiment_dir}/log"
        self.snapshot_dir = f"{self.experiment_dir}/snapshot"
        self.validation_dir = f"{self.experiment_dir}/validation"

    def make_directories(self):
        """
        Create all necessary directories for saving experiment results.
        
        Creates the directory structure needed for storing checkpoints, logs,
        snapshots, validation results, test results, and tensorboard logs.
        All directories are created recursively if they don't exist.
        """
        directories = [
            self.save_root,
            self.experiment_dir,
            self.checkpoint_dir,
            self.log_dir,
            self.snapshot_dir,
            self.validation_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
    
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
        # Variables validation
        if not self.input_variables or not self.target_variables:
            raise ValueError("Input and target variables lists cannot be empty")
        
        if self.num_input_variables <= 0 or self.num_target_variables <= 0:
            raise ValueError("Number of input and target variables must be positive")
        
        # Sequence length validation
        if self.input_sequence_length <= 0 or self.target_sequence_length <= 0:
            raise ValueError("Input and target sequence lengths must be positive")
        
        # Transformer parameters validation
        if self.transformer_d_model <= 0:
            raise ValueError("Transformer d_model must be positive")
        
        if self.transformer_d_model % self.transformer_nhead != 0:
            raise ValueError("Transformer d_model must be divisible by nhead")
        
        if self.transformer_nhead <= 0 or self.transformer_num_layers <= 0:
            raise ValueError("Transformer nhead and num_layers must be positive")
        
        if not (0.0 <= self.transformer_dropout <= 1.0):
            raise ValueError("Transformer dropout must be between 0 and 1")
        
        # ConvLSTM parameters validation
        if self.convlstm_input_channels <= 0 or self.convlstm_hidden_channels <= 0 or self.convlstm_num_layers <= 0:
            raise ValueError("ConvLSTM channels and num_layers must be positive")
        
        if self.convlstm_kernel_size <= 0 or self.convlstm_kernel_size % 2 == 0:
            raise ValueError("ConvLSTM kernel_size must be positive and odd")
        
        # Image settings validation
        if self.image_size <= 0:
            raise ValueError("Image size must be positive")
        
        # Fusion parameters validation
        if self.fusion_num_heads <= 0:
            raise ValueError("Fusion number of heads must be positive")
        
        if not (0.0 <= self.fusion_dropout <= 1.0):
            raise ValueError("Fusion dropout must be between 0 and 1")
        
        # Training parameters validation
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError("Device must be one of: cpu, cuda, mps")
        
        if self.phase not in ['train', 'validation']:
            raise ValueError("Phase must be one of: train, validation")
        
        # Create statistics file directory if it doesn't exist
        stat_file_parent = Path(self.stat_file_path).parent
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
        
        # Group related fields for better readability
        groups = {
            "Basic Settings": ["seed", "device", "experiment_name"],
            "Data Paths": ["data_root", "save_root", "stat_file_path", "train_list_path", "validation_list_path"],
            "Input/Output Settings": ["input_variables", "input_sequence_length", "target_variables", "target_sequence_length"],
            "Transformer Settings": ["transformer_d_model", "transformer_nhead", "transformer_num_layers", 
                                   "transformer_dim_feedforward", "transformer_dropout"],
            "ConvLSTM Settings": ["convlstm_input_channels", "convlstm_hidden_channels", "convlstm_kernel_size", "convlstm_num_layers"],
            "Fusion Settings": ["fusion_num_heads", "fusion_dropout"],
            "Image Settings": ["image_size"],
            "Training Settings": ["batch_size", "num_workers", "phase", "num_epochs", "loss_type", "learning_rate"],
            "Logging Settings": ["report_freq", "model_save_freq"],
            "Computed Properties": ["num_input_variables", "num_target_variables"]
        }
        
        for group_name, field_names in groups.items():
            lines.append(f"  {group_name}:")
            for field_name in field_names:
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    lines.append(f"    {field_name}: {value}")
        
        return "\n".join(lines)