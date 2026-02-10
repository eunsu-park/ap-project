"""
Centralized configuration for solar wind prediction.

All configuration dataclasses in one place for type safety and consistency.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EnvironmentConfig:
    """Environment and system configuration."""
    seed: int
    device: str  # 'cuda', 'cpu', or 'mps'
    data_root: str
    save_root: str


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Dataset paths
    dataset_dir_name: str
    dataset_name: str
    
    # SDO (image) configuration
    sdo_wavelengths: List[str]
    sdo_image_size: int
    sdo_sequence_length: int
    
    # OMNI (time series) configuration
    input_variables: List[str]
    target_variables: List[str]
    input_sequence_length: int
    target_sequence_length: int
    target_day: int
    split_index: int
    
    # Computed properties
    @property
    def omni_variables(self) -> List[str]:
        """Get all OMNI variables (input + target, deduplicated)."""
        return list(set(self.input_variables + self.target_variables))


@dataclass
class ExperimentConfig:
    """Experiment-specific configuration."""
    experiment_name: str
    phase: str  # 'train' or 'validation'
    batch_size: int
    num_workers: int
    
    # Sampling strategies
    enable_undersampling: bool = False
    num_subsample: int = 1
    subsample_index: int = 0
    enable_oversampling: bool = False
    num_oversample: int = 1


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Transformer
    transformer_d_model: int
    transformer_nhead: int
    transformer_num_layers: int
    transformer_dim_feedforward: int
    transformer_dropout: float
    
    # ConvLSTM
    convlstm_input_channels: int
    convlstm_hidden_channels: int
    convlstm_kernel_size: int
    convlstm_num_layers: int
    
    # Cross-modal fusion
    fusion_num_heads: int
    fusion_dropout: float


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int
    learning_rate: float
    optimizer: str  # 'adam' or 'sgd'
    
    # Loss configuration
    loss_type: str  # 'mse' or 'mae'
    
    # Contrastive loss
    contrastive_type: str  # 'mse' or 'infonce'
    contrastive_temperature: float
    lambda_contrastive: float
    
    # Logging
    report_freq: int
    model_save_freq: int


@dataclass
class ValidationConfig:
    """Validation configuration."""
    checkpoint_path: str
    output_dir: str
    compute_alignment: bool = True
    save_plots: bool = True


@dataclass
class Config:
    """Main configuration container."""
    environment: EnvironmentConfig
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    validation: Optional[ValidationConfig] = None
    
    @classmethod
    def from_hydra(cls, hydra_cfg):
        """
        Convert Hydra DictConfig to structured Config.
        
        Args:
            hydra_cfg: Hydra configuration object
            
        Returns:
            Structured Config object
        """
        # Environment
        env = EnvironmentConfig(
            seed=hydra_cfg.environment.seed,
            device=hydra_cfg.environment.device,
            data_root=hydra_cfg.environment.data_root,
            save_root=hydra_cfg.environment.save_root
        )
        
        # Data
        data = DataConfig(
            dataset_dir_name=hydra_cfg.data.dataset_dir_name,
            dataset_name=hydra_cfg.data.dataset_name,
            sdo_wavelengths=list(hydra_cfg.data.sdo_wavelengths),
            sdo_image_size=hydra_cfg.data.sdo_image_size,
            sdo_sequence_length=hydra_cfg.data.sdo_sequence_length,
            input_variables=list(hydra_cfg.data.input_variables),
            target_variables=list(hydra_cfg.data.target_variables),
            input_sequence_length=hydra_cfg.data.input_sequence_length,
            target_sequence_length=hydra_cfg.data.target_sequence_length,
            target_day=hydra_cfg.data.target_day,
            split_index=hydra_cfg.data.split_index
        )
        
        # Experiment
        experiment = ExperimentConfig(
            experiment_name=hydra_cfg.experiment.experiment_name,
            phase=hydra_cfg.experiment.phase,
            batch_size=hydra_cfg.experiment.batch_size,
            num_workers=hydra_cfg.experiment.num_workers,
            enable_undersampling=hydra_cfg.experiment.enable_undersampling,
            num_subsample=hydra_cfg.experiment.num_subsample,
            subsample_index=hydra_cfg.experiment.subsample_index,
            enable_oversampling=hydra_cfg.experiment.enable_oversampling,
            num_oversample=hydra_cfg.experiment.num_oversample
        )
        
        # Model
        model = ModelConfig(
            transformer_d_model=hydra_cfg.model.transformer_d_model,
            transformer_nhead=hydra_cfg.model.transformer_nhead,
            transformer_num_layers=hydra_cfg.model.transformer_num_layers,
            transformer_dim_feedforward=hydra_cfg.model.transformer_dim_feedforward,
            transformer_dropout=hydra_cfg.model.transformer_dropout,
            convlstm_input_channels=hydra_cfg.model.convlstm_input_channels,
            convlstm_hidden_channels=hydra_cfg.model.convlstm_hidden_channels,
            convlstm_kernel_size=hydra_cfg.model.convlstm_kernel_size,
            convlstm_num_layers=hydra_cfg.model.convlstm_num_layers,
            fusion_num_heads=hydra_cfg.model.fusion_num_heads,
            fusion_dropout=hydra_cfg.model.fusion_dropout
        )
        
        # Training
        training = TrainingConfig(
            num_epochs=hydra_cfg.training.num_epochs,
            learning_rate=hydra_cfg.training.learning_rate,
            optimizer=hydra_cfg.training.optimizer,
            loss_type=hydra_cfg.training.loss_type,
            contrastive_type=hydra_cfg.training.contrastive_type,
            contrastive_temperature=hydra_cfg.training.contrastive_temperature,
            lambda_contrastive=hydra_cfg.training.lambda_contrastive,
            report_freq=hydra_cfg.training.report_freq,
            model_save_freq=hydra_cfg.training.model_save_freq
        )
        
        # Validation (optional)
        validation = None
        if hasattr(hydra_cfg, 'validation'):
            validation = ValidationConfig(
                checkpoint_path=hydra_cfg.validation.checkpoint_path,
                output_dir=hydra_cfg.validation.output_dir,
                compute_alignment=hydra_cfg.validation.get('compute_alignment', True),
                save_plots=hydra_cfg.validation.get('save_plots', True)
            )
        
        return cls(
            environment=env,
            experiment=experiment,
            data=data,
            model=model,
            training=training,
            validation=validation
        )
    
    # Convenience properties for backward compatibility
    @property
    def dataset_path(self) -> str:
        """Get full dataset path."""
        return f"{self.environment.data_root}/{self.data.dataset_name}"
    
    @property
    def train_list_path(self) -> str:
        """Get training list CSV path."""
        return f"{self.dataset_path}_train.csv"
    
    @property
    def validation_list_path(self) -> str:
        """Get validation list CSV path."""
        return f"{self.dataset_path}_validation.csv"
    
    @property
    def stat_file_path(self) -> str:
        """Get statistics pickle file path."""
        return f"{self.dataset_path}_statistics.pkl"
    
    @property
    def experiment_dir(self) -> str:
        """Get experiment directory."""
        return f"{self.environment.save_root}/{self.experiment.experiment_name}"
    
    @property
    def checkpoint_dir(self) -> str:
        """Get checkpoint directory."""
        return f"{self.experiment_dir}/checkpoint"
    
    @property
    def log_dir(self) -> str:
        """Get log directory."""
        return f"{self.experiment_dir}/log"
