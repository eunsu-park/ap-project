"""
Model package for solar wind prediction.
"""

from .transformer import TransformerEncoderModel
from .convlstm import ConvLSTMModel
from .fusion import CrossModalFusion
from .multimodal import MultiModalModel

from utils import get_logger


def create_model(config):
    """Create MultiModalModel from configuration."""
    logger = get_logger()
    
    num_input_variables = len(config.data.input_variables)
    num_target_groups = config.data.target_sequence_length
    num_target_variables = len(config.data.target_variables)
    
    logger.info(
        f"Creating MultiModalModel: "
        f"Output shape (batch, {num_target_groups}, {num_target_variables})"
    )
    
    return MultiModalModel(
        num_input_variables=num_input_variables,
        input_sequence_length=config.data.input_sequence_length,
        num_target_variables=num_target_variables,
        num_target_groups=num_target_groups,
        transformer_d_model=config.model.transformer_d_model,
        transformer_nhead=config.model.transformer_nhead,
        transformer_num_layers=config.model.transformer_num_layers,
        transformer_dim_feedforward=config.model.transformer_dim_feedforward,
        transformer_dropout=config.model.transformer_dropout,
        convlstm_input_channels=config.model.convlstm_input_channels,
        convlstm_hidden_channels=config.model.convlstm_hidden_channels,
        convlstm_kernel_size=config.model.convlstm_kernel_size,
        convlstm_num_layers=config.model.convlstm_num_layers,
        fusion_num_heads=config.model.fusion_num_heads,
        fusion_dropout=config.model.fusion_dropout
    )


__all__ = [
    'create_model',
    'MultiModalModel',
    'TransformerEncoderModel',
    'ConvLSTMModel',
    'CrossModalFusion',
]