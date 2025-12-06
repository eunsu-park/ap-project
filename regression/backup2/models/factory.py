"""
Factory function for creating model instances.

This module provides a convenience function to create MultiModalModel
instances from configuration objects.
"""

from .multimodal import MultiModalModel


def create_model(config, logger=None):
    """Create MultiModalModel instance from configuration.
    
    Args:
        config: Configuration object containing model parameters.
        logger: Optional logger for output.
        
    Returns:
        MultiModalModel instance configured for regression.
    """
    # Calculate number of target groups from target sequence length
    num_input_variables = len(config.data.input_variables)
    num_target_groups = config.data.target_sequence_length  # Use config value
    num_target_variables = len(config.data.target_variables)
    
    message = (f"Enhanced MultiModalModel (Regression) with Transformer, ConvLSTM, "
               f"and CrossModalFusion created. Output shape: (batch, {num_target_groups}, {num_target_variables})")

    if logger:
        logger.info(message)
    else:
        print(message)
        
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
