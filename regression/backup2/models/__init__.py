"""
Multimodal neural network models for solar wind prediction.

This package provides neural network components for combining
transformer-based time series processing with ConvLSTM-based
image sequence processing.

Main components:
- MultiModalModel: Main model integrating all components
- create_model: Factory function for model creation
- ConvLSTMCell, ConvLSTMModel: ConvLSTM components
- TransformerEncoderModel: Transformer encoder
- CrossModalFusion: Cross-modal attention fusion

Example usage:
    from model import create_model
    
    # Using the factory function (recommended)
    model = create_model(config, logger)
    
    # Or directly
    from model import MultiModalModel
    
    model = MultiModalModel(
        num_input_variables=num_vars,
        input_sequence_length=seq_len,
        ...
    )
"""

# ConvLSTM components
from .convlstm import ConvLSTMCell, ConvLSTMModel

# Transformer components
from .transformer import PositionalEncoding, TransformerEncoderModel

# Fusion components
from .fusion import CrossModalAttention, CrossModalFusion

# Main model
from .multimodal import MultiModalModel

# Factory function
from .factory import create_model


__all__ = [
    # ConvLSTM
    'ConvLSTMCell',
    'ConvLSTMModel',
    
    # Transformer
    'PositionalEncoding',
    'TransformerEncoderModel',
    
    # Fusion
    'CrossModalAttention',
    'CrossModalFusion',
    
    # Main model (most commonly used)
    'MultiModalModel',
    
    # Factory function (most commonly used)
    'create_model',
]
