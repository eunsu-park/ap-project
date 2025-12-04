"""
Transformer components for time series processing.

This module provides transformer encoder with positional encoding for
processing multivariate time series data.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input sequences.
    
    Adds sinusoidal positional embeddings to input sequences to provide
    temporal information for the transformer model.
    
    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
        dropout: Dropout rate.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    """Transformer encoder model for solar wind time series data.
    
    Processes multivariate time series data using transformer encoder layers
    with positional encoding and multi-head attention mechanism.
    
    Args:
        num_input_variables: Number of input variables (features).
        input_sequence_length: Length of input sequences.
        d_model: Model dimension for transformer.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Dimension of feedforward network.
        dropout: Dropout rate.
    """
    
    def __init__(self, num_input_variables: int, input_sequence_length: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Validate input parameters
        if num_input_variables <= 0:
            raise ValueError(f"Number of input variables must be positive, got {num_input_variables}")
        if input_sequence_length <= 0:
            raise ValueError(f"Input sequence length must be positive, got {input_sequence_length}")
        if d_model <= 0:
            raise ValueError(f"Model dimension must be positive, got {d_model}")
        if d_model % nhead != 0:
            raise ValueError(f"Model dimension {d_model} must be divisible by number of heads {nhead}")
        if nhead <= 0 or num_layers <= 0:
            raise ValueError("Number of heads and layers must be positive")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("Dropout must be between 0 and 1")
        
        self.d_model = d_model
        self.input_sequence_length = input_sequence_length
        self.num_input_variables = num_input_variables
        
        # Input projection layer
        self.input_projection = nn.Linear(num_input_variables, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, input_sequence_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_variables).
            
        Returns:
            Encoded features of shape (batch_size, d_model).
            
        Raises:
            ValueError: If input tensor has invalid dimensions.
        """
        # Validate input tensor dimensions
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (batch, seq_len, variables), got {x.dim()}D tensor")
        
        batch_size, seq_len, num_vars = x.size()
        
        if seq_len != self.input_sequence_length:
            raise ValueError(f"Expected sequence length {self.input_sequence_length}, got {seq_len}")
        if num_vars != self.num_input_variables:
            raise ValueError(f"Expected {self.num_input_variables} variables, got {num_vars}")
        
        # Project to model dimension: (batch, seq_len, d_model)
        x = self.input_projection(x)
        
        # Transpose for transformer: (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        
        # Transpose back: (batch, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # Global average pooling: (batch, d_model, 1) -> (batch, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Final projection
        x = self.output_projection(x)
        
        return x
