"""
ConvLSTM components for spatial-temporal sequence processing.

This module provides ConvLSTM cells and models for processing image sequences
while preserving spatial structure.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for processing spatial-temporal data.
    
    Combines convolutional operations with LSTM gates to process
    spatial-temporal sequences while preserving spatial structure.
    
    Args:
        input_channels: Number of input channels.
        hidden_channels: Number of hidden state channels.
        kernel_size: Size of convolutional kernel.
        bias: Whether to use bias in convolutions.
    """
    
    def __init__(self, input_channels: int, hidden_channels: int, 
                 kernel_size: int = 3, bias: bool = True):
        super().__init__()
        
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Input and hidden channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolutional layers for input-to-hidden and hidden-to-hidden
        self.conv_ih = nn.Conv2d(
            input_channels, 4 * hidden_channels, 
            kernel_size, padding=self.padding, bias=bias
        )
        self.conv_hh = nn.Conv2d(
            hidden_channels, 4 * hidden_channels,
            kernel_size, padding=self.padding, bias=bias
        )
    
    def forward(self, input_tensor: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ConvLSTM cell.
        
        Args:
            input_tensor: Input tensor of shape (batch, channels, height, width).
            hidden_state: Tuple of (hidden, cell) states or None for initialization.
            
        Returns:
            Tuple of (new_hidden, new_cell) states.
        """
        batch_size, _, height, width = input_tensor.size()
        
        # Initialize hidden and cell states if not provided
        if hidden_state is None:
            device = input_tensor.device
            hidden = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            cell = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        else:
            hidden, cell = hidden_state
        
        # Compute convolutions
        conv_ih = self.conv_ih(input_tensor)
        conv_hh = self.conv_hh(hidden)
        combined_conv = conv_ih + conv_hh
        
        # Split into gate components
        i_gate, f_gate, o_gate, g_gate = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        # Apply activations
        input_gate = torch.sigmoid(i_gate)
        forget_gate = torch.sigmoid(f_gate)
        output_gate = torch.sigmoid(o_gate)
        candidate_gate = torch.tanh(g_gate)
        
        # Update cell and hidden states
        new_cell = forget_gate * cell + input_gate * candidate_gate
        new_hidden = output_gate * torch.tanh(new_cell)
        
        return new_hidden, new_cell


class ConvLSTMModel(nn.Module):
    """ConvLSTM model for spatial-temporal sequence processing.
    
    Processes image sequences using ConvLSTM layers to capture
    both spatial and temporal dependencies simultaneously.
    
    Args:
        input_channels: Number of input channels.
        hidden_channels: Number of hidden channels for ConvLSTM.
        kernel_size: Kernel size for ConvLSTM convolutions.
        num_layers: Number of ConvLSTM layers.
        output_dim: Final output dimension.
    """
    
    def __init__(self, input_channels: int, hidden_channels: int = 64,
                 kernel_size: int = 3, num_layers: int = 2, 
                 output_dim: int = 256):
        super().__init__()
        
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Input and hidden channels must be positive")
        if num_layers <= 0:
            raise ValueError("Number of layers must be positive")
        if output_dim <= 0:
            raise ValueError("Output dimension must be positive")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Create ConvLSTM layers
        self.convlstm_layers = nn.ModuleList()
        
        # First layer takes input channels
        self.convlstm_layers.append(
            ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        )
        
        # Subsequent layers take hidden channels
        for _ in range(1, num_layers):
            self.convlstm_layers.append(
                ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
            )
        
        # Spatial pooling and output projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvLSTM layers.
        
        Args:
            x: Input tensor of shape (batch, channels, seq_len, height, width).
            
        Returns:
            Output tensor of shape (batch, output_dim).
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input tensor (batch, channels, seq_len, height, width), got {x.dim()}D tensor")
        
        batch_size, channels, seq_len, height, width = x.size()
        
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {channels}")
        
        # Initialize hidden states for all layers
        hidden_states = [None] * self.num_layers
        
        # Process each time step
        for t in range(seq_len):
            input_frame = x[:, :, t, :, :]  # (batch, channels, height, width)
            
            # Pass through each ConvLSTM layer
            for layer_idx, convlstm_layer in enumerate(self.convlstm_layers):
                if layer_idx == 0:
                    # First layer gets input frame
                    hidden_states[layer_idx] = convlstm_layer(input_frame, hidden_states[layer_idx])
                else:
                    # Subsequent layers get previous layer's hidden state
                    hidden_states[layer_idx] = convlstm_layer(
                        hidden_states[layer_idx - 1][0], hidden_states[layer_idx]
                    )
        
        # Use final hidden state from last layer
        final_hidden = hidden_states[-1][0]  # (batch, hidden_channels, height, width)
        
        # Global average pooling
        pooled = self.global_pool(final_hidden).squeeze(-1).squeeze(-1)  # (batch, hidden_channels)
        
        # Output projection
        output = self.output_projection(pooled)
        
        return output
