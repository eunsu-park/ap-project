# python standard library
from typing import Tuple, Optional, Union
import math

# third-party library
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

# custom library


class ConvLSTMCell(nn.Module):    
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


class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
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
            batch_first=True  # (seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        
        # Transpose back: (batch, seq_len, d_model)
        # x = x.transpose(0, 1)
        
        # Global average pooling: (batch, d_model, 1) -> (batch, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Final projection
        x = self.output_projection(x)
        
        return x


class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        if feature_dim <= 0:
            raise ValueError("Feature dimension must be positive")
        if feature_dim % num_heads != 0:
            raise ValueError("Feature dimension must be divisible by number of heads")
        if num_heads <= 0:
            raise ValueError("Number of heads must be positive")
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query_features: torch.Tensor, key_value_features: torch.Tensor) -> torch.Tensor:
        batch_size = query_features.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(query_features)  # (batch, feature_dim)
        K = self.k_proj(key_value_features)  # (batch, feature_dim)
        V = self.v_proj(key_value_features)  # (batch, feature_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch, num_heads, 1, head_dim)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.feature_dim)
        
        # Output projection and residual connection
        output = self.out_proj(attended)
        output = self.norm(output + query_features)
        
        return output


class CrossModalFusion(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Cross-attention layers
        self.transformer_to_convlstm = CrossModalAttention(feature_dim, num_heads, dropout)
        self.convlstm_to_transformer = CrossModalAttention(feature_dim, num_heads, dropout)
        
        # Feature alignment and combination
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        self.combination_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Final normalization
        self.final_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, transformer_features: torch.Tensor, convlstm_features: torch.Tensor) -> torch.Tensor:
        # Cross-attention in both directions
        transformer_attended = self.transformer_to_convlstm(transformer_features, convlstm_features)
        convlstm_attended = self.convlstm_to_transformer(convlstm_features, transformer_features)
        
        # Concatenate attended features
        concatenated = torch.cat([transformer_attended, convlstm_attended], dim=1)
        
        # Compute gating weights
        gate_weights = self.feature_gate(concatenated)
        
        # Weighted combination
        weighted_transformer = gate_weights * transformer_attended
        weighted_convlstm = (1 - gate_weights) * convlstm_attended
        
        # Final combination
        combined = torch.cat([weighted_transformer, weighted_convlstm], dim=1)
        fused_features = self.combination_layer(combined)
        
        # Add residual connection and normalize
        residual = (transformer_features + convlstm_features) / 2
        output = self.final_norm(fused_features + residual)
        
        return output


class MultiModalModel(nn.Module):
    def __init__(
        self, num_input_variables: int, input_sequence_length: int,
        num_target_variables: int, target_sequence_length: int,
        transformer_d_model: int, transformer_nhead: int, transformer_num_layers: int,
        transformer_dim_feedforward: int, transformer_dropout: float,
        convlstm_input_channels: int, convlstm_hidden_channels: int,
        convlstm_kernel_size: int, convlstm_num_layers: int,
        fusion_num_heads: int = 4, fusion_dropout: float = 0.1
        ):
        super().__init__()

        # Validate input parameters
        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and number of groups must be positive")

        # Transformer model for solar wind time series
        self.transformer_model = TransformerEncoderModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout)

        # ConvLSTM model for image sequences
        self.convlstm_model = ConvLSTMModel(
            input_channels=convlstm_input_channels,
            hidden_channels=convlstm_hidden_channels,
            kernel_size=convlstm_kernel_size,
            num_layers=convlstm_num_layers,
            output_dim=transformer_d_model)  # Same dimension as transformer

        # Cross-modal fusion module
        self.cross_modal_fusion = CrossModalFusion(
            feature_dim=transformer_d_model,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout
        )
        
        # Regression head - outputs continuous values (no final activation)
        # Output shape: (batch_size, target_sequence_length, num_target_variables)
        self.regression_head = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(transformer_d_model // 2, target_sequence_length * num_target_variables)
        )

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

    def forward(
        self, 
        solar_wind_input: torch.Tensor, 
        image_input: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Validate input tensors
        if solar_wind_input is None or image_input is None:
            raise ValueError("Both solar_wind_input and image_input must be provided")
        
        if solar_wind_input.size(0) != image_input.size(0):
            raise ValueError(f"Batch sizes must match: solar_wind={solar_wind_input.size(0)}, image={image_input.size(0)}")

        # Extract features from each modality
        transformer_features = self.transformer_model(solar_wind_input)
        convlstm_features = self.convlstm_model(image_input)
        
        # Apply cross-modal fusion
        fused_features = self.cross_modal_fusion(transformer_features, convlstm_features)
        
        # Generate regression predictions
        predictions = self.regression_head(fused_features)
        
        # Reshape to (batch, target_sequence_length, num_target_variables)
        output = predictions.reshape(predictions.size(0), self.target_sequence_length, self.num_target_variables)
        
        # # Optional: Clamp values to valid range [0, 400] during inference
        # if not self.training:
        #     output = torch.clamp(output, min=0, max=400)
        
        # Return features if requested (for contrastive loss)
        if return_features:
            return output, transformer_features, convlstm_features
        
        return output


def create_model(config):
    """Create MultiModalModel from configuration."""
    
    num_input_variables = len(config.data.input_variables)
    input_sequence_length = config.data.input_end_index - config.data.input_start_index

    num_target_variables = len(config.data.target_variables)
    target_sequence_length = config.data.target_end_index - config.data.target_start_index

    print(
        f"Creating MultiModalModel: "
        f"Output shape (batch, {target_sequence_length}, {num_target_variables})"
    )
    
    return MultiModalModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
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


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    model = create_model(config)
    print(model)

    sdo_shape = (
        config.experiment.batch_size,
        len(config.data.sdo_wavelengths),
        config.data.sdo_end_index - config.data.sdo_start_index,
        config.data.sdo_image_size,
        config.data.sdo_image_size
    )

    inputs_shape = (
        config.experiment.batch_size,
        config.data.input_end_index - config.data.input_start_index,
        len(config.data.input_variables)
    )

    targets_shape = (
        config.experiment.batch_size,
        config.data.target_end_index - config.data.target_start_index,
        len(config.data.target_variables)
    )

    sdo = torch.randn(size=sdo_shape)
    print(f"sdo: {sdo.shape}")
    inputs = torch.randn(size=inputs_shape)
    print(f"inputs: {inputs.shape}")
    targets = torch.randn(size=targets_shape)
    print(f"targets: {targets.shape}")

    outputs, transformer_features, convlstm_features = model(inputs, sdo, return_features=True)
    print(f"outputs: {outputs.shape}")
    

if __name__ == "__main__" :
    main()
