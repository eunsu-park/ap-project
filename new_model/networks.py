import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import hydra


# Network architecture constants
CONV_KERNEL_1X1 = 1
CONV_KERNEL_3X3 = (1, 3, 3)
CONV_KERNEL_5X5 = (1, 5, 5)
CONV_KERNEL_7X7 = (1, 7, 7)

PADDING_NONE = 0
PADDING_1 = (0, 1, 1)
PADDING_2 = (0, 2, 2)
PADDING_3 = (0, 3, 3)

STRIDE_1 = 1
STRIDE_2x2 = (1, 2, 2)

POOL_KERNEL_3X3 = (1, 3, 3)
INCEPTION_BRANCHES = 4


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


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for feature fusion.
    
    Enables interaction between transformer and ConvLSTM features
    through mutual attention computation.
    
    Args:
        feature_dim: Dimension of input features.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """
    
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
        """Forward pass of cross-modal attention.
        
        Args:
            query_features: Query features of shape (batch, feature_dim).
            key_value_features: Key-Value features of shape (batch, feature_dim).
            
        Returns:
            Attended features of shape (batch, feature_dim).
        """
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
    """Cross-modal fusion module for combining transformer and ConvLSTM features.
    
    Uses bidirectional cross-attention to enable interaction between
    solar wind (transformer) and image (ConvLSTM) features.
    
    Args:
        feature_dim: Dimension of input features.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """
    
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
        """Forward pass of cross-modal fusion.
        
        Args:
            transformer_features: Features from transformer of shape (batch, feature_dim).
            convlstm_features: Features from ConvLSTM of shape (batch, feature_dim).
            
        Returns:
            Fused features of shape (batch, feature_dim).
        """
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


class MultiModalContrastiveLoss(nn.Module):
    """Contrastive loss for aligning multimodal representations.
    
    Uses InfoNCE-style loss to align transformer (solar wind) and ConvLSTM (image)
    features in the same embedding space. Features from the same sample are treated
    as positive pairs, while features from different samples in the batch serve as
    negative pairs (in-batch negatives).
    
    Args:
        temperature: Temperature parameter for scaling similarity scores.
                    Higher values (0.3-0.5) recommended for small batch sizes.
                    Lower values (0.07-0.1) for large batch sizes.
        normalize: Whether to L2-normalize features before computing similarity.
                  True is strongly recommended for stable training.
    """
    
    def __init__(self, temperature: float = 0.3, normalize: bool = True):
        super().__init__()
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two sets of features.
        
        Args:
            features1: Features from first modality of shape (batch_size, feature_dim).
            features2: Features from second modality of shape (batch_size, feature_dim).
            
        Returns:
            Contrastive loss scalar.
            
        Raises:
            ValueError: If input tensors have mismatched dimensions.
        """
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")
        
        if features1.size(0) != features2.size(0):
            raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")
        
        if features1.size(1) != features2.size(1):
            raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")
        
        batch_size = features1.size(0)
        
        # L2 normalization for cosine similarity
        if self.normalize:
            features1 = F.normalize(features1, p=2, dim=1)
            features2 = F.normalize(features2, p=2, dim=1)
        
        # Concatenate features from both modalities
        # Shape: (2 * batch_size, feature_dim)
        features = torch.cat([features1, features2], dim=0)
        
        # Compute similarity matrix
        # Shape: (2 * batch_size, 2 * batch_size)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for positive pairs
        # For each sample i in features1, its positive pair is sample i in features2
        # which is at index (i + batch_size) in the concatenated tensor
        labels = torch.arange(batch_size, device=features1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Create mask to exclude self-similarity (diagonal elements)
        # We don't want to compare a sample with itself
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Compute cross-entropy loss
        # This encourages high similarity with positive pairs and low similarity with negatives
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class MultiModalMSELoss(nn.Module):
    """MSE-based consistency loss for multimodal alignment.
    
    Directly minimizes the Euclidean distance between features from two modalities.
    This approach encourages the same sample's features from different modalities
    to have similar representations in the embedding space.
    
    Unlike contrastive learning approaches (InfoNCE), MSE loss does not use negative
    samples, making it more suitable for small batch sizes where negative samples
    would be limited.
    
    Args:
        reduction: Specifies the reduction to apply to the output.
                  'mean' (default): average of all elements
                  'sum': sum of all elements
                  'none': no reduction
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'")
        
        self.reduction = reduction
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between two sets of features.
        
        Args:
            features1: Features from first modality of shape (batch_size, feature_dim).
            features2: Features from second modality of shape (batch_size, feature_dim).
            
        Returns:
            MSE loss scalar (or tensor if reduction='none').
            
        Raises:
            ValueError: If input tensors have mismatched dimensions.
        """
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")
        
        if features1.size(0) != features2.size(0):
            raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")
        
        if features1.size(1) != features2.size(1):
            raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")
        
        # Compute MSE loss - directly aligns features from both modalities
        loss = F.mse_loss(features1, features2, reduction=self.reduction)
        
        return loss


class MultiModalModel(nn.Module):
    """Enhanced multi-modal classification model with cross-modal fusion.
    
    Integrates transformer processing of solar wind time series with 
    ConvLSTM processing of image sequences, enhanced with cross-modal
    attention fusion. Outputs binary classification for each target variable
    across multiple time groups.
    
    Args:
        num_input_variables: Number of solar wind input variables.
        input_sequence_length: Length of input sequences.
        num_target_variables: Number of target variables to predict.
        num_target_groups: Number of time groups for classification.
        transformer_d_model: Transformer model dimension.
        transformer_nhead: Number of transformer attention heads.
        transformer_num_layers: Number of transformer encoder layers.
        transformer_dim_feedforward: Transformer feedforward dimension.
        transformer_dropout: Transformer dropout rate.
        convlstm_input_channels: Number of input channels for ConvLSTM.
        convlstm_hidden_channels: Number of hidden channels for ConvLSTM.
        convlstm_kernel_size: Kernel size for ConvLSTM.
        convlstm_num_layers: Number of ConvLSTM layers.
        fusion_num_heads: Number of attention heads for cross-modal fusion.
        fusion_dropout: Dropout rate for fusion module.
    """
    
    def __init__(
        self, num_input_variables, input_sequence_length,
        num_target_variables, num_target_groups,
        transformer_d_model, transformer_nhead, transformer_num_layers,
        transformer_dim_feedforward, transformer_dropout,
        convlstm_input_channels, convlstm_hidden_channels,
        convlstm_kernel_size, convlstm_num_layers,
        fusion_num_heads=4, fusion_dropout=0.1
        ):
        super(MultiModalModel, self).__init__()

        # Validate input parameters
        if num_target_variables <= 0 or num_target_groups <= 0:
            raise ValueError("Target variables and number of groups must be positive")

        self.transformer_model = TransformerEncoderModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout)

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
        
        # Classification head - outputs logits for binary classification
        # Output shape: (batch_size, num_target_groups, num_target_variables)
        self.classification_head = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(transformer_d_model // 2, num_target_groups * num_target_variables)
        )

        self.num_target_variables = num_target_variables
        self.num_target_groups = num_target_groups

    def forward(self, solar_wind_input, image_input, return_features=False):
        """Forward pass through the enhanced multi-modal classification model.
        
        Args:
            solar_wind_input: Solar wind data of shape (batch, sequence, variables).
            image_input: Image data of shape (batch, channels, seq_len, height, width).
            return_features: If True, return intermediate features for contrastive loss.
            
        Returns:
            If return_features=False:
                Classification logits of shape (batch, num_target_groups, num_target_variables).
            If return_features=True:
                Tuple of (logits, transformer_features, convlstm_features).
            
        Raises:
            ValueError: If inputs are None or batch sizes don't match.
        """
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
        
        # Generate classification logits
        logits = self.classification_head(fused_features)
        
        # Reshape to (batch, num_target_groups, num_target_variables)
        output = logits.reshape(logits.size(0), self.num_target_groups, self.num_target_variables)
        
        # Return features if requested (for contrastive loss)
        if return_features:
            return output, transformer_features, convlstm_features
        
        return output


def create_model(config, logger=None):
    """Create MultiModalModel instance from configuration config.
    
    Args:
        config: Configuration object containing model parameters.
        logger: Optional logger for output.
        
    Returns:
        MultiModalModel instance configured for binary classification.
    """
    # Calculate number of target groups from target sequence length and group size
    
    num_input_variables = len(config.data.input_variables)
    num_target_groups = 1
    num_target_variables = len(config.data.target_variables)
    message = f"Enhanced MultiModalModel (Classification) with Transformer, ConvLSTM, and CrossModalFusion created. Output shape: (batch, {num_target_groups}, {num_target_variables})"

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


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    model = create_model(config, logger=None)

    num_input_variables = len(config.data.input_variables)
    num_target_variables = len(config.data.target_variables)
    num_target_groups = 1

    sdo_shape = (
        config.experiment.batch_size,
        config.model.convlstm_input_channels,
        # config.model.convlstm_input_image_frames,
        20,
        config.data.sdo_image_size,
        config.data.sdo_image_size
    )
    
    inputs_shape = (
        config.experiment.batch_size,
        config.data.input_sequence_length,
        num_input_variables
    )

    targets_shape = (
        config.experiment.batch_size,
        1,
        num_target_variables
    )

    sdo = torch.randn(sdo_shape)
    inputs = torch.randn(inputs_shape)

    # Test 1: Normal forward pass (backward compatibility)
    print("=" * 60)
    print("Test 1: Normal forward pass (return_features=False)")
    print("=" * 60)
    outputs = model(
        solar_wind_input=inputs,
        image_input=sdo
    )
    targets = torch.ones(targets_shape)
    print("Output shape:", outputs.shape)  # Expected: (batch_size, num_target_groups, num_target_variables)
    print("Output sample:\n", outputs[0])

    loss = nn.BCEWithLogitsLoss()
    output_loss = loss(outputs, targets)
    print("Classification Loss:", output_loss.item())
    
    # Test 2: Forward pass with features (for contrastive loss)
    print("\n" + "=" * 60)
    print("Test 2: Forward pass with features (return_features=True)")
    print("=" * 60)
    outputs, transformer_features, convlstm_features = model(
        solar_wind_input=inputs,
        image_input=sdo,
        return_features=True
    )
    print("Output shape:", outputs.shape)
    print("Transformer features shape:", transformer_features.shape)
    print("ConvLSTM features shape:", convlstm_features.shape)
    
    # Test 3: Contrastive Losses (InfoNCE vs MSE)
    print("\n" + "=" * 60)
    print("Test 3: Contrastive Losses")
    print("=" * 60)
    
    # Test 3a: InfoNCE (original)
    print("\n[3a] InfoNCE Contrastive Loss:")
    contrastive_criterion_infonce = MultiModalContrastiveLoss(temperature=0.3, normalize=True)
    contrastive_loss_infonce = contrastive_criterion_infonce(transformer_features, convlstm_features)
    print(f"  InfoNCE Loss: {contrastive_loss_infonce.item():.6f}")
    print(f"  - Uses batch negatives (batch size: {config.experiment.batch_size})")
    print(f"  - Temperature: 0.3")
    
    # Test 3b: MSE (new)
    print("\n[3b] MSE Consistency Loss:")
    mse_criterion = MultiModalMSELoss(reduction='mean')
    mse_loss = mse_criterion(transformer_features, convlstm_features)
    print(f"  MSE Loss: {mse_loss.item():.6f}")
    print(f"  - Direct feature alignment")
    print(f"  - No negative samples needed")
    
    # Test 4: Combined loss (both variants)
    print("\n" + "=" * 60)
    print("Test 4: Combined Loss Comparisons")
    print("=" * 60)
    lambda_contrastive = 0.1
    
    print("\n[4a] With InfoNCE:")
    total_loss_infonce = output_loss + lambda_contrastive * contrastive_loss_infonce
    print(f"  Classification Loss: {output_loss.item():.6f}")
    print(f"  InfoNCE Loss:        {contrastive_loss_infonce.item():.6f}")
    print(f"  Total Loss (λ={lambda_contrastive}): {total_loss_infonce.item():.6f}")
    
    print("\n[4b] With MSE:")
    total_loss_mse = output_loss + lambda_contrastive * mse_loss
    print(f"  Classification Loss: {output_loss.item():.6f}")
    print(f"  MSE Loss:            {mse_loss.item():.6f}")
    print(f"  Total Loss (λ={lambda_contrastive}): {total_loss_mse.item():.6f}")
    
    # Test 5: Feature alignment (cosine similarity)
    print("\n" + "=" * 60)
    print("Test 5: Feature Alignment Metrics")
    print("=" * 60)
    cosine_sim = F.cosine_similarity(transformer_features, convlstm_features, dim=1)
    print(f"Cosine Similarity - Mean: {cosine_sim.mean().item():.4f}")
    print(f"Cosine Similarity - Std: {cosine_sim.std().item():.4f}")
    print(f"Cosine Similarity - Min: {cosine_sim.min().item():.4f}")
    print(f"Cosine Similarity - Max: {cosine_sim.max().item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

    # sdo_shape = (
    #     config.batch_size,
    #     config.convlstm_input_channels,
    #     config.convlstm_input_image_frames,
    #     config.image_size,
    #     config.image_size
    # )
    
    # inputs_shape = (
    #     config.batch_size,
    #     config.input_sequence_length,
    #     config.num_input_variables
    # )

    # targets_shape = (
    #     config.batch_size,
    #     1,
    #     config.num_target_variables
    # )

    # sdo = torch.randn(sdo_shape)
    # inputs = torch.randn(inputs_shape)

    # outputs = model(
    #     solar_wind_input=inputs,
    #     image_input=sdo
    # )

    # targets = torch.ones(targets_shape)

    # print("Output shape:", outputs.shape)  # Expected: (batch_size, num_target
    # print(outputs)

    # loss = nn.BCEWithLogitsLoss()
    # output_loss = loss(outputs, targets)
    # print("Loss:", output_loss.item())






    # sdo = torch.randn(
    #     )

    #     inputs = data_dict['inputs']
    #     targets = data_dict['targets']
    #     file_names = data_dict['file_names']
    #     print(sdo.shape, inputs.shape, targets.shape, file_names)
    #     print(targets)
    #     if i >= 2:
    #         break

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Tuple, Optional
# import hydra


# # Network architecture constants
# CONV_KERNEL_1X1 = 1
# CONV_KERNEL_3X3 = (1, 3, 3)
# CONV_KERNEL_5X5 = (1, 5, 5)
# CONV_KERNEL_7X7 = (1, 7, 7)

# PADDING_NONE = 0
# PADDING_1 = (0, 1, 1)
# PADDING_2 = (0, 2, 2)
# PADDING_3 = (0, 3, 3)

# STRIDE_1 = 1
# STRIDE_2x2 = (1, 2, 2)

# POOL_KERNEL_3X3 = (1, 3, 3)
# INCEPTION_BRANCHES = 4


# class ConvLSTMCell(nn.Module):
#     """ConvLSTM cell for processing spatial-temporal data.
    
#     Combines convolutional operations with LSTM gates to process
#     spatial-temporal sequences while preserving spatial structure.
    
#     Args:
#         input_channels: Number of input channels.
#         hidden_channels: Number of hidden state channels.
#         kernel_size: Size of convolutional kernel.
#         bias: Whether to use bias in convolutions.
#     """
    
#     def __init__(self, input_channels: int, hidden_channels: int, 
#                  kernel_size: int = 3, bias: bool = True):
#         super().__init__()
        
#         if input_channels <= 0 or hidden_channels <= 0:
#             raise ValueError("Input and hidden channels must be positive")
#         if kernel_size <= 0 or kernel_size % 2 == 0:
#             raise ValueError("Kernel size must be positive and odd")
        
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.padding = kernel_size // 2
#         self.bias = bias
        
#         # Convolutional layers for input-to-hidden and hidden-to-hidden
#         self.conv_ih = nn.Conv2d(
#             input_channels, 4 * hidden_channels, 
#             kernel_size, padding=self.padding, bias=bias
#         )
#         self.conv_hh = nn.Conv2d(
#             hidden_channels, 4 * hidden_channels,
#             kernel_size, padding=self.padding, bias=bias
#         )
    
#     def forward(self, input_tensor: torch.Tensor, 
#                 hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Forward pass of ConvLSTM cell.
        
#         Args:
#             input_tensor: Input tensor of shape (batch, channels, height, width).
#             hidden_state: Tuple of (hidden, cell) states or None for initialization.
            
#         Returns:
#             Tuple of (new_hidden, new_cell) states.
#         """
#         batch_size, _, height, width = input_tensor.size()
        
#         # Initialize hidden and cell states if not provided
#         if hidden_state is None:
#             device = input_tensor.device
#             hidden = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
#             cell = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
#         else:
#             hidden, cell = hidden_state
        
#         # Compute convolutions
#         conv_ih = self.conv_ih(input_tensor)
#         conv_hh = self.conv_hh(hidden)
#         combined_conv = conv_ih + conv_hh
        
#         # Split into gate components
#         i_gate, f_gate, o_gate, g_gate = torch.split(combined_conv, self.hidden_channels, dim=1)
        
#         # Apply activations
#         input_gate = torch.sigmoid(i_gate)
#         forget_gate = torch.sigmoid(f_gate)
#         output_gate = torch.sigmoid(o_gate)
#         candidate_gate = torch.tanh(g_gate)
        
#         # Update cell and hidden states
#         new_cell = forget_gate * cell + input_gate * candidate_gate
#         new_hidden = output_gate * torch.tanh(new_cell)
        
#         return new_hidden, new_cell


# class ConvLSTMModel(nn.Module):
#     """ConvLSTM model for spatial-temporal sequence processing.
    
#     Processes image sequences using ConvLSTM layers to capture
#     both spatial and temporal dependencies simultaneously.
    
#     Args:
#         input_channels: Number of input channels.
#         hidden_channels: Number of hidden channels for ConvLSTM.
#         kernel_size: Kernel size for ConvLSTM convolutions.
#         num_layers: Number of ConvLSTM layers.
#         output_dim: Final output dimension.
#     """
    
#     def __init__(self, input_channels: int, hidden_channels: int = 64,
#                  kernel_size: int = 3, num_layers: int = 2, 
#                  output_dim: int = 256):
#         super().__init__()
        
#         if input_channels <= 0 or hidden_channels <= 0:
#             raise ValueError("Input and hidden channels must be positive")
#         if num_layers <= 0:
#             raise ValueError("Number of layers must be positive")
#         if output_dim <= 0:
#             raise ValueError("Output dimension must be positive")
        
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.num_layers = num_layers
#         self.output_dim = output_dim
        
#         # Create ConvLSTM layers
#         self.convlstm_layers = nn.ModuleList()
        
#         # First layer takes input channels
#         self.convlstm_layers.append(
#             ConvLSTMCell(input_channels, hidden_channels, kernel_size)
#         )
        
#         # Subsequent layers take hidden channels
#         for _ in range(1, num_layers):
#             self.convlstm_layers.append(
#                 ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
#             )
        
#         # Spatial pooling and output projection
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.output_projection = nn.Sequential(
#             nn.Linear(hidden_channels, output_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through ConvLSTM layers.
        
#         Args:
#             x: Input tensor of shape (batch, channels, seq_len, height, width).
            
#         Returns:
#             Output tensor of shape (batch, output_dim).
#         """
#         if x.dim() != 5:
#             raise ValueError(f"Expected 5D input tensor (batch, channels, seq_len, height, width), got {x.dim()}D tensor")
        
#         batch_size, channels, seq_len, height, width = x.size()
        
#         if channels != self.input_channels:
#             raise ValueError(f"Expected {self.input_channels} input channels, got {channels}")
        
#         # Initialize hidden states for all layers
#         hidden_states = [None] * self.num_layers
        
#         # Process each time step
#         for t in range(seq_len):
#             input_frame = x[:, :, t, :, :]  # (batch, channels, height, width)
            
#             # Pass through each ConvLSTM layer
#             for layer_idx, convlstm_layer in enumerate(self.convlstm_layers):
#                 if layer_idx == 0:
#                     # First layer gets input frame
#                     hidden_states[layer_idx] = convlstm_layer(input_frame, hidden_states[layer_idx])
#                 else:
#                     # Subsequent layers get previous layer's hidden state
#                     hidden_states[layer_idx] = convlstm_layer(
#                         hidden_states[layer_idx - 1][0], hidden_states[layer_idx]
#                     )
        
#         # Use final hidden state from last layer
#         final_hidden = hidden_states[-1][0]  # (batch, hidden_channels, height, width)
        
#         # Global average pooling
#         pooled = self.global_pool(final_hidden).squeeze(-1).squeeze(-1)  # (batch, hidden_channels)
        
#         # Output projection
#         output = self.output_projection(pooled)
        
#         return output


# class PositionalEncoding(nn.Module):
#     """Positional encoding for transformer input sequences.
    
#     Adds sinusoidal positional embeddings to input sequences to provide
#     temporal information for the transformer model.
    
#     Args:
#         d_model: Model dimension.
#         max_len: Maximum sequence length.
#         dropout: Dropout rate.
#     """
    
#     def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Create positional encoding matrix
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
#                            (-math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
        
#         # Register as buffer (not a parameter)
#         self.register_buffer('pe', pe)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Add positional encoding to input tensor.
        
#         Args:
#             x: Input tensor of shape (seq_len, batch_size, d_model).
            
#         Returns:
#             Tensor with positional encoding added.
#         """
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


# class TransformerEncoderModel(nn.Module):
#     """Transformer encoder model for solar wind time series data.
    
#     Processes multivariate time series data using transformer encoder layers
#     with positional encoding and multi-head attention mechanism.
    
#     Args:
#         num_input_variables: Number of input variables (features).
#         input_sequence_length: Length of input sequences.
#         d_model: Model dimension for transformer.
#         nhead: Number of attention heads.
#         num_layers: Number of transformer encoder layers.
#         dim_feedforward: Dimension of feedforward network.
#         dropout: Dropout rate.
#     """
    
#     def __init__(self, num_input_variables: int, input_sequence_length: int,
#                  d_model: int = 256, nhead: int = 8, num_layers: int = 3,
#                  dim_feedforward: int = 512, dropout: float = 0.1):
#         super().__init__()
        
#         # Validate input parameters
#         if num_input_variables <= 0:
#             raise ValueError(f"Number of input variables must be positive, got {num_input_variables}")
#         if input_sequence_length <= 0:
#             raise ValueError(f"Input sequence length must be positive, got {input_sequence_length}")
#         if d_model <= 0:
#             raise ValueError(f"Model dimension must be positive, got {d_model}")
#         if d_model % nhead != 0:
#             raise ValueError(f"Model dimension {d_model} must be divisible by number of heads {nhead}")
#         if nhead <= 0 or num_layers <= 0:
#             raise ValueError("Number of heads and layers must be positive")
#         if not (0.0 <= dropout <= 1.0):
#             raise ValueError("Dropout must be between 0 and 1")
        
#         self.d_model = d_model
#         self.input_sequence_length = input_sequence_length
#         self.num_input_variables = num_input_variables
        
#         # Input projection layer
#         self.input_projection = nn.Linear(num_input_variables, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model, input_sequence_length, dropout)
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=False  # (seq_len, batch, d_model)
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
#         # Output aggregation
#         self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
#         self.output_projection = nn.Linear(d_model, d_model)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through transformer encoder.
        
#         Args:
#             x: Input tensor of shape (batch_size, seq_len, num_variables).
            
#         Returns:
#             Encoded features of shape (batch_size, d_model).
            
#         Raises:
#             ValueError: If input tensor has invalid dimensions.
#         """
#         # Validate input tensor dimensions
#         if x.dim() != 3:
#             raise ValueError(f"Expected 3D input tensor (batch, seq_len, variables), got {x.dim()}D tensor")
        
#         batch_size, seq_len, num_vars = x.size()
        
#         if seq_len != self.input_sequence_length:
#             raise ValueError(f"Expected sequence length {self.input_sequence_length}, got {seq_len}")
#         if num_vars != self.num_input_variables:
#             raise ValueError(f"Expected {self.num_input_variables} variables, got {num_vars}")
        
#         # Project to model dimension: (batch, seq_len, d_model)
#         x = self.input_projection(x)
        
#         # Transpose for transformer: (seq_len, batch, d_model)
#         x = x.transpose(0, 1)
        
#         # Add positional encoding
#         x = self.pos_encoder(x)
        
#         # Apply transformer encoder
#         x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        
#         # Transpose back: (batch, seq_len, d_model)
#         x = x.transpose(0, 1)
        
#         # Global average pooling: (batch, d_model, 1) -> (batch, d_model)
#         x = x.transpose(1, 2)  # (batch, d_model, seq_len)
#         x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
#         # Final projection
#         x = self.output_projection(x)
        
#         return x


# class CrossModalAttention(nn.Module):
#     """Cross-modal attention mechanism for feature fusion.
    
#     Enables interaction between transformer and ConvLSTM features
#     through mutual attention computation.
    
#     Args:
#         feature_dim: Dimension of input features.
#         num_heads: Number of attention heads.
#         dropout: Dropout rate.
#     """
    
#     def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
#         super().__init__()
        
#         if feature_dim <= 0:
#             raise ValueError("Feature dimension must be positive")
#         if feature_dim % num_heads != 0:
#             raise ValueError("Feature dimension must be divisible by number of heads")
#         if num_heads <= 0:
#             raise ValueError("Number of heads must be positive")
        
#         self.feature_dim = feature_dim
#         self.num_heads = num_heads
#         self.head_dim = feature_dim // num_heads
        
#         # Query, Key, Value projections
#         self.q_proj = nn.Linear(feature_dim, feature_dim)
#         self.k_proj = nn.Linear(feature_dim, feature_dim)
#         self.v_proj = nn.Linear(feature_dim, feature_dim)
        
#         # Output projection
#         self.out_proj = nn.Linear(feature_dim, feature_dim)
#         self.dropout = nn.Dropout(dropout)
        
#         # Layer normalization
#         self.norm = nn.LayerNorm(feature_dim)
        
#     def forward(self, query_features: torch.Tensor, key_value_features: torch.Tensor) -> torch.Tensor:
#         """Forward pass of cross-modal attention.
        
#         Args:
#             query_features: Query features of shape (batch, feature_dim).
#             key_value_features: Key-Value features of shape (batch, feature_dim).
            
#         Returns:
#             Attended features of shape (batch, feature_dim).
#         """
#         batch_size = query_features.size(0)
        
#         # Project to Q, K, V
#         Q = self.q_proj(query_features)  # (batch, feature_dim)
#         K = self.k_proj(key_value_features)  # (batch, feature_dim)
#         V = self.v_proj(key_value_features)  # (batch, feature_dim)
        
#         # Reshape for multi-head attention
#         Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
#         K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
#         V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        
#         # Compute attention scores
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         attention_weights = F.softmax(attention_scores, dim=-1)
#         attention_weights = self.dropout(attention_weights)
        
#         # Apply attention to values
#         attended = torch.matmul(attention_weights, V)  # (batch, num_heads, 1, head_dim)
        
#         # Concatenate heads
#         attended = attended.transpose(1, 2).contiguous().view(batch_size, self.feature_dim)
        
#         # Output projection and residual connection
#         output = self.out_proj(attended)
#         output = self.norm(output + query_features)
        
#         return output


# class CrossModalFusion(nn.Module):
#     """Cross-modal fusion module for combining transformer and ConvLSTM features.
    
#     Uses bidirectional cross-attention to enable interaction between
#     solar wind (transformer) and image (ConvLSTM) features.
    
#     Args:
#         feature_dim: Dimension of input features.
#         num_heads: Number of attention heads.
#         dropout: Dropout rate.
#     """
    
#     def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
#         super().__init__()
        
#         self.feature_dim = feature_dim
        
#         # Cross-attention layers
#         self.transformer_to_convlstm = CrossModalAttention(feature_dim, num_heads, dropout)
#         self.convlstm_to_transformer = CrossModalAttention(feature_dim, num_heads, dropout)
        
#         # Feature alignment and combination
#         self.feature_gate = nn.Sequential(
#             nn.Linear(feature_dim * 2, feature_dim),
#             nn.Sigmoid()
#         )
        
#         self.combination_layer = nn.Sequential(
#             nn.Linear(feature_dim * 2, feature_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(feature_dim, feature_dim)
#         )
        
#         # Final normalization
#         self.final_norm = nn.LayerNorm(feature_dim)
        
#     def forward(self, transformer_features: torch.Tensor, convlstm_features: torch.Tensor) -> torch.Tensor:
#         """Forward pass of cross-modal fusion.
        
#         Args:
#             transformer_features: Features from transformer of shape (batch, feature_dim).
#             convlstm_features: Features from ConvLSTM of shape (batch, feature_dim).
            
#         Returns:
#             Fused features of shape (batch, feature_dim).
#         """
#         # Cross-attention in both directions
#         transformer_attended = self.transformer_to_convlstm(transformer_features, convlstm_features)
#         convlstm_attended = self.convlstm_to_transformer(convlstm_features, transformer_features)
        
#         # Concatenate attended features
#         concatenated = torch.cat([transformer_attended, convlstm_attended], dim=1)
        
#         # Compute gating weights
#         gate_weights = self.feature_gate(concatenated)
        
#         # Weighted combination
#         weighted_transformer = gate_weights * transformer_attended
#         weighted_convlstm = (1 - gate_weights) * convlstm_attended
        
#         # Final combination
#         combined = torch.cat([weighted_transformer, weighted_convlstm], dim=1)
#         fused_features = self.combination_layer(combined)
        
#         # Add residual connection and normalize
#         residual = (transformer_features + convlstm_features) / 2
#         output = self.final_norm(fused_features + residual)
        
#         return output


# class MultiModalContrastiveLoss(nn.Module):
#     """Contrastive loss for aligning multimodal representations.
    
#     Uses InfoNCE-style loss to align transformer (solar wind) and ConvLSTM (image)
#     features in the same embedding space. Features from the same sample are treated
#     as positive pairs, while features from different samples in the batch serve as
#     negative pairs (in-batch negatives).
    
#     Args:
#         temperature: Temperature parameter for scaling similarity scores.
#                     Higher values (0.3-0.5) recommended for small batch sizes.
#                     Lower values (0.07-0.1) for large batch sizes.
#         normalize: Whether to L2-normalize features before computing similarity.
#                   True is strongly recommended for stable training.
#     """
    
#     def __init__(self, temperature: float = 0.3, normalize: bool = True):
#         super().__init__()
        
#         if temperature <= 0:
#             raise ValueError(f"Temperature must be positive, got {temperature}")
        
#         self.temperature = temperature
#         self.normalize = normalize
        
#     def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
#         """Compute contrastive loss between two sets of features.
        
#         Args:
#             features1: Features from first modality of shape (batch_size, feature_dim).
#             features2: Features from second modality of shape (batch_size, feature_dim).
            
#         Returns:
#             Contrastive loss scalar.
            
#         Raises:
#             ValueError: If input tensors have mismatched dimensions.
#         """
#         if features1.dim() != 2 or features2.dim() != 2:
#             raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")
        
#         if features1.size(0) != features2.size(0):
#             raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")
        
#         if features1.size(1) != features2.size(1):
#             raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")
        
#         batch_size = features1.size(0)
        
#         # L2 normalization for cosine similarity
#         if self.normalize:
#             features1 = F.normalize(features1, p=2, dim=1)
#             features2 = F.normalize(features2, p=2, dim=1)
        
#         # Concatenate features from both modalities
#         # Shape: (2 * batch_size, feature_dim)
#         features = torch.cat([features1, features2], dim=0)
        
#         # Compute similarity matrix
#         # Shape: (2 * batch_size, 2 * batch_size)
#         similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
#         # Create labels for positive pairs
#         # For each sample i in features1, its positive pair is sample i in features2
#         # which is at index (i + batch_size) in the concatenated tensor
#         labels = torch.arange(batch_size, device=features1.device)
#         labels = torch.cat([labels + batch_size, labels], dim=0)
        
#         # Create mask to exclude self-similarity (diagonal elements)
#         # We don't want to compare a sample with itself
#         mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features1.device)
#         similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
#         # Compute cross-entropy loss
#         # This encourages high similarity with positive pairs and low similarity with negatives
#         loss = F.cross_entropy(similarity_matrix, labels)
        
#         return loss


# class MultiModalModel(nn.Module):
#     """Enhanced multi-modal classification model with cross-modal fusion.
    
#     Integrates transformer processing of solar wind time series with 
#     ConvLSTM processing of image sequences, enhanced with cross-modal
#     attention fusion. Outputs binary classification for each target variable
#     across multiple time groups.
    
#     Args:
#         num_input_variables: Number of solar wind input variables.
#         input_sequence_length: Length of input sequences.
#         num_target_variables: Number of target variables to predict.
#         num_target_groups: Number of time groups for classification.
#         transformer_d_model: Transformer model dimension.
#         transformer_nhead: Number of transformer attention heads.
#         transformer_num_layers: Number of transformer encoder layers.
#         transformer_dim_feedforward: Transformer feedforward dimension.
#         transformer_dropout: Transformer dropout rate.
#         convlstm_input_channels: Number of input channels for ConvLSTM.
#         convlstm_hidden_channels: Number of hidden channels for ConvLSTM.
#         convlstm_kernel_size: Kernel size for ConvLSTM.
#         convlstm_num_layers: Number of ConvLSTM layers.
#         fusion_num_heads: Number of attention heads for cross-modal fusion.
#         fusion_dropout: Dropout rate for fusion module.
#     """
    
#     def __init__(
#         self, num_input_variables, input_sequence_length,
#         num_target_variables, num_target_groups,
#         transformer_d_model, transformer_nhead, transformer_num_layers,
#         transformer_dim_feedforward, transformer_dropout,
#         convlstm_input_channels, convlstm_hidden_channels,
#         convlstm_kernel_size, convlstm_num_layers,
#         fusion_num_heads=4, fusion_dropout=0.1
#         ):
#         super(MultiModalModel, self).__init__()

#         # Validate input parameters
#         if num_target_variables <= 0 or num_target_groups <= 0:
#             raise ValueError("Target variables and number of groups must be positive")

#         self.transformer_model = TransformerEncoderModel(
#             num_input_variables=num_input_variables,
#             input_sequence_length=input_sequence_length,
#             d_model=transformer_d_model,
#             nhead=transformer_nhead,
#             num_layers=transformer_num_layers,
#             dim_feedforward=transformer_dim_feedforward,
#             dropout=transformer_dropout)

#         self.convlstm_model = ConvLSTMModel(
#             input_channels=convlstm_input_channels,
#             hidden_channels=convlstm_hidden_channels,
#             kernel_size=convlstm_kernel_size,
#             num_layers=convlstm_num_layers,
#             output_dim=transformer_d_model)  # Same dimension as transformer

#         # Cross-modal fusion module
#         self.cross_modal_fusion = CrossModalFusion(
#             feature_dim=transformer_d_model,
#             num_heads=fusion_num_heads,
#             dropout=fusion_dropout
#         )
        
#         # Classification head - outputs logits for binary classification
#         # Output shape: (batch_size, num_target_groups, num_target_variables)
#         self.classification_head = nn.Sequential(
#             nn.Linear(transformer_d_model, transformer_d_model // 2),
#             nn.ReLU(),
#             nn.Dropout(fusion_dropout),
#             nn.Linear(transformer_d_model // 2, num_target_groups * num_target_variables)
#         )

#         self.num_target_variables = num_target_variables
#         self.num_target_groups = num_target_groups

#     def forward(self, solar_wind_input, image_input, return_features=False):
#         """Forward pass through the enhanced multi-modal classification model.
        
#         Args:
#             solar_wind_input: Solar wind data of shape (batch, sequence, variables).
#             image_input: Image data of shape (batch, channels, seq_len, height, width).
#             return_features: If True, return intermediate features for contrastive loss.
            
#         Returns:
#             If return_features=False:
#                 Classification logits of shape (batch, num_target_groups, num_target_variables).
#             If return_features=True:
#                 Tuple of (logits, transformer_features, convlstm_features).
            
#         Raises:
#             ValueError: If inputs are None or batch sizes don't match.
#         """
#         # Validate input tensors
#         if solar_wind_input is None or image_input is None:
#             raise ValueError("Both solar_wind_input and image_input must be provided")
        
#         if solar_wind_input.size(0) != image_input.size(0):
#             raise ValueError(f"Batch sizes must match: solar_wind={solar_wind_input.size(0)}, image={image_input.size(0)}")

#         # Extract features from each modality
#         transformer_features = self.transformer_model(solar_wind_input)
#         convlstm_features = self.convlstm_model(image_input)
        
#         # Apply cross-modal fusion
#         fused_features = self.cross_modal_fusion(transformer_features, convlstm_features)
        
#         # Generate classification logits
#         logits = self.classification_head(fused_features)
        
#         # Reshape to (batch, num_target_groups, num_target_variables)
#         output = logits.reshape(logits.size(0), self.num_target_groups, self.num_target_variables)
        
#         # Return features if requested (for contrastive loss)
#         if return_features:
#             return output, transformer_features, convlstm_features
        
#         return output


# def create_model(config, logger=None):
#     """Create MultiModalModel instance from configuration config.
    
#     Args:
#         config: Configuration object containing model parameters.
#         logger: Optional logger for output.
        
#     Returns:
#         MultiModalModel instance configured for binary classification.
#     """
#     # Calculate number of target groups from target sequence length and group size
    
#     num_input_variables = len(config.data.input_variables)
#     num_target_groups = 1
#     num_target_variables = len(config.data.target_variables)
#     message = f"Enhanced MultiModalModel (Classification) with Transformer, ConvLSTM, and CrossModalFusion created. Output shape: (batch, {num_target_groups}, {num_target_variables})"

#     if logger:
#         logger.info(message)
#     else:
#         print(message)
        
#     return MultiModalModel(
#         num_input_variables=num_input_variables,
#         input_sequence_length=config.data.input_sequence_length,
#         num_target_variables=num_target_variables,
#         num_target_groups=num_target_groups,
#         transformer_d_model=config.model.transformer_d_model,
#         transformer_nhead=config.model.transformer_nhead,
#         transformer_num_layers=config.model.transformer_num_layers,
#         transformer_dim_feedforward=config.model.transformer_dim_feedforward,
#         transformer_dropout=config.model.transformer_dropout,
#         convlstm_input_channels=config.model.convlstm_input_channels,
#         convlstm_hidden_channels=config.model.convlstm_hidden_channels,
#         convlstm_kernel_size=config.model.convlstm_kernel_size,
#         convlstm_num_layers=config.model.convlstm_num_layers,
#         fusion_num_heads=config.model.fusion_num_heads,
#         fusion_dropout=config.model.fusion_dropout
#     )


# @hydra.main(config_path="./configs", version_base=None)
# def main(config):

#     model = create_model(config, logger=None)

#     num_input_variables = len(config.data.input_variables)
#     num_target_variables = len(config.data.target_variables)
#     num_target_groups = 1

#     sdo_shape = (
#         config.experiment.batch_size,
#         config.model.convlstm_input_channels,
#         # config.model.convlstm_input_image_frames,
#         20,
#         config.data.sdo_image_size,
#         config.data.sdo_image_size
#     )
    
#     inputs_shape = (
#         config.experiment.batch_size,
#         config.data.input_sequence_length,
#         num_input_variables
#     )

#     targets_shape = (
#         config.experiment.batch_size,
#         1,
#         num_target_variables
#     )

#     sdo = torch.randn(sdo_shape)
#     inputs = torch.randn(inputs_shape)

#     # Test 1: Normal forward pass (backward compatibility)
#     print("=" * 60)
#     print("Test 1: Normal forward pass (return_features=False)")
#     print("=" * 60)
#     outputs = model(
#         solar_wind_input=inputs,
#         image_input=sdo
#     )
#     targets = torch.ones(targets_shape)
#     print("Output shape:", outputs.shape)  # Expected: (batch_size, num_target_groups, num_target_variables)
#     print("Output sample:\n", outputs[0])

#     loss = nn.BCEWithLogitsLoss()
#     output_loss = loss(outputs, targets)
#     print("Classification Loss:", output_loss.item())
    
#     # Test 2: Forward pass with features (for contrastive loss)
#     print("\n" + "=" * 60)
#     print("Test 2: Forward pass with features (return_features=True)")
#     print("=" * 60)
#     outputs, transformer_features, convlstm_features = model(
#         solar_wind_input=inputs,
#         image_input=sdo,
#         return_features=True
#     )
#     print("Output shape:", outputs.shape)
#     print("Transformer features shape:", transformer_features.shape)
#     print("ConvLSTM features shape:", convlstm_features.shape)
    
#     # Test 3: Contrastive loss
#     print("\n" + "=" * 60)
#     print("Test 3: Contrastive Loss")
#     print("=" * 60)
#     contrastive_criterion = MultiModalContrastiveLoss(temperature=0.3, normalize=True)
#     contrastive_loss = contrastive_criterion(transformer_features, convlstm_features)
#     print("Contrastive Loss:", contrastive_loss.item())
    
#     # Test 4: Combined loss
#     print("\n" + "=" * 60)
#     print("Test 4: Combined Loss")
#     print("=" * 60)
#     lambda_contrastive = 0.1
#     total_loss = output_loss + lambda_contrastive * contrastive_loss
#     print(f"Classification Loss: {output_loss.item():.6f}")
#     print(f"Contrastive Loss: {contrastive_loss.item():.6f}")
#     print(f"Total Loss (λ={lambda_contrastive}): {total_loss.item():.6f}")
    
#     # Test 5: Feature alignment (cosine similarity)
#     print("\n" + "=" * 60)
#     print("Test 5: Feature Alignment Metrics")
#     print("=" * 60)
#     cosine_sim = F.cosine_similarity(transformer_features, convlstm_features, dim=1)
#     print(f"Cosine Similarity - Mean: {cosine_sim.mean().item():.4f}")
#     print(f"Cosine Similarity - Std: {cosine_sim.std().item():.4f}")
#     print(f"Cosine Similarity - Min: {cosine_sim.min().item():.4f}")
#     print(f"Cosine Similarity - Max: {cosine_sim.max().item():.4f}")
    
#     print("\n" + "=" * 60)
#     print("All tests passed successfully!")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()

#     # sdo_shape = (
#     #     config.batch_size,
#     #     config.convlstm_input_channels,
#     #     config.convlstm_input_image_frames,
#     #     config.image_size,
#     #     config.image_size
#     # )
    
#     # inputs_shape = (
#     #     config.batch_size,
#     #     config.input_sequence_length,
#     #     config.num_input_variables
#     # )

#     # targets_shape = (
#     #     config.batch_size,
#     #     1,
#     #     config.num_target_variables
#     # )

#     # sdo = torch.randn(sdo_shape)
#     # inputs = torch.randn(inputs_shape)

#     # outputs = model(
#     #     solar_wind_input=inputs,
#     #     image_input=sdo
#     # )

#     # targets = torch.ones(targets_shape)

#     # print("Output shape:", outputs.shape)  # Expected: (batch_size, num_target
#     # print(outputs)

#     # loss = nn.BCEWithLogitsLoss()
#     # output_loss = loss(outputs, targets)
#     # print("Loss:", output_loss.item())






#     # sdo = torch.randn(
#     #     )

#     #     inputs = data_dict['inputs']
#     #     targets = data_dict['targets']
#     #     file_names = data_dict['file_names']
#     #     print(sdo.shape, inputs.shape, targets.shape, file_names)
#     #     print(targets)
#     #     if i >= 2:
#     #         break


# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import math
# # from typing import Tuple, Optional
# # import hydra


# # # Network architecture constants
# # CONV_KERNEL_1X1 = 1
# # CONV_KERNEL_3X3 = (1, 3, 3)
# # CONV_KERNEL_5X5 = (1, 5, 5)
# # CONV_KERNEL_7X7 = (1, 7, 7)

# # PADDING_NONE = 0
# # PADDING_1 = (0, 1, 1)
# # PADDING_2 = (0, 2, 2)
# # PADDING_3 = (0, 3, 3)

# # STRIDE_1 = 1
# # STRIDE_2x2 = (1, 2, 2)

# # POOL_KERNEL_3X3 = (1, 3, 3)
# # INCEPTION_BRANCHES = 4


# # class ConvLSTMCell(nn.Module):
# #     """ConvLSTM cell for processing spatial-temporal data.
    
# #     Combines convolutional operations with LSTM gates to process
# #     spatial-temporal sequences while preserving spatial structure.
    
# #     Args:
# #         input_channels: Number of input channels.
# #         hidden_channels: Number of hidden state channels.
# #         kernel_size: Size of convolutional kernel.
# #         bias: Whether to use bias in convolutions.
# #     """
    
# #     def __init__(self, input_channels: int, hidden_channels: int, 
# #                  kernel_size: int = 3, bias: bool = True):
# #         super().__init__()
        
# #         if input_channels <= 0 or hidden_channels <= 0:
# #             raise ValueError("Input and hidden channels must be positive")
# #         if kernel_size <= 0 or kernel_size % 2 == 0:
# #             raise ValueError("Kernel size must be positive and odd")
        
# #         self.input_channels = input_channels
# #         self.hidden_channels = hidden_channels
# #         self.kernel_size = kernel_size
# #         self.padding = kernel_size // 2
# #         self.bias = bias
        
# #         # Convolutional layers for input-to-hidden and hidden-to-hidden
# #         self.conv_ih = nn.Conv2d(
# #             input_channels, 4 * hidden_channels, 
# #             kernel_size, padding=self.padding, bias=bias
# #         )
# #         self.conv_hh = nn.Conv2d(
# #             hidden_channels, 4 * hidden_channels,
# #             kernel_size, padding=self.padding, bias=bias
# #         )
    
# #     def forward(self, input_tensor: torch.Tensor, 
# #                 hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
# #         """Forward pass of ConvLSTM cell.
        
# #         Args:
# #             input_tensor: Input tensor of shape (batch, channels, height, width).
# #             hidden_state: Tuple of (hidden, cell) states or None for initialization.
            
# #         Returns:
# #             Tuple of (new_hidden, new_cell) states.
# #         """
# #         batch_size, _, height, width = input_tensor.size()
        
# #         # Initialize hidden and cell states if not provided
# #         if hidden_state is None:
# #             device = input_tensor.device
# #             hidden = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
# #             cell = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
# #         else:
# #             hidden, cell = hidden_state
        
# #         # Compute convolutions
# #         conv_ih = self.conv_ih(input_tensor)
# #         conv_hh = self.conv_hh(hidden)
# #         combined_conv = conv_ih + conv_hh
        
# #         # Split into gate components
# #         i_gate, f_gate, o_gate, g_gate = torch.split(combined_conv, self.hidden_channels, dim=1)
        
# #         # Apply activations
# #         input_gate = torch.sigmoid(i_gate)
# #         forget_gate = torch.sigmoid(f_gate)
# #         output_gate = torch.sigmoid(o_gate)
# #         candidate_gate = torch.tanh(g_gate)
        
# #         # Update cell and hidden states
# #         new_cell = forget_gate * cell + input_gate * candidate_gate
# #         new_hidden = output_gate * torch.tanh(new_cell)
        
# #         return new_hidden, new_cell


# # class ConvLSTMModel(nn.Module):
# #     """ConvLSTM model for spatial-temporal sequence processing.
    
# #     Processes image sequences using ConvLSTM layers to capture
# #     both spatial and temporal dependencies simultaneously.
    
# #     Args:
# #         input_channels: Number of input channels.
# #         hidden_channels: Number of hidden channels for ConvLSTM.
# #         kernel_size: Kernel size for ConvLSTM convolutions.
# #         num_layers: Number of ConvLSTM layers.
# #         output_dim: Final output dimension.
# #     """
    
# #     def __init__(self, input_channels: int, hidden_channels: int = 64,
# #                  kernel_size: int = 3, num_layers: int = 2, 
# #                  output_dim: int = 256):
# #         super().__init__()
        
# #         if input_channels <= 0 or hidden_channels <= 0:
# #             raise ValueError("Input and hidden channels must be positive")
# #         if num_layers <= 0:
# #             raise ValueError("Number of layers must be positive")
# #         if output_dim <= 0:
# #             raise ValueError("Output dimension must be positive")
        
# #         self.input_channels = input_channels
# #         self.hidden_channels = hidden_channels
# #         self.num_layers = num_layers
# #         self.output_dim = output_dim
        
# #         # Create ConvLSTM layers
# #         self.convlstm_layers = nn.ModuleList()
        
# #         # First layer takes input channels
# #         self.convlstm_layers.append(
# #             ConvLSTMCell(input_channels, hidden_channels, kernel_size)
# #         )
        
# #         # Subsequent layers take hidden channels
# #         for _ in range(1, num_layers):
# #             self.convlstm_layers.append(
# #                 ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
# #             )
        
# #         # Spatial pooling and output projection
# #         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
# #         self.output_projection = nn.Sequential(
# #             nn.Linear(hidden_channels, output_dim),
# #             nn.ReLU(),
# #             nn.Dropout(0.1)
# #         )
    
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """Forward pass through ConvLSTM layers.
        
# #         Args:
# #             x: Input tensor of shape (batch, channels, seq_len, height, width).
            
# #         Returns:
# #             Output tensor of shape (batch, output_dim).
# #         """
# #         if x.dim() != 5:
# #             raise ValueError(f"Expected 5D input tensor (batch, channels, seq_len, height, width), got {x.dim()}D tensor")
        
# #         batch_size, channels, seq_len, height, width = x.size()
        
# #         if channels != self.input_channels:
# #             raise ValueError(f"Expected {self.input_channels} input channels, got {channels}")
        
# #         # Initialize hidden states for all layers
# #         hidden_states = [None] * self.num_layers
        
# #         # Process each time step
# #         for t in range(seq_len):
# #             input_frame = x[:, :, t, :, :]  # (batch, channels, height, width)
            
# #             # Pass through each ConvLSTM layer
# #             for layer_idx, convlstm_layer in enumerate(self.convlstm_layers):
# #                 if layer_idx == 0:
# #                     # First layer gets input frame
# #                     hidden_states[layer_idx] = convlstm_layer(input_frame, hidden_states[layer_idx])
# #                 else:
# #                     # Subsequent layers get previous layer's hidden state
# #                     hidden_states[layer_idx] = convlstm_layer(
# #                         hidden_states[layer_idx - 1][0], hidden_states[layer_idx]
# #                     )
        
# #         # Use final hidden state from last layer
# #         final_hidden = hidden_states[-1][0]  # (batch, hidden_channels, height, width)
        
# #         # Global average pooling
# #         pooled = self.global_pool(final_hidden).squeeze(-1).squeeze(-1)  # (batch, hidden_channels)
        
# #         # Output projection
# #         output = self.output_projection(pooled)
        
# #         return output


# # class PositionalEncoding(nn.Module):
# #     """Positional encoding for transformer input sequences.
    
# #     Adds sinusoidal positional embeddings to input sequences to provide
# #     temporal information for the transformer model.
    
# #     Args:
# #         d_model: Model dimension.
# #         max_len: Maximum sequence length.
# #         dropout: Dropout rate.
# #     """
    
# #     def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
# #         super().__init__()
# #         self.dropout = nn.Dropout(p=dropout)
        
# #         # Create positional encoding matrix
# #         pe = torch.zeros(max_len, d_model)
# #         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
# #         div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
# #                            (-math.log(10000.0) / d_model))
        
# #         pe[:, 0::2] = torch.sin(position * div_term)
# #         pe[:, 1::2] = torch.cos(position * div_term)
# #         pe = pe.unsqueeze(0).transpose(0, 1)
        
# #         # Register as buffer (not a parameter)
# #         self.register_buffer('pe', pe)
    
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """Add positional encoding to input tensor.
        
# #         Args:
# #             x: Input tensor of shape (seq_len, batch_size, d_model).
            
# #         Returns:
# #             Tensor with positional encoding added.
# #         """
# #         x = x + self.pe[:x.size(0), :]
# #         return self.dropout(x)


# # class TransformerEncoderModel(nn.Module):
# #     """Transformer encoder model for solar wind time series data.
    
# #     Processes multivariate time series data using transformer encoder layers
# #     with positional encoding and multi-head attention mechanism.
    
# #     Args:
# #         num_input_variables: Number of input variables (features).
# #         input_sequence_length: Length of input sequences.
# #         d_model: Model dimension for transformer.
# #         nhead: Number of attention heads.
# #         num_layers: Number of transformer encoder layers.
# #         dim_feedforward: Dimension of feedforward network.
# #         dropout: Dropout rate.
# #     """
    
# #     def __init__(self, num_input_variables: int, input_sequence_length: int,
# #                  d_model: int = 256, nhead: int = 8, num_layers: int = 3,
# #                  dim_feedforward: int = 512, dropout: float = 0.1):
# #         super().__init__()
        
# #         # Validate input parameters
# #         if num_input_variables <= 0:
# #             raise ValueError(f"Number of input variables must be positive, got {num_input_variables}")
# #         if input_sequence_length <= 0:
# #             raise ValueError(f"Input sequence length must be positive, got {input_sequence_length}")
# #         if d_model <= 0:
# #             raise ValueError(f"Model dimension must be positive, got {d_model}")
# #         if d_model % nhead != 0:
# #             raise ValueError(f"Model dimension {d_model} must be divisible by number of heads {nhead}")
# #         if nhead <= 0 or num_layers <= 0:
# #             raise ValueError("Number of heads and layers must be positive")
# #         if not (0.0 <= dropout <= 1.0):
# #             raise ValueError("Dropout must be between 0 and 1")
        
# #         self.d_model = d_model
# #         self.input_sequence_length = input_sequence_length
# #         self.num_input_variables = num_input_variables
        
# #         # Input projection layer
# #         self.input_projection = nn.Linear(num_input_variables, d_model)
        
# #         # Positional encoding
# #         self.pos_encoder = PositionalEncoding(d_model, input_sequence_length, dropout)
        
# #         # Transformer encoder
# #         encoder_layer = nn.TransformerEncoderLayer(
# #             d_model=d_model,
# #             nhead=nhead,
# #             dim_feedforward=dim_feedforward,
# #             dropout=dropout,
# #             batch_first=False  # (seq_len, batch, d_model)
# #         )
# #         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
# #         # Output aggregation
# #         self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
# #         self.output_projection = nn.Linear(d_model, d_model)
        
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """Forward pass through transformer encoder.
        
# #         Args:
# #             x: Input tensor of shape (batch_size, seq_len, num_variables).
            
# #         Returns:
# #             Encoded features of shape (batch_size, d_model).
            
# #         Raises:
# #             ValueError: If input tensor has invalid dimensions.
# #         """
# #         # Validate input tensor dimensions
# #         if x.dim() != 3:
# #             raise ValueError(f"Expected 3D input tensor (batch, seq_len, variables), got {x.dim()}D tensor")
        
# #         batch_size, seq_len, num_vars = x.size()
        
# #         if seq_len != self.input_sequence_length:
# #             raise ValueError(f"Expected sequence length {self.input_sequence_length}, got {seq_len}")
# #         if num_vars != self.num_input_variables:
# #             raise ValueError(f"Expected {self.num_input_variables} variables, got {num_vars}")
        
# #         # Project to model dimension: (batch, seq_len, d_model)
# #         x = self.input_projection(x)
        
# #         # Transpose for transformer: (seq_len, batch, d_model)
# #         x = x.transpose(0, 1)
        
# #         # Add positional encoding
# #         x = self.pos_encoder(x)
        
# #         # Apply transformer encoder
# #         x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        
# #         # Transpose back: (batch, seq_len, d_model)
# #         x = x.transpose(0, 1)
        
# #         # Global average pooling: (batch, d_model, 1) -> (batch, d_model)
# #         x = x.transpose(1, 2)  # (batch, d_model, seq_len)
# #         x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
# #         # Final projection
# #         x = self.output_projection(x)
        
# #         return x


# # class CrossModalAttention(nn.Module):
# #     """Cross-modal attention mechanism for feature fusion.
    
# #     Enables interaction between transformer and ConvLSTM features
# #     through mutual attention computation.
    
# #     Args:
# #         feature_dim: Dimension of input features.
# #         num_heads: Number of attention heads.
# #         dropout: Dropout rate.
# #     """
    
# #     def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
# #         super().__init__()
        
# #         if feature_dim <= 0:
# #             raise ValueError("Feature dimension must be positive")
# #         if feature_dim % num_heads != 0:
# #             raise ValueError("Feature dimension must be divisible by number of heads")
# #         if num_heads <= 0:
# #             raise ValueError("Number of heads must be positive")
        
# #         self.feature_dim = feature_dim
# #         self.num_heads = num_heads
# #         self.head_dim = feature_dim // num_heads
        
# #         # Query, Key, Value projections
# #         self.q_proj = nn.Linear(feature_dim, feature_dim)
# #         self.k_proj = nn.Linear(feature_dim, feature_dim)
# #         self.v_proj = nn.Linear(feature_dim, feature_dim)
        
# #         # Output projection
# #         self.out_proj = nn.Linear(feature_dim, feature_dim)
# #         self.dropout = nn.Dropout(dropout)
        
# #         # Layer normalization
# #         self.norm = nn.LayerNorm(feature_dim)
        
# #     def forward(self, query_features: torch.Tensor, key_value_features: torch.Tensor) -> torch.Tensor:
# #         """Forward pass of cross-modal attention.
        
# #         Args:
# #             query_features: Query features of shape (batch, feature_dim).
# #             key_value_features: Key-Value features of shape (batch, feature_dim).
            
# #         Returns:
# #             Attended features of shape (batch, feature_dim).
# #         """
# #         batch_size = query_features.size(0)
        
# #         # Project to Q, K, V
# #         Q = self.q_proj(query_features)  # (batch, feature_dim)
# #         K = self.k_proj(key_value_features)  # (batch, feature_dim)
# #         V = self.v_proj(key_value_features)  # (batch, feature_dim)
        
# #         # Reshape for multi-head attention
# #         Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
# #         K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
# #         V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        
# #         # Compute attention scores
# #         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
# #         attention_weights = F.softmax(attention_scores, dim=-1)
# #         attention_weights = self.dropout(attention_weights)
        
# #         # Apply attention to values
# #         attended = torch.matmul(attention_weights, V)  # (batch, num_heads, 1, head_dim)
        
# #         # Concatenate heads
# #         attended = attended.transpose(1, 2).contiguous().view(batch_size, self.feature_dim)
        
# #         # Output projection and residual connection
# #         output = self.out_proj(attended)
# #         output = self.norm(output + query_features)
        
# #         return output


# # class CrossModalFusion(nn.Module):
# #     """Cross-modal fusion module for combining transformer and ConvLSTM features.
    
# #     Uses bidirectional cross-attention to enable interaction between
# #     solar wind (transformer) and image (ConvLSTM) features.
    
# #     Args:
# #         feature_dim: Dimension of input features.
# #         num_heads: Number of attention heads.
# #         dropout: Dropout rate.
# #     """
    
# #     def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
# #         super().__init__()
        
# #         self.feature_dim = feature_dim
        
# #         # Cross-attention layers
# #         self.transformer_to_convlstm = CrossModalAttention(feature_dim, num_heads, dropout)
# #         self.convlstm_to_transformer = CrossModalAttention(feature_dim, num_heads, dropout)
        
# #         # Feature alignment and combination
# #         self.feature_gate = nn.Sequential(
# #             nn.Linear(feature_dim * 2, feature_dim),
# #             nn.Sigmoid()
# #         )
        
# #         self.combination_layer = nn.Sequential(
# #             nn.Linear(feature_dim * 2, feature_dim),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(feature_dim, feature_dim)
# #         )
        
# #         # Final normalization
# #         self.final_norm = nn.LayerNorm(feature_dim)
        
# #     def forward(self, transformer_features: torch.Tensor, convlstm_features: torch.Tensor) -> torch.Tensor:
# #         """Forward pass of cross-modal fusion.
        
# #         Args:
# #             transformer_features: Features from transformer of shape (batch, feature_dim).
# #             convlstm_features: Features from ConvLSTM of shape (batch, feature_dim).
            
# #         Returns:
# #             Fused features of shape (batch, feature_dim).
# #         """
# #         # Cross-attention in both directions
# #         transformer_attended = self.transformer_to_convlstm(transformer_features, convlstm_features)
# #         convlstm_attended = self.convlstm_to_transformer(convlstm_features, transformer_features)
        
# #         # Concatenate attended features
# #         concatenated = torch.cat([transformer_attended, convlstm_attended], dim=1)
        
# #         # Compute gating weights
# #         gate_weights = self.feature_gate(concatenated)
        
# #         # Weighted combination
# #         weighted_transformer = gate_weights * transformer_attended
# #         weighted_convlstm = (1 - gate_weights) * convlstm_attended
        
# #         # Final combination
# #         combined = torch.cat([weighted_transformer, weighted_convlstm], dim=1)
# #         fused_features = self.combination_layer(combined)
        
# #         # Add residual connection and normalize
# #         residual = (transformer_features + convlstm_features) / 2
# #         output = self.final_norm(fused_features + residual)
        
# #         return output


# # class MultiModalModel(nn.Module):
# #     """Enhanced multi-modal classification model with cross-modal fusion.
    
# #     Integrates transformer processing of solar wind time series with 
# #     ConvLSTM processing of image sequences, enhanced with cross-modal
# #     attention fusion. Outputs binary classification for each target variable
# #     across multiple time groups.
    
# #     Args:
# #         num_input_variables: Number of solar wind input variables.
# #         input_sequence_length: Length of input sequences.
# #         num_target_variables: Number of target variables to predict.
# #         num_target_groups: Number of time groups for classification.
# #         transformer_d_model: Transformer model dimension.
# #         transformer_nhead: Number of transformer attention heads.
# #         transformer_num_layers: Number of transformer encoder layers.
# #         transformer_dim_feedforward: Transformer feedforward dimension.
# #         transformer_dropout: Transformer dropout rate.
# #         convlstm_input_channels: Number of input channels for ConvLSTM.
# #         convlstm_hidden_channels: Number of hidden channels for ConvLSTM.
# #         convlstm_kernel_size: Kernel size for ConvLSTM.
# #         convlstm_num_layers: Number of ConvLSTM layers.
# #         fusion_num_heads: Number of attention heads for cross-modal fusion.
# #         fusion_dropout: Dropout rate for fusion module.
# #     """
    
# #     def __init__(
# #         self, num_input_variables, input_sequence_length,
# #         num_target_variables, num_target_groups,
# #         transformer_d_model, transformer_nhead, transformer_num_layers,
# #         transformer_dim_feedforward, transformer_dropout,
# #         convlstm_input_channels, convlstm_hidden_channels,
# #         convlstm_kernel_size, convlstm_num_layers,
# #         fusion_num_heads=4, fusion_dropout=0.1
# #         ):
# #         super(MultiModalModel, self).__init__()

# #         # Validate input parameters
# #         if num_target_variables <= 0 or num_target_groups <= 0:
# #             raise ValueError("Target variables and number of groups must be positive")

# #         self.transformer_model = TransformerEncoderModel(
# #             num_input_variables=num_input_variables,
# #             input_sequence_length=input_sequence_length,
# #             d_model=transformer_d_model,
# #             nhead=transformer_nhead,
# #             num_layers=transformer_num_layers,
# #             dim_feedforward=transformer_dim_feedforward,
# #             dropout=transformer_dropout)

# #         self.convlstm_model = ConvLSTMModel(
# #             input_channels=convlstm_input_channels,
# #             hidden_channels=convlstm_hidden_channels,
# #             kernel_size=convlstm_kernel_size,
# #             num_layers=convlstm_num_layers,
# #             output_dim=transformer_d_model)  # Same dimension as transformer

# #         # Cross-modal fusion module
# #         self.cross_modal_fusion = CrossModalFusion(
# #             feature_dim=transformer_d_model,
# #             num_heads=fusion_num_heads,
# #             dropout=fusion_dropout
# #         )
        
# #         # Classification head - outputs logits for binary classification
# #         # Output shape: (batch_size, num_target_groups, num_target_variables)
# #         self.classification_head = nn.Sequential(
# #             nn.Linear(transformer_d_model, transformer_d_model // 2),
# #             nn.ReLU(),
# #             nn.Dropout(fusion_dropout),
# #             nn.Linear(transformer_d_model // 2, num_target_groups * num_target_variables)
# #         )

# #         self.num_target_variables = num_target_variables
# #         self.num_target_groups = num_target_groups

# #     def forward(self, solar_wind_input, image_input):
# #         """Forward pass through the enhanced multi-modal classification model.
        
# #         Args:
# #             solar_wind_input: Solar wind data of shape (batch, sequence, variables).
# #             image_input: Image data of shape (batch, channels, seq_len, height, width).
            
# #         Returns:
# #             Classification logits of shape (batch, num_target_groups, num_target_variables).
# #             Apply sigmoid to get probabilities for binary classification.
            
# #         Raises:
# #             ValueError: If inputs are None or batch sizes don't match.
# #         """
# #         # Validate input tensors
# #         if solar_wind_input is None or image_input is None:
# #             raise ValueError("Both solar_wind_input and image_input must be provided")
        
# #         if solar_wind_input.size(0) != image_input.size(0):
# #             raise ValueError(f"Batch sizes must match: solar_wind={solar_wind_input.size(0)}, image={image_input.size(0)}")

# #         # Extract features from each modality
# #         transformer_features = self.transformer_model(solar_wind_input)
# #         convlstm_features = self.convlstm_model(image_input)
        
# #         # Apply cross-modal fusion
# #         fused_features = self.cross_modal_fusion(transformer_features, convlstm_features)
        
# #         # Generate classification logits
# #         logits = self.classification_head(fused_features)
        
# #         # Reshape to (batch, num_target_groups, num_target_variables)
# #         output = logits.reshape(logits.size(0), self.num_target_groups, self.num_target_variables)
        
# #         return output


# # def create_model(config, logger=None):
# #     """Create MultiModalModel instance from configuration config.
    
# #     Args:
# #         config: Configuration object containing model parameters.
# #         logger: Optional logger for output.
        
# #     Returns:
# #         MultiModalModel instance configured for binary classification.
# #     """
# #     # Calculate number of target groups from target sequence length and group size
    
# #     num_input_variables = len(config.data.input_variables)
# #     num_target_groups = 1
# #     num_target_variables = len(config.data.target_variables)
# #     message = f"Enhanced MultiModalModel (Classification) with Transformer, ConvLSTM, and CrossModalFusion created. Output shape: (batch, {num_target_groups}, {num_target_variables})"

# #     if logger:
# #         logger.info(message)
# #     else:
# #         print(message)
        
# #     return MultiModalModel(
# #         num_input_variables=num_input_variables,
# #         input_sequence_length=config.data.input_sequence_length,
# #         num_target_variables=num_target_variables,
# #         num_target_groups=num_target_groups,
# #         transformer_d_model=config.model.transformer_d_model,
# #         transformer_nhead=config.model.transformer_nhead,
# #         transformer_num_layers=config.model.transformer_num_layers,
# #         transformer_dim_feedforward=config.model.transformer_dim_feedforward,
# #         transformer_dropout=config.model.transformer_dropout,
# #         convlstm_input_channels=config.model.convlstm_input_channels,
# #         convlstm_hidden_channels=config.model.convlstm_hidden_channels,
# #         convlstm_kernel_size=config.model.convlstm_kernel_size,
# #         convlstm_num_layers=config.model.convlstm_num_layers,
# #         fusion_num_heads=config.model.fusion_num_heads,
# #         fusion_dropout=config.model.fusion_dropout
# #     )


# # @hydra.main(config_path="./configs", version_base=None)
# # def main(config):

# #     model = create_model(config, logger=None)

# #     num_input_variables = len(config.data.input_variables)
# #     num_target_variables = len(config.data.target_variables)
# #     num_target_groups = 1

# #     sdo_shape = (
# #         config.experiment.batch_size,
# #         config.model.convlstm_input_channels,
# #         # config.model.convlstm_input_image_frames,
# #         20,
# #         config.data.sdo_image_size,
# #         config.data.sdo_image_size
# #     )
    
# #     inputs_shape = (
# #         config.experiment.batch_size,
# #         config.data.input_sequence_length,
# #         num_input_variables
# #     )

# #     targets_shape = (
# #         config.experiment.batch_size,
# #         1,
# #         num_target_variables
# #     )

# #     sdo = torch.randn(sdo_shape)
# #     inputs = torch.randn(inputs_shape)

# #     outputs = model(
# #         solar_wind_input=inputs,
# #         image_input=sdo
# #     )

# #     targets = torch.ones(targets_shape)

# #     print("Output shape:", outputs.shape)  # Expected: (batch_size, num_target
# #     print(outputs)

# #     loss = nn.BCEWithLogitsLoss()
# #     output_loss = loss(outputs, targets)
# #     print("Loss:", output_loss.item())


# # if __name__ == "__main__":
# #     main()

# #     # sdo_shape = (
# #     #     config.batch_size,
# #     #     config.convlstm_input_channels,
# #     #     config.convlstm_input_image_frames,
# #     #     config.image_size,
# #     #     config.image_size
# #     # )
    
# #     # inputs_shape = (
# #     #     config.batch_size,
# #     #     config.input_sequence_length,
# #     #     config.num_input_variables
# #     # )

# #     # targets_shape = (
# #     #     config.batch_size,
# #     #     1,
# #     #     config.num_target_variables
# #     # )

# #     # sdo = torch.randn(sdo_shape)
# #     # inputs = torch.randn(inputs_shape)

# #     # outputs = model(
# #     #     solar_wind_input=inputs,
# #     #     image_input=sdo
# #     # )

# #     # targets = torch.ones(targets_shape)

# #     # print("Output shape:", outputs.shape)  # Expected: (batch_size, num_target
# #     # print(outputs)

# #     # loss = nn.BCEWithLogitsLoss()
# #     # output_loss = loss(outputs, targets)
# #     # print("Loss:", output_loss.item())






# #     # sdo = torch.randn(
# #     #     )

# #     #     inputs = data_dict['inputs']
# #     #     targets = data_dict['targets']
# #     #     file_names = data_dict['file_names']
# #     #     print(sdo.shape, inputs.shape, targets.shape, file_names)
# #     #     print(targets)
# #     #     if i >= 2:
# #     #         break