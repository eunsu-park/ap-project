"""
Cross-modal fusion components for multimodal learning.

This module provides attention mechanisms for fusing features from
different modalities (transformer and ConvLSTM).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
