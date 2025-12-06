"""
Main multimodal model for solar wind prediction.

This module integrates transformer, ConvLSTM, and cross-modal fusion
components into a unified model for regression prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Union

from .transformer import TransformerEncoderModel
from .convlstm import ConvLSTMModel
from .fusion import CrossModalFusion


class MultiModalModel(nn.Module):
    """Enhanced multi-modal regression model with cross-modal fusion.
    
    Integrates transformer processing of solar wind time series with 
    ConvLSTM processing of image sequences, enhanced with cross-modal
    attention fusion. Outputs regression predictions for each target variable
    across multiple time groups.
    
    Args:
        num_input_variables: Number of solar wind input variables.
        input_sequence_length: Length of input sequences.
        num_target_variables: Number of target variables to predict.
        num_target_groups: Number of time groups for prediction.
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
        self, num_input_variables: int, input_sequence_length: int,
        num_target_variables: int, num_target_groups: int,
        transformer_d_model: int, transformer_nhead: int, transformer_num_layers: int,
        transformer_dim_feedforward: int, transformer_dropout: float,
        convlstm_input_channels: int, convlstm_hidden_channels: int,
        convlstm_kernel_size: int, convlstm_num_layers: int,
        fusion_num_heads: int = 4, fusion_dropout: float = 0.1
        ):
        super().__init__()

        # Validate input parameters
        if num_target_variables <= 0 or num_target_groups <= 0:
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
        # Output shape: (batch_size, num_target_groups, num_target_variables)
        self.regression_head = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(transformer_d_model // 2, num_target_groups * num_target_variables)
        )

        self.num_target_variables = num_target_variables
        self.num_target_groups = num_target_groups

    def forward(
        self, 
        solar_wind_input: torch.Tensor, 
        image_input: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through the enhanced multi-modal regression model.
        
        Args:
            solar_wind_input: Solar wind data of shape (batch, sequence, variables).
            image_input: Image data of shape (batch, channels, seq_len, height, width).
            return_features: If True, return intermediate features for contrastive loss.
            
        Returns:
            If return_features=False:
                Regression predictions of shape (batch, num_target_groups, num_target_variables).
            If return_features=True:
                Tuple of (predictions, transformer_features, convlstm_features).
            
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
        
        # Generate regression predictions
        predictions = self.regression_head(fused_features)
        
        # Reshape to (batch, num_target_groups, num_target_variables)
        output = predictions.reshape(predictions.size(0), self.num_target_groups, self.num_target_variables)
        
        # # Optional: Clamp values to valid range [0, 400] during inference
        # if not self.training:
        #     output = torch.clamp(output, min=0, max=400)
        
        # Return features if requested (for contrastive loss)
        if return_features:
            return output, transformer_features, convlstm_features
        
        return output
