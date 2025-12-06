import time
import random

import numpy as np
import torch
import torch.nn as nn
import hydra
import torch.nn.functional as F

from models import create_model
from losses import create_loss_functions, MultiModalContrastiveLoss, MultiModalMSELoss


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    model = create_model(config, logger=None)

    num_input_variables = len(config.data.input_variables)
    num_target_variables = len(config.data.target_variables)

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
        config.data.target_sequence_length,
        num_target_variables
    )

    sdo = torch.randn(sdo_shape)
    inputs = torch.randn(inputs_shape)
    targets = torch.rand(targets_shape) * 400
    print(sdo.shape)
    print(inputs.shape)
    print(targets.shape)


    # Test 1: Normal forward pass (backward compatibility)
    print("=" * 60)
    print("Test 1: Normal forward pass (return_features=False)")
    print("=" * 60)
    outputs = model(
        solar_wind_input=inputs,
        image_input=sdo
    )
    # Generate realistic targets in ap_Index range [0, 400]
    print("Output shape:", outputs.shape)  # Expected: (batch_size, num_target_groups, num_target_variables)
    print("Output sample:\n", outputs[0])

    # Use MSE loss for regression
    loss = nn.MSELoss()
    output_loss = loss(outputs, targets)
    print("Regression Loss (MSE):", output_loss.item())
    
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
    print(f"  Regression Loss (MSE): {output_loss.item():.6f}")
    print(f"  InfoNCE Loss:          {contrastive_loss_infonce.item():.6f}")
    print(f"  Total Loss (λ={lambda_contrastive}): {total_loss_infonce.item():.6f}")
    
    print("\n[4b] With MSE:")
    total_loss_mse = output_loss + lambda_contrastive * mse_loss
    print(f"  Regression Loss (MSE): {output_loss.item():.6f}")
    print(f"  MSE Loss:              {mse_loss.item():.6f}")
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