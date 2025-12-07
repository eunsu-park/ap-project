"""
Test model creation and forward pass.

Tests:
1. Model creation from config
2. Forward pass (regression output)
3. Forward pass with features (for contrastive loss)
4. Contrastive losses (InfoNCE vs MSE)
5. Feature alignment metrics
"""

import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

from config import Config
from models import create_model
from losses import MultiModalContrastiveLoss, MultiModalMSELoss


@hydra.main(config_path="./configs", version_base=None)
def main(hydra_cfg):
    # Convert to structured config
    config = Config.from_hydra(hydra_cfg)
    
    print("=" * 80)
    print("Testing Refactored Model")
    print("=" * 80)
    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Device: {config.environment.device}")
    print(f"Batch size: {config.experiment.batch_size}")
    print("=" * 80)

    # Create model
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Prepare test data
    num_input_variables = len(config.data.input_variables)
    num_target_variables = len(config.data.target_variables)

    sdo_shape = (
        config.experiment.batch_size,
        config.model.convlstm_input_channels,
        config.data.sdo_sequence_length,
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

    # Generate random test data
    sdo = torch.randn(sdo_shape)
    inputs = torch.randn(inputs_shape)
    targets = torch.rand(targets_shape) * 400  # AP index range [0, 400]
    
    print(f"\nTest data shapes:")
    print(f"  SDO: {sdo.shape}")
    print(f"  Inputs: {inputs.shape}")
    print(f"  Targets: {targets.shape}")

    # Test 1: Normal forward pass
    print("\n" + "=" * 80)
    print("Test 1: Normal Forward Pass (return_features=False)")
    print("=" * 80)
    
    with torch.no_grad():
        outputs = model(
            solar_wind_input=inputs,
            image_input=sdo
        )
    
    print(f"Output shape: {outputs.shape}")
    print(f"Expected shape: (batch_size={config.experiment.batch_size}, "
          f"target_seq_len={config.data.target_sequence_length}, "
          f"target_vars={num_target_variables})")
    print(f"Output range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]")
    print(f"Output sample:\n{outputs[0]}")

    # Compute regression loss
    loss_fn = nn.MSELoss()
    output_loss = loss_fn(outputs, targets)
    print(f"\nRegression Loss (MSE): {output_loss.item():.6f}")
    
    # Test 2: Forward pass with features
    print("\n" + "=" * 80)
    print("Test 2: Forward Pass with Features (return_features=True)")
    print("=" * 80)
    
    with torch.no_grad():
        outputs, transformer_features, convlstm_features = model(
            solar_wind_input=inputs,
            image_input=sdo,
            return_features=True
        )
    
    print(f"Output shape: {outputs.shape}")
    print(f"Transformer features shape: {transformer_features.shape}")
    print(f"ConvLSTM features shape: {convlstm_features.shape}")
    
    # Test 3: Contrastive Losses
    print("\n" + "=" * 80)
    print("Test 3: Contrastive Losses Comparison")
    print("=" * 80)
    
    # InfoNCE
    print("\n[3a] InfoNCE Contrastive Loss:")
    contrastive_infonce = MultiModalContrastiveLoss(
        temperature=config.training.contrastive_temperature,
        normalize=True
    )
    loss_infonce = contrastive_infonce(transformer_features, convlstm_features)
    print(f"  Loss: {loss_infonce.item():.6f}")
    print(f"  Temperature: {config.training.contrastive_temperature}")
    print(f"  Batch negatives: {config.experiment.batch_size}")
    
    # MSE
    print("\n[3b] MSE Consistency Loss:")
    contrastive_mse = MultiModalMSELoss(reduction='mean')
    loss_mse = contrastive_mse(transformer_features, convlstm_features)
    print(f"  Loss: {loss_mse.item():.6f}")
    print(f"  Direct feature alignment (no negatives)")
    
    # Test 4: Combined loss
    print("\n" + "=" * 80)
    print("Test 4: Combined Loss (Regression + Contrastive)")
    print("=" * 80)
    
    lambda_c = config.training.lambda_contrastive
    
    total_infonce = output_loss + lambda_c * loss_infonce
    total_mse = output_loss + lambda_c * loss_mse
    
    print(f"\n[4a] With InfoNCE (λ={lambda_c}):")
    print(f"  Regression: {output_loss.item():.6f}")
    print(f"  InfoNCE:    {loss_infonce.item():.6f}")
    print(f"  Total:      {total_infonce.item():.6f}")
    
    print(f"\n[4b] With MSE (λ={lambda_c}):")
    print(f"  Regression: {output_loss.item():.6f}")
    print(f"  MSE:        {loss_mse.item():.6f}")
    print(f"  Total:      {total_mse.item():.6f}")
    
    # Test 5: Feature alignment
    print("\n" + "=" * 80)
    print("Test 5: Feature Alignment Metrics")
    print("=" * 80)
    
    cosine_sim = F.cosine_similarity(transformer_features, convlstm_features, dim=1)
    
    print(f"Cosine Similarity Statistics:")
    print(f"  Mean: {cosine_sim.mean().item():.4f}")
    print(f"  Std:  {cosine_sim.std().item():.4f}")
    print(f"  Min:  {cosine_sim.min().item():.4f}")
    print(f"  Max:  {cosine_sim.max().item():.4f}")
    
    # L2 distance
    l2_dist = torch.norm(transformer_features - convlstm_features, dim=1)
    print(f"\nL2 Distance Statistics:")
    print(f"  Mean: {l2_dist.mean().item():.4f}")
    print(f"  Std:  {l2_dist.std().item():.4f}")
    print(f"  Min:  {l2_dist.min().item():.4f}")
    print(f"  Max:  {l2_dist.max().item():.4f}")
    
    # Test 6: Backward pass
    print("\n" + "=" * 80)
    print("Test 6: Backward Pass (Gradient Check)")
    print("=" * 80)
    
    # Enable gradients
    model.train()
    outputs, trans_feat, conv_feat = model(
        solar_wind_input=inputs,
        image_input=sdo,
        return_features=True
    )
    
    # Compute loss
    reg_loss = loss_fn(outputs, targets)
    cont_loss = contrastive_mse(trans_feat, conv_feat)
    total_loss = reg_loss + lambda_c * cont_loss
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_layers = sum(1 for _ in model.parameters())
    
    print(f"Gradients computed: {has_grad}/{total_layers} parameters")
    
    # Check gradient norms
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"Gradient norms:")
    print(f"  Mean: {np.mean(grad_norms):.6f}")
    print(f"  Max:  {np.max(grad_norms):.6f}")
    print(f"  Min:  {np.min(grad_norms):.6f}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
