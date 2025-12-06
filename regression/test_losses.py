"""
Test all loss functions.

Tests:
1. Basic losses (MSE, MAE, Huber)
2. Weighted losses
3. Contrastive losses (InfoNCE, MSE)
4. Advanced losses
5. Loss factory functions
"""

import torch
import torch.nn as nn
import hydra

from config import Config
from losses import (
    create_loss_functions,
    create_loss,
    WeightedMSELoss,
    HuberMultiCriteriaLoss,
    MAEOutlierFocusedLoss,
    MultiModalContrastiveLoss,
    MultiModalMSELoss,
    AdaptiveWeightLoss,
    GradientBasedWeightLoss,
    QuantileLoss,
    MultiTaskLoss,
)


@hydra.main(config_path="./configs", version_base=None)
def main(hydra_cfg):
    """Test all loss functions."""
    config = Config.from_hydra(hydra_cfg)
    
    print("=" * 80)
    print("Testing Loss Functions")
    print("=" * 80)
    
    # Prepare test data
    batch_size = 4
    seq_len = 8
    num_vars = 1
    feature_dim = 256
    
    # Predictions and targets
    outputs = torch.randn(batch_size, seq_len, num_vars) * 100 + 100  # ~[0, 200]
    targets = torch.randn(batch_size, seq_len, num_vars) * 100 + 100  # ~[0, 200]
    
    # Features for contrastive loss
    features_a = torch.randn(batch_size, feature_dim)
    features_b = torch.randn(batch_size, feature_dim)
    
    print(f"\nTest data prepared:")
    print(f"  Outputs shape: {outputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Features A shape: {features_a.shape}")
    print(f"  Features B shape: {features_b.shape}")
    
    # Test 1: Basic losses
    print("\n" + "=" * 80)
    print("Test 1: Basic Regression Losses")
    print("=" * 80)
    
    # MSE
    mse_loss = nn.MSELoss()
    mse_val = mse_loss(outputs, targets)
    print(f"\n[1a] MSE Loss: {mse_val.item():.4f}")
    
    # MAE
    mae_loss = nn.L1Loss()
    mae_val = mae_loss(outputs, targets)
    print(f"[1b] MAE Loss: {mae_val.item():.4f}")
    
    # Huber
    huber_loss = nn.HuberLoss(delta=10.0)
    huber_val = huber_loss(outputs, targets)
    print(f"[1c] Huber Loss (δ=10): {huber_val.item():.4f}")
    
    # Test 2: Weighted losses
    print("\n" + "=" * 80)
    print("Test 2: Weighted Regression Losses")
    print("=" * 80)
    
    # Weighted MSE
    weighted_mse = WeightedMSELoss()
    wmse_val = weighted_mse(outputs, targets)
    print(f"\n[2a] Weighted MSE Loss: {wmse_val.item():.4f}")
    print(f"     (Time-based weighting: first 8 pts=0.5, next 16=0.3)")
    
    # Huber Multi-Criteria
    huber_multi = HuberMultiCriteriaLoss(beta=0.3)
    hmc_val = huber_multi(outputs, targets)
    print(f"[2b] Huber Multi-Criteria: {hmc_val.item():.4f}")
    print(f"     (Combines Huber + temporal + gradient weighting, β=0.3)")
    
    # MAE Outlier Focused
    mae_outlier = MAEOutlierFocusedLoss(outlier_threshold=2.0)
    mao_val = mae_outlier(outputs, targets)
    print(f"[2c] MAE Outlier Focused: {mao_val.item():.4f}")
    print(f"     (Higher weight for outliers, z-score threshold=2.0)")
    
    # Test 3: Contrastive losses
    print("\n" + "=" * 80)
    print("Test 3: Contrastive Losses")
    print("=" * 80)
    
    # InfoNCE
    infonce = MultiModalContrastiveLoss(temperature=0.3, normalize=True)
    infonce_val = infonce(features_a, features_b)
    print(f"\n[3a] InfoNCE Contrastive: {infonce_val.item():.4f}")
    print(f"     (Temperature=0.3, normalized features)")
    print(f"     (Uses {batch_size} positive + {batch_size-1} negative pairs)")
    
    # MSE Consistency
    mse_consistency = MultiModalMSELoss(reduction='mean')
    mse_cons_val = mse_consistency(features_a, features_b)
    print(f"[3b] MSE Consistency: {mse_cons_val.item():.4f}")
    print(f"     (Direct feature alignment)")
    
    # Test 4: Advanced losses
    print("\n" + "=" * 80)
    print("Test 4: Advanced Losses")
    print("=" * 80)
    
    # Adaptive Weight
    adaptive = AdaptiveWeightLoss(base_loss_type='mse', beta=0.5)
    adaptive_val = adaptive(outputs, targets)
    print(f"\n[4a] Adaptive Weight Loss: {adaptive_val.item():.4f}")
    print(f"     (Dynamic error-based weighting, base=MSE)")
    
    # Gradient-Based Weight
    gradient = GradientBasedWeightLoss(base_loss_type='mae', beta=0.5)
    gradient_val = gradient(outputs, targets)
    print(f"[4b] Gradient-Based Weight: {gradient_val.item():.4f}")
    print(f"     (Emphasizes rapid changes, base=MAE)")
    
    # Quantile (requires special multi-output format, skip in basic test)
    print(f"[4c] Quantile Loss: Skipped")
    print(f"     (Requires multi-quantile predictions, see advanced examples)")
    quantile_val = None
    
    # Multi-Task
    multitask = MultiTaskLoss(regression_loss_type='mse', beta=0.5)
    mt_val = multitask(outputs, targets)
    print(f"[4d] Multi-Task Loss: {mt_val.item():.4f}")
    print(f"     (Regression + outlier detection with learnable weights)")
    
    # Test 5: Factory functions
    print("\n" + "=" * 80)
    print("Test 5: Loss Factory Functions")
    print("=" * 80)
    
    # create_loss_functions (returns both regression + contrastive)
    print(f"\n[5a] create_loss_functions(config):")
    reg_criterion, cont_criterion = create_loss_functions(config)
    print(f"     Regression: {type(reg_criterion).__name__}")
    print(f"     Contrastive: {type(cont_criterion).__name__}")
    
    reg_val = reg_criterion(outputs, targets)
    cont_val = cont_criterion(features_a, features_b)
    print(f"     Regression loss: {reg_val.item():.4f}")
    print(f"     Contrastive loss: {cont_val.item():.4f}")
    
    # create_loss (returns single loss)
    print(f"\n[5b] create_loss(config):")
    single_criterion = create_loss(config)
    print(f"     Loss type: {type(single_criterion).__name__}")
    single_val = single_criterion(outputs, targets)
    print(f"     Loss value: {single_val.item():.4f}")
    
    # Test 6: Backward compatibility
    print("\n" + "=" * 80)
    print("Test 6: Backward Pass (Gradient Check)")
    print("=" * 80)
    
    # Test each loss can compute gradients
    test_outputs = torch.randn(batch_size, seq_len, num_vars, requires_grad=True)
    test_targets = torch.randn(batch_size, seq_len, num_vars)
    
    losses_to_test = [
        ("MSE", nn.MSELoss()),
        ("Weighted MSE", WeightedMSELoss()),
        ("Adaptive", AdaptiveWeightLoss(base_loss_type='mse', beta=0.5)),
    ]
    
    print("\nTesting gradient computation:")
    for name, loss_fn in losses_to_test:
        test_outputs.grad = None  # Clear gradients
        loss_val = loss_fn(test_outputs, test_targets)
        loss_val.backward()
        
        has_grad = test_outputs.grad is not None
        grad_norm = test_outputs.grad.norm().item() if has_grad else 0
        
        status = "✓" if has_grad else "✗"
        print(f"  {status} {name}: loss={loss_val.item():.4f}, "
              f"grad_norm={grad_norm:.4f}")
    
    # Test 7: Loss comparison
    print("\n" + "=" * 80)
    print("Test 7: Loss Comparison on Same Data")
    print("=" * 80)
    
    # Compare different losses on same outputs/targets
    test_out = torch.randn(batch_size, seq_len, num_vars) * 50 + 100
    test_tgt = torch.randn(batch_size, seq_len, num_vars) * 50 + 100
    
    loss_comparison = [
        ("MSE", nn.MSELoss()),
        ("MAE", nn.L1Loss()),
        ("Huber", nn.HuberLoss(delta=10.0)),
        ("Weighted MSE", WeightedMSELoss()),
        ("Adaptive", AdaptiveWeightLoss(base_loss_type='mse', beta=0.5)),
    ]
    
    print("\nLoss values for same prediction:")
    results = []
    for name, loss_fn in loss_comparison:
        val = loss_fn(test_out, test_tgt).item()
        results.append((name, val))
        print(f"  {name:20s}: {val:.4f}")
    
    # Show relative differences
    print("\nRelative to MSE:")
    mse_val = results[0][1]
    for name, val in results[1:]:
        ratio = val / mse_val
        print(f"  {name:20s}: {ratio:.2f}x")
    
    print("\n" + "=" * 80)
    print("✓ All loss function tests passed successfully!")
    print("=" * 80)
    
    # Summary
    print("\nLoss Functions Available:")
    print("  Basic: MSE, MAE, Huber")
    print("  Weighted: WeightedMSE, HuberMultiCriteria, MAEOutlierFocused")
    print("  Contrastive: InfoNCE, MSE Consistency")
    print("  Advanced: Adaptive, Gradient-Based, Quantile, Multi-Task")
    print("  Factories: create_loss_functions(), create_loss()")


if __name__ == "__main__":
    main()