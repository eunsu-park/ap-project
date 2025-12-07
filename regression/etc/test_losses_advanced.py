"""
Advanced loss function tests.

Demonstrates special-purpose losses that require specific output formats:
- QuantileLoss (multi-quantile predictions)
- MultiTaskLoss with outlier detection
"""

import torch
import torch.nn as nn
import hydra

from config import Config
from losses import QuantileLoss, MultiTaskLoss


@hydra.main(config_path="./configs", version_base=None)
def main(hydra_cfg):
    """Test advanced loss functions."""
    config = Config.from_hydra(hydra_cfg)
    
    print("=" * 80)
    print("Testing Advanced Loss Functions")
    print("=" * 80)
    
    # Test data
    batch_size = 4
    seq_len = 8
    num_vars = 1
    
    # Test 1: QuantileLoss
    print("\n" + "=" * 80)
    print("Test 1: QuantileLoss (Multi-Quantile Predictions)")
    print("=" * 80)
    
    print("\nQuantileLoss requires predictions in format:")
    print("  (batch_size, seq_len, feature_dim, num_quantiles)")
    print("\nThis allows predicting multiple quantiles simultaneously,")
    print("e.g., [0.1, 0.5, 0.9] for 10th, 50th (median), 90th percentiles")
    
    # Standard targets
    targets = torch.randn(batch_size, seq_len, num_vars) * 100 + 100
    
    # Multi-quantile predictions (requires model to output multiple quantiles)
    quantiles = [0.1, 0.5, 0.9]
    num_quantiles = len(quantiles)
    
    # Simulate model that predicts multiple quantiles
    # Shape: (batch, seq, features, quantiles)
    quantile_preds = torch.randn(batch_size, seq_len, num_vars, num_quantiles) * 100 + 100
    
    # Ensure quantiles are ordered: q_0.1 < q_0.5 < q_0.9
    quantile_preds = torch.sort(quantile_preds, dim=-1)[0]
    
    print(f"\nTargets shape: {targets.shape}")
    print(f"Quantile predictions shape: {quantile_preds.shape}")
    print(f"  - Predicting {num_quantiles} quantiles: {quantiles}")
    
    # Create and test QuantileLoss
    quantile_loss = QuantileLoss(quantiles=quantiles)
    loss_val = quantile_loss(quantile_preds, targets)
    
    print(f"\nQuantileLoss: {loss_val.item():.4f}")
    
    # Show prediction intervals
    print(f"\nPrediction Intervals (first sample, first timestep):")
    q_low = quantile_preds[0, 0, 0, 0].item()  # 10th percentile
    q_med = quantile_preds[0, 0, 0, 1].item()  # 50th percentile (median)
    q_high = quantile_preds[0, 0, 0, 2].item()  # 90th percentile
    actual = targets[0, 0, 0].item()
    
    print(f"  10th percentile: {q_low:.2f}")
    print(f"  50th percentile (median): {q_med:.2f}")
    print(f"  90th percentile: {q_high:.2f}")
    print(f"  80% prediction interval: [{q_low:.2f}, {q_high:.2f}]")
    print(f"  Actual target: {actual:.2f}")
    
    # Test 2: MultiTaskLoss
    print("\n" + "=" * 80)
    print("Test 2: MultiTaskLoss (Regression + Outlier Detection)")
    print("=" * 80)
    
    print("\nMultiTaskLoss can work in two modes:")
    print("  1. Regression only (outlier_logits=None)")
    print("  2. Regression + Outlier Detection (with outlier_logits)")
    
    # Standard predictions
    preds = torch.randn(batch_size, seq_len, num_vars) * 100 + 100
    
    # Mode 1: Regression only
    print("\n[Mode 1] Regression Only:")
    multitask_loss = MultiTaskLoss(regression_loss_type='mse')
    loss_val1 = multitask_loss(preds, targets)
    print(f"  Loss: {loss_val1.item():.4f}")
    print(f"  (Only regression loss with temporal weighting)")
    
    # Mode 2: Regression + Outlier Detection
    print("\n[Mode 2] Regression + Outlier Detection:")
    
    # Simulate outlier detection logits (binary classification per timestep)
    # Shape: (batch, seq)
    outlier_logits = torch.randn(batch_size, seq_len)
    
    print(f"  Outlier logits shape: {outlier_logits.shape}")
    print(f"  (Binary outlier probability for each timestep)")
    
    multitask_loss2 = MultiTaskLoss(
        regression_loss_type='huber',
        beta=0.5,
        outlier_loss_weight=0.3
    )
    loss_val2 = multitask_loss2(preds, targets, outlier_logits=outlier_logits)
    print(f"  Combined Loss: {loss_val2.item():.4f}")
    print(f"  (Regression + 0.3 * Outlier Detection)")
    
    # Show generated outlier labels
    with torch.no_grad():
        outlier_labels = multitask_loss2._generate_outlier_labels(targets)
    
    print(f"\n  Auto-generated outlier labels (z-score > 2.0):")
    print(f"    Shape: {outlier_labels.shape}")
    print(f"    Outlier ratio: {outlier_labels.mean().item():.2%}")
    
    # Test 3: Model Integration Example
    print("\n" + "=" * 80)
    print("Test 3: How to Use in Your Model")
    print("=" * 80)
    
    print("\n[Example 1] QuantileLoss - Model needs to output multiple quantiles:")
    print("""
class MultiQuantileModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ...  # Your transformer/LSTM/etc
        
        # Final layer outputs multiple quantiles
        self.quantile_head = nn.Linear(hidden_dim, num_quantiles)
    
    def forward(self, x):
        features = self.backbone(x)
        # Output shape: (batch, seq, features, num_quantiles)
        quantiles = self.quantile_head(features)
        return quantiles

# Training
model = MultiQuantileModel()
criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

outputs = model(inputs)  # Shape: (batch, seq, features, 3)
loss = criterion(outputs, targets)  # targets: (batch, seq, features)
""")
    
    print("\n[Example 2] MultiTaskLoss - Optional outlier detection head:")
    print("""
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ...
        self.regression_head = nn.Linear(hidden_dim, num_vars)
        self.outlier_head = nn.Linear(hidden_dim, 1)  # Binary per timestep
    
    def forward(self, x):
        features = self.backbone(x)
        regression_out = self.regression_head(features)
        outlier_logits = self.outlier_head(features).squeeze(-1)
        return regression_out, outlier_logits

# Training
model = MultiTaskModel()
criterion = MultiTaskLoss(outlier_loss_weight=0.3)

reg_out, outlier_logits = model(inputs)
loss = criterion(reg_out, targets, outlier_logits=outlier_logits)
""")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print("\n✓ QuantileLoss:")
    print("  - Use for uncertainty quantification")
    print("  - Requires model to output multiple quantiles")
    print("  - Provides prediction intervals")
    
    print("\n✓ MultiTaskLoss:")
    print("  - Use for multi-task learning")
    print("  - Can work with or without outlier detection")
    print("  - Automatic outlier labeling based on z-scores")
    
    print("\n" + "=" * 80)
    print("✓ Advanced loss tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()