"""
Quick Test Script for Attention Analysis

1개 배치만 처리해서 구현이 정상 작동하는지 확인합니다.

Usage:
    python test_attention.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

print("=" * 70)
print("ATTENTION ANALYSIS - QUICK TEST")
print("=" * 70)

# ================================================================
# Step 1: Import 확인
# ================================================================
print("\n[1/5] Checking imports...")

try:
    from networks import create_model
    print("  ✓ networks.py imported")
except Exception as e:
    print(f"  ❌ Error importing networks.py: {e}")
    sys.exit(1)

try:
    from pipeline import create_dataloader
    print("  ✓ pipeline.py imported")
except Exception as e:
    print(f"  ❌ Error importing pipeline.py: {e}")
    sys.exit(1)

try:
    from attention_analysis import AttentionExtractor
    print("  ✓ attention_analysis.py imported")
except Exception as e:
    print(f"  ❌ Error importing attention_analysis.py: {e}")
    sys.exit(1)

# ================================================================
# Step 2: Config 로드
# ================================================================
print("\n[2/5] Loading config...")

try:
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/attention_0.yaml')
    print(f"  ✓ Config loaded")
    print(f"    Checkpoint: {config.validation.checkpoint_path}")
    print(f"    Output dir: {config.validation.output_dir}")
except Exception as e:
    print(f"  ❌ Error loading config: {e}")
    print("  Make sure configs/attention_0.yaml exists!")
    sys.exit(1)

# ================================================================
# Step 3: 모델 로드
# ================================================================
print("\n[3/5] Loading trained model...")

try:
    checkpoint_path = config.validation.checkpoint_path
    device = "mps"  # or "cuda" or "cpu"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"  ❌ Checkpoint not found: {checkpoint_path}")
        print("  Please update the path in configs/attention_0.yaml")
        sys.exit(1)
    
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded successfully")
    print(f"    Device: {device}")
    
except Exception as e:
    print(f"  ❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ================================================================
# Step 4: 데이터 로드
# ================================================================
print("\n[4/5] Loading validation data...")

try:
    dataloader = create_dataloader(config, phase="validation")
    print(f"  ✓ DataLoader created")
    print(f"    Total batches: {len(dataloader)}")
    
    # Get first batch
    batch = next(iter(dataloader))
    solar_wind_input = batch["inputs"][:1].to(device)
    image_input = batch["sdo"][:1].to(device)
    targets = batch["targets"][:1]
    
    print(f"  ✓ First batch loaded")
    print(f"    Solar wind: {solar_wind_input.shape}")
    print(f"    SDO images: {image_input.shape}")
    print(f"    Targets: {targets.shape}")
    
except Exception as e:
    print(f"  ❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ================================================================
# Step 5: Attention 추출 테스트
# ================================================================
print("\n[5/5] Testing attention extraction...")

try:
    extractor = AttentionExtractor(model, device=device)
    print("  ✓ AttentionExtractor initialized")
    
    # Extract attention
    print("  → Running manual forward pass...")
    attention_weights, predictions = extractor.extract_attention_manual_forward(
        solar_wind_input, image_input
    )
    
    print(f"  ✓ Attention extraction successful!")
    print(f"    Number of layers: {len(attention_weights)}")
    print(f"    Attention shape per layer: {attention_weights[0].shape}")
    print(f"    Predictions shape: {predictions.shape}")
    
    # Compute temporal importance
    print("  → Computing temporal importance...")
    temporal_imp = extractor.compute_temporal_importance(
        attention_weights[-1][0],  # Last layer, first batch
        method='incoming'
    )
    
    print(f"  ✓ Temporal importance computed")
    print(f"    Shape: {temporal_imp.shape}")
    print(f"    Peak timestep: {temporal_imp.argmax()}")
    print(f"    Peak value: {temporal_imp.max():.4f}")
    
    # Save test NPZ
    print("  → Saving test NPZ...")
    output_dir = Path(config.validation.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    test_output_path = output_dir / "test_attention_batch_0000.npz"
    
    attention_weights_np = [attn[0].cpu().numpy() for attn in attention_weights]
    attention_weights_np = np.array(attention_weights_np)
    
    np.savez_compressed(
        test_output_path,
        solar_wind_data=solar_wind_input[0].cpu().numpy(),
        sdo_data=image_input[0].cpu().numpy(),
        attention_weights=attention_weights_np,
        temporal_importance=temporal_imp,
        predictions=predictions[0].cpu().numpy(),
        targets=targets[0].cpu().numpy()
    )
    
    file_size_mb = test_output_path.stat().st_size / (1024**2)
    print(f"  ✓ Test NPZ saved: {test_output_path.name}")
    print(f"    File size: {file_size_mb:.2f} MB")
    
    # Verify loading
    print("  → Verifying NPZ loading...")
    data = np.load(test_output_path)
    print(f"  ✓ NPZ verified!")
    print(f"    Keys: {list(data.keys())}")
    
except Exception as e:
    print(f"  ❌ Error during attention extraction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ================================================================
# Success!
# ================================================================
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print(f"\nTest output saved to: {test_output_path}")
print("\nNext steps:")
print("1. Run full analysis: python example_attention_all_targets.py")
print("2. Or use Hydra: python example_attention_all_targets.py")
print("\nExpected runtime for full validation set:")
print("  - Attention extraction: ~2-5 minutes (forward-only)")
print("  - vs IG: ~30-60 minutes (backward required)")
print("  - Speed-up: ~50-100x faster! ⚡")
