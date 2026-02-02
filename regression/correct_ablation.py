"""
Correct Ablation Test - Feature Level
"""

import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path

from networks import create_model
from pipeline import create_dataloader


def load_trained_model(checkpoint_path: str, config: DictConfig, device: str = 'cuda'):
    """훈련된 모델 로드"""
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
    
    print(f"✓ Model loaded from {checkpoint_path}")
    return model


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """메인 실행 함수"""
    
    print("=" * 70)
    print("CORRECT ABLATION TEST - Feature Level")
    print("=" * 70)
    
    checkpoint_path = config.validation.checkpoint_path
    device = "mps"
    
    model = load_trained_model(checkpoint_path, config, device)
    dataloader = create_dataloader(config, phase="validation")
    print("✓ DataLoader loaded\n")
    
    # Get batch
    batch = next(iter(dataloader))
    solar_wind = batch["inputs"][:1].to(device)
    images = batch["sdo"][:1].to(device)
    
    print("=" * 70)
    print("TEST 1: Input-Level Ablation (Wrong Method)")
    print("=" * 70)
    
    with torch.no_grad():
        # Normal
        out_normal = model(solar_wind, images)
        
        # Zero images
        out_zero_img = model(solar_wind, torch.zeros_like(images))
        
        diff_input = (out_normal - out_zero_img).abs().mean().item()
        
        print(f"\nZeroing images:")
        print(f"  |Normal - Zero_images|: {diff_input:.6e}")
        print(f"  → This is WRONG because ConvLSTM produces non-zero output from zero input (bias)")
    
    print("\n" + "=" * 70)
    print("TEST 2: Feature-Level Ablation (Correct Method)")
    print("=" * 70)
    
    with torch.no_grad():
        # Extract features
        transformer_feat = model.transformer_model(solar_wind)
        convlstm_feat = model.convlstm_model(images)
        
        print(f"\nFeature shapes:")
        print(f"  Transformer: {transformer_feat.shape}")
        print(f"  ConvLSTM: {convlstm_feat.shape}")
        
        print(f"\nConvLSTM features from real images:")
        print(f"  Mean: {convlstm_feat.mean().item():.6f}")
        print(f"  Std: {convlstm_feat.std().item():.6f}")
        
        # What does ConvLSTM produce for zero images?
        zero_images = torch.zeros_like(images)
        convlstm_from_zero = model.convlstm_model(zero_images)
        
        print(f"\nConvLSTM features from ZERO images:")
        print(f"  Mean: {convlstm_from_zero.mean().item():.6f}")
        print(f"  Std: {convlstm_from_zero.std().item():.6f}")
        
        diff_convlstm = (convlstm_feat - convlstm_from_zero).abs().mean().item()
        print(f"\n  |ConvLSTM(real) - ConvLSTM(zero)|: {diff_convlstm:.6e}")
        
        if diff_convlstm < 0.01:
            print(f"  → Real and zero images produce similar ConvLSTM features!")
            print(f"  → This is why input-level ablation shows no difference")
        
        # Now do CORRECT ablation: zero the features directly
        print("\n" + "-" * 70)
        print("Correct ablation: Zeroing ConvLSTM FEATURES")
        print("-" * 70)
        
        # Normal fusion
        fused_normal = model.cross_modal_fusion(transformer_feat, convlstm_feat)
        pred_normal = model.regression_head(fused_normal)
        output_normal = pred_normal.reshape(1, 24, 1)
        
        # Zero ConvLSTM features
        zero_convlstm_feat = torch.zeros_like(convlstm_feat)
        fused_zero = model.cross_modal_fusion(transformer_feat, zero_convlstm_feat)
        pred_zero = model.regression_head(fused_zero)
        output_zero = pred_zero.reshape(1, 24, 1)
        
        diff_feature = (output_normal - output_zero).abs().mean().item()
        
        print(f"\n  |Normal - Zero_ConvLSTM_features|: {diff_feature:.6e}")
        
        # Show per-target
        print(f"\n  Per-target differences:")
        print(f"  {'Target':<10} {'Normal':<12} {'Zero_feat':<12} {'Diff':<12}")
        print("  " + "-" * 50)
        
        for t in [0, 5, 11, 17, 23]:
            normal_val = output_normal[0, t, 0].item()
            zero_val = output_zero[0, t, 0].item()
            diff_val = abs(normal_val - zero_val)
            
            print(f"  {t:<10} {normal_val:<12.6f} {zero_val:<12.6f} {diff_val:<12.6e}")
    
    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    
    print(f"\nInput-level ablation (wrong):  {diff_input:.6e}")
    print(f"Feature-level ablation (right): {diff_feature:.6e}")
    
    print(f"\nWHY THE DIFFERENCE?")
    print(f"  ConvLSTM has bias → produces non-zero output even from zero input")
    print(f"  |ConvLSTM(real) - ConvLSTM(zero)|: {diff_convlstm:.6e}")
    
    if diff_convlstm < 0.01:
        print(f"\n❌ REAL PROBLEM FOUND!")
        print(f"  → Your IMAGES are producing nearly the same ConvLSTM features as ZERO images!")
        print(f"  → This means ConvLSTM is NOT extracting meaningful information from images")
        print(f"\n  Possible causes:")
        print(f"     1. Images are poorly normalized (too close to zero)")
        print(f"     2. ConvLSTM is undertrained on image content")
        print(f"     3. ConvLSTM relies mostly on bias, not image patterns")
        
        print(f"\n  SOLUTION:")
        print(f"     Check image preprocessing:")
        print(f"       - Are images normalized correctly?")
        print(f"       - Image mean: {images.mean().item():.6f}")
        print(f"       - Image std: {images.std().item():.6f}")
        print(f"       - Are these values reasonable for SDO images?")
    
    else:
        print(f"\n✓ ConvLSTM extracts meaningful features from images")
        print(f"  Feature-level ablation shows {diff_feature:.6e} difference")
        print(f"  → Images DO contribute to predictions")
        print(f"  → Previous test was misleading due to bias")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()