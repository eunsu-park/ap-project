"""
Quick Ablation Test - 모델이 이미지를 사용하는지 빠르게 확인

IG 계산 없이 Ablation Test만 수행
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
    print("QUICK ABLATION TEST")
    print("=" * 70)
    print("Testing which input (solar wind vs images) affects predictions...\n")
    
    # ========================================
    # 설정
    # ========================================
    checkpoint_path = config.validation.checkpoint_path
    device = "mps"  # or "cpu" or "cuda"
    
    # ========================================
    # 모델 로드
    # ========================================
    model = load_trained_model(checkpoint_path, config, device)
    
    # ========================================
    # DataLoader
    # ========================================
    dataloader = create_dataloader(config, phase="validation")
    print("✓ DataLoader loaded\n")
    
    # ========================================
    # Ablation Test
    # ========================================
    print("=" * 70)
    print("ABLATION TEST")
    print("=" * 70)
    
    # Get one batch
    batch = next(iter(dataloader))
    solar_wind = batch["inputs"][:1].to(device)
    images = batch["sdo"][:1].to(device)
    
    print(f"\nInput shapes:")
    print(f"  Solar wind: {solar_wind.shape}")
    print(f"  Images: {images.shape}")
    
    with torch.no_grad():
        # 1. Normal (both inputs)
        print("\n1. Computing with both inputs...")
        out_normal = model(solar_wind, images)
        
        # 2. Solar wind only (images = 0)
        print("2. Computing with solar wind only (images=0)...")
        out_sw_only = model(solar_wind, torch.zeros_like(images))
        
        # 3. Images only (solar wind = 0)
        print("3. Computing with images only (solar_wind=0)...")
        out_img_only = model(torch.zeros_like(solar_wind), images)
    
    # Show results for first few targets
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nPredictions for different input combinations:")
    print(f"{'Target':<10} {'Normal':<12} {'SW only':<12} {'Img only':<12} {'Diff(SW)':<12}")
    print("-" * 68)
    
    for target_idx in [0, 5, 11, 17, 23]:
        normal_val = out_normal[0, target_idx, 0].item()
        sw_val = out_sw_only[0, target_idx, 0].item()
        img_val = out_img_only[0, target_idx, 0].item()
        diff = abs(normal_val - sw_val)
        
        print(f"{target_idx:<10} {normal_val:<12.6f} {sw_val:<12.6f} {img_val:<12.6f} {diff:<12.6e}")
    
    # Calculate differences
    diff_sw = (out_normal - out_sw_only).abs().mean().item()
    diff_img = (out_normal - out_img_only).abs().mean().item()
    
    print(f"\n" + "-" * 70)
    print("AVERAGE DIFFERENCES")
    print("-" * 70)
    print(f"  |Normal - SW_only|:  {diff_sw:.6e}")
    print(f"  |Normal - Img_only|: {diff_img:.6e}")
    
    # Per-target analysis
    print(f"\n" + "-" * 70)
    print("PER-TARGET ANALYSIS")
    print("-" * 70)
    
    max_diff_sw = 0
    max_diff_target = 0
    
    for target_idx in range(out_normal.shape[1]):
        diff = (out_normal[0, target_idx, 0] - out_sw_only[0, target_idx, 0]).abs().item()
        if diff > max_diff_sw:
            max_diff_sw = diff
            max_diff_target = target_idx
    
    print(f"  Maximum difference at target {max_diff_target}: {max_diff_sw:.6e}")
    print(f"  Minimum would be: 0 (if all identical)")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if diff_sw < 1e-6:
        print("\n❌ CRITICAL: Output ≈ SW only!")
        print(f"  Difference: {diff_sw:.6e} (< 1e-6)")
        print("\n  → Model COMPLETELY ignores images")
        print("  → ConvLSTM output is NOT reaching final layer")
        print("\n  EVIDENCE:")
        print(f"     - All targets produce nearly identical output")
        print(f"     - Removing images changes output by only {diff_sw:.6e}")
        print(f"     - This is ~{(diff_sw/abs(out_normal.abs().mean().item()))*100:.6f}% of typical prediction")
        print("\n  ROOT CAUSE:")
        print("     Forward pass likely has structure like:")
        print("       def forward(sw, img):")
        print("         img_feat = convlstm(img)  # computed but not used")
        print("         sw_feat = transformer(sw)")
        print("         output = final_fc(sw_feat)  # img_feat ignored!")
        print("\n  SOLUTION:")
        print("     1. Examine model.forward() in networks.py")
        print("     2. Verify ConvLSTM features reach final_fc")
        print("     3. Fix architecture to use fused features")
        
    elif diff_sw < 1e-3:
        print("\n⚠️  WARNING: Output ≈ SW only with tiny image contribution")
        print(f"  Difference: {diff_sw:.6e} (< 1e-3)")
        
        # Calculate contribution percentage
        normal_magnitude = out_normal.abs().mean().item()
        img_contribution = (diff_sw / normal_magnitude) * 100 if normal_magnitude > 0 else 0
        
        print(f"\n  → Images contribute only ~{img_contribution:.3f}% to predictions")
        print(f"  → Model HEAVILY relies on solar wind ({100-img_contribution:.3f}%)")
        print("\n  POSSIBLE CAUSES:")
        print("     1. Cross-modal fusion heavily favors solar wind")
        print("     2. ConvLSTM features are very weak compared to Transformer")
        print("     3. Model learned during training that images are less predictive")
        print("\n  THIS EXPLAINS WHY:")
        print(f"     - Gradient w.r.t images is {1.9e-12:.2e} (extremely small)")
        print("     - IG produces all zeros (gradient too small to accumulate)")
        print("\n  OPTIONS:")
        print("     A. Accept this behavior:")
        print("        - If solar wind is truly more predictive, this is OK")
        print("        - Model is doing what data supports")
        print("     B. Retrain with constraints:")
        print("        - Add image importance loss")
        print("        - Ensure ConvLSTM contributes meaningfully")
        print("     C. For IG analysis:")
        print("        - Use amplified gradients (×1e6)")
        print("        - Or use different baseline (mean instead of zeros)")
        
    else:
        print("\n✓ Both inputs matter!")
        print(f"  → Solar wind change affects output by {diff_sw:.6e}")
        print(f"  → Image change affects output by {diff_img:.6e}")
        
        if diff_img < diff_sw * 0.01:
            print("\n  Note: Images matter, but much less than solar wind")
            print(f"  Image contribution: ~{(diff_img/(diff_sw+diff_img))*100:.1f}%")
            print(f"  Solar wind contribution: ~{(diff_sw/(diff_sw+diff_img))*100:.1f}%")
            print("\n  This is why IG gradients are small but not zero")
        else:
            print("\n  Both inputs have comparable contributions!")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if diff_sw < 1e-6:
        print("\n🔴 Model does NOT use images - Fix architecture required")
        print("   Action: Check forward() and fix ConvLSTM connection")
    elif diff_sw < 1e-3:
        print("\n🟡 Model uses images weakly (~0.1% contribution)")
        print("   Action: Retrain or accept this behavior")
    else:
        print("\n🟢 Model uses both inputs appropriately")
        print("   Action: IG should work (may need baseline adjustment)")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()