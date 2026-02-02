"""
모델이 SDO 이미지를 실제로 사용하는지 검증하는 스크립트

이 스크립트는:
1. 정상 입력으로 예측
2. SDO 이미지를 0으로 만들어서 예측
3. 두 예측 간 차이 확인

차이가 거의 없다면 → 모델이 이미지를 사용하지 않음
"""

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

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
    
    return model


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """메인 실행 함수"""
    
    checkpoint_path = "/opt/projects/10_Harim/01_AP/04_Result/model_epoch0100.pth"
    device = "cpu"  # CPU로 확실하게
    
    print("=" * 70)
    print("MODEL IMAGE USAGE VERIFICATION")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Device: {device}\n")
    
    # 모델 로드
    model = load_trained_model(checkpoint_path, config, device)
    print("✓ Model loaded\n")
    
    # 데이터 로드
    dataloader = create_dataloader(config, phase="validation")
    
    # 여러 샘플 테스트
    NUM_SAMPLES = 10
    
    results = []
    
    for idx, batch in enumerate(dataloader):
        if idx >= NUM_SAMPLES:
            break
        
        solar_wind = batch["inputs"][:1].to(device)
        images = batch["sdo"][:1].to(device)
        
        with torch.no_grad():
            # 1. 정상 예측
            pred_normal = model(solar_wind, images)
            
            # 2. 이미지를 0으로 만들어서 예측
            images_zero = torch.zeros_like(images)
            pred_no_image = model(solar_wind, images_zero)
            
            # 3. 이미지를 랜덤 노이즈로 바꿔서 예측
            images_noise = torch.randn_like(images)
            pred_noise = model(solar_wind, images_noise)
        
        # 차이 계산
        diff_zero = torch.abs(pred_normal - pred_no_image).mean().item()
        diff_noise = torch.abs(pred_normal - pred_noise).mean().item()
        
        results.append({
            'sample': idx,
            'pred_normal': pred_normal[0, 0, 0].item(),
            'pred_no_image': pred_no_image[0, 0, 0].item(),
            'pred_noise': pred_noise[0, 0, 0].item(),
            'diff_zero': diff_zero,
            'diff_noise': diff_noise
        })
        
        print(f"Sample {idx:2d}: "
              f"Normal={pred_normal[0,0,0].item():6.3f}, "
              f"NoImage={pred_no_image[0,0,0].item():6.3f}, "
              f"Noise={pred_noise[0,0,0].item():6.3f}, "
              f"Diff_zero={diff_zero:.6f}, "
              f"Diff_noise={diff_noise:.6f}")
    
    # 통계
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    diffs_zero = [r['diff_zero'] for r in results]
    diffs_noise = [r['diff_noise'] for r in results]
    
    mean_diff_zero = np.mean(diffs_zero)
    max_diff_zero = np.max(diffs_zero)
    
    mean_diff_noise = np.mean(diffs_noise)
    max_diff_noise = np.max(diffs_noise)
    
    print(f"\nWhen replacing images with ZEROS:")
    print(f"  Mean prediction difference: {mean_diff_zero:.6f}")
    print(f"  Max prediction difference:  {max_diff_zero:.6f}")
    
    print(f"\nWhen replacing images with NOISE:")
    print(f"  Mean prediction difference: {mean_diff_noise:.6f}")
    print(f"  Max prediction difference:  {max_diff_noise:.6f}")
    
    # 진단
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    THRESHOLD = 0.01  # 1% 변화
    
    if mean_diff_zero < THRESHOLD:
        print("\n🔴 CRITICAL ISSUE DETECTED!")
        print(f"   Prediction changes by only {mean_diff_zero:.6f} when images are removed.")
        print("   → Model is NOT using SDO images effectively!")
        print("\n   Possible reasons:")
        print("   1. ConvLSTM features have very small magnitude")
        print("   2. Cross-modal fusion heavily favors transformer features")
        print("   3. Model learned to rely only on OMNI data")
        print("   4. ConvLSTM weights are near zero / not trained properly")
        
        print("\n   Recommended actions:")
        print("   a) Check training logs - did ConvLSTM loss decrease?")
        print("   b) Examine cross-modal fusion weights")
        print("   c) Try training with higher weight on image loss")
        print("   d) Validate that SDO images have signal (not all zeros)")
        
    elif mean_diff_zero < 0.1:
        print("\n⚠️  WARNING: Low image importance")
        print(f"   Prediction changes by {mean_diff_zero:.6f} when images are removed.")
        print("   → Model uses images, but they have minimal impact.")
        print("   → OMNI data is much more important than SDO images.")
        
    else:
        print("\n✅ Model is using images!")
        print(f"   Prediction changes by {mean_diff_zero:.6f} when images are removed.")
        print("   → Images contribute meaningfully to predictions.")
    
    # Saliency map 유의미성
    print("\n" + "=" * 70)
    print("SALIENCY MAP IMPLICATIONS")
    print("=" * 70)
    
    if mean_diff_zero < THRESHOLD:
        print("\n❌ Saliency maps will NOT be meaningful!")
        print("   Since the model doesn't use images, gradients will be:")
        print("   - Near zero")
        print("   - Uniform across all pixels")
        print("   - Not interpretable")
        print("\n   → Fix the model training first before analyzing saliency!")
        
    else:
        print("\n✅ Saliency maps should be interpretable.")
        print("   The model uses images, so gradients should show:")
        print("   - Spatial patterns")
        print("   - Temporal variations")
        print("   - Meaningful attributions")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
