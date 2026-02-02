"""
실제 사용 예제: Saliency Maps 추출 (수정 버전)

문제점 수정:
1. 변수명 충돌 해결
2. 디버깅 출력 추가
3. Gradient 계산 검증
4. MPS 호환성 개선
"""

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from networks import create_model
from pipeline import create_dataloader
from saliency_maps import SaliencyExtractor


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


def debug_gradients(model, solar_wind_input, image_input, device):
    """Gradient 계산이 제대로 되는지 확인"""
    print("\n" + "=" * 60)
    print("DEBUG: Gradient Check")
    print("=" * 60)
    
    solar_wind_input = solar_wind_input.to(device)
    image_input = image_input.to(device)
    
    # 이미지에 gradient 활성화
    image_input.requires_grad = True
    
    # Forward
    model.zero_grad()
    output = model(solar_wind_input, image_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output[0, 0, 0].item():.4f}")
    
    # Backward
    target = output[0, 0, 0]
    target.backward()
    
    # Gradient 확인
    if image_input.grad is not None:
        grad_magnitude = image_input.grad.abs().mean().item()
        grad_max = image_input.grad.abs().max().item()
        grad_min = image_input.grad.abs().min().item()
        
        print(f"\n✓ Gradient computed successfully!")
        print(f"  Gradient mean: {grad_magnitude:.6f}")
        print(f"  Gradient max: {grad_max:.6f}")
        print(f"  Gradient min: {grad_min:.6f}")
        
        if grad_magnitude < 1e-10:
            print("\n⚠️  WARNING: Gradient is extremely small!")
            print("   → Model may not be using image input effectively")
            return False
        
        return True
    else:
        print("\n❌ ERROR: No gradient computed!")
        print("   → Check if model is in eval mode or if requires_grad=True")
        return False


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """메인 실행 함수"""
    
    # ========================================
    # 설정
    # ========================================
    checkpoint_path = "/Users/eunsupark/checkpoints/SINGLE_7_1_05/checkpoint/model_epoch0100.pth"
    
    # MPS 사용 시 주의
    device = "cpu"
    if device == "mps":
        print("⚠️  Using MPS (Apple Silicon)")
        print("   Some gradient operations may not work correctly.")
        print("   Consider using 'cpu' if results are abnormal.\n")
    
    output_root = Path("saliency_outputs")
    output_root.mkdir(exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output root directory: {output_root}\n")
    
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
    # 첫 번째 배치로 디버깅
    # ========================================
    batch = next(iter(dataloader))
    solar_wind_input = batch["inputs"][:1]  # 1개만
    image_input = batch["sdo"][:1]
    
    print(f"Data shapes:")
    print(f"  Solar wind: {solar_wind_input.shape}")
    print(f"  SDO images: {image_input.shape}")
    
    # Gradient 테스트
    gradient_ok = debug_gradients(model, solar_wind_input, image_input, device)
    
    if not gradient_ok:
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print("Gradient computation failed or too small.")
        print("\nPossible solutions:")
        print("1. Try device='cpu' instead of 'mps'")
        print("2. Check if model architecture has gradient flow issues")
        print("3. Verify model is actually using SDO images in forward pass")
        print("4. Try a different sample (this one might have low activity)")
        print("\nContinuing anyway, but results may not be meaningful...")
    
    # ========================================
    # Saliency Extractor 초기화
    # ========================================
    extractor = SaliencyExtractor(model, device=device)
    print("\n✓ SaliencyExtractor initialized")
    
    # ========================================
    # 처음 3개 배치만 처리
    # ========================================
    MAX_BATCHES = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= MAX_BATCHES:
            print(f"\n✓ Processed {MAX_BATCHES} batches. Stopping.")
            break
        
        print("\n" + "=" * 60)
        print(f"Processing batch {batch_idx + 1}/{MAX_BATCHES}")
        print("=" * 60)
        
        # 데이터 준비
        solar_wind_input = batch["inputs"][:1]  # 첫 샘플만
        image_input = batch["sdo"][:1]
        
        output_dir = output_root / f"batch_{batch_idx:04d}"
        output_dir.mkdir(exist_ok=True)
        
        # 분석 타겟
        target_index = 0
        target_variable = 0
        
        # 채널 이름 정의
        channel_names = ['193Å', '211Å', 'magnetogram']  # SDO AIA 파장
        
        # ========================================
        # 옵션 1: 개별 채널 분석 (기존 방식)
        # ========================================
        USE_INDIVIDUAL_ANALYSIS = False  # True로 바꾸면 개별 분석 실행
        
        if USE_INDIVIDUAL_ANALYSIS:
            channel_idx = 0  # 첫 번째 채널만
            
            print("\n[Individual Channel Analysis]")
            print(f"Analyzing channel: {channel_names[channel_idx]}")
            print("-" * 60)
            
            try:
                grad_cam_maps = extractor.grad_cam(
                    solar_wind_input, image_input,
                    target_index=target_index,
                    target_variable=target_variable
                )
                
                print(f"✓ Shape: {grad_cam_maps.shape}")
                print(f"  Min: {grad_cam_maps.min():.6f}")
                print(f"  Max: {grad_cam_maps.max():.6f}")
                print(f"  Mean: {grad_cam_maps.mean():.6f}")
                
                extractor.visualize_grad_cam(
                    grad_cam_maps, image_input,
                    channel_idx=channel_idx,
                    save_path=output_dir / f"grad_cam_{channel_names[channel_idx]}.png"
                )
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # ========================================
        # 옵션 2: 모든 채널 자동 분석 (권장!)
        # ========================================
        print("\n" + "=" * 60)
        print("MULTI-CHANNEL ANALYSIS")
        print("=" * 60)
        
        try:
            extractor.visualize_all_channels_analysis(
                solar_wind_input,
                image_input,
                target_index=target_index,
                target_variable=target_variable,
                channel_names=channel_names,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"❌ Error in multi-channel analysis: {e}")
            import traceback
            traceback.print_exc()
        
        # ========================================
        # 추가: Temporal Importance (이미 생성됨)
        # ========================================
        print("\n" + "-" * 60)
        print("Additional Analysis")
        print("-" * 60)
        
        # Integrated Gradients (전체 채널)
        print("\nComputing Integrated Gradients for channel importance...")
        try:
            ig_maps = extractor.integrated_gradients(
                solar_wind_input,
                image_input,
                target_index=target_index,
                target_variable=target_variable,
                n_steps=30
            )
            
            # Channel 중요도 계산
            channel_importance = np.abs(ig_maps).sum(axis=(1, 2, 3))
            channel_importance = channel_importance / (channel_importance.max() + 1e-10)
            
            print("  Channel importance:")
            for ch_idx, name in enumerate(channel_names):
                print(f"    {name}: {channel_importance[ch_idx]:.3f}")
            
            # 저장
            np.savez(
                output_dir / "channel_importance.npz",
                channel_importance=channel_importance,
                channel_names=channel_names
            )
            print("  ✓ Saved: channel_importance.npz")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # ========================================
        # 저장
        # ========================================
        print("\n" + "-" * 60)
        print(f"✓ Batch {batch_idx} complete: {output_dir}")
        print("\nGenerated structure:")
        print(f"  {output_dir}/")
        for ch_idx, ch_name in enumerate(channel_names):
            ch_dir_name = f"channel_{ch_idx}_{ch_name.replace('Å', 'A')}"
            print(f"    ├─ {ch_dir_name}/")
            print(f"    │   ├─ grad_cam_{ch_name}.png")
            print(f"    │   ├─ full_sequence_{ch_name}.png")
            print(f"    │   └─ comprehensive_{ch_name}.png")
        print(f"    ├─ channel_comparison.png")
        print(f"    ├─ temporal_importance_all_channels.png")
        print(f"    └─ channel_importance.npz")
    
    # ========================================
    # 최종 요약
    # ========================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n✓ Processed {min(MAX_BATCHES, len(dataloader))} batches")
    print(f"✓ Results saved to: {output_root}")
    
    print("\nGenerated structure per batch:")
    print("  batch_XXXX/")
    print("    ├─ channel_0_193A/  (193Å wavelength)")
    print("    │   ├─ grad_cam_193Å.png")
    print("    │   ├─ full_sequence_193Å.png")
    print("    │   └─ comprehensive_193Å.png")
    print("    ├─ channel_1_211A/  (211Å wavelength)")
    print("    │   ├─ grad_cam_211Å.png")
    print("    │   ├─ full_sequence_211Å.png")
    print("    │   └─ comprehensive_211Å.png")
    print("    ├─ channel_2_304A/  (magnetogram wavelength)")
    print("    │   ├─ grad_cam_magnetogram.png")
    print("    │   ├─ full_sequence_magnetogram.png")
    print("    │   └─ comprehensive_magnetogram.png")
    print("    ├─ channel_comparison.png  ← ALL CHANNELS")
    print("    ├─ temporal_importance_all_channels.png")
    print("    └─ channel_importance.npz")
    
    print("\n" + "=" * 60)
    print("KEY FEATURES")
    print("=" * 60)
    print("✓ Multi-channel: Each wavelength analyzed separately")
    print("✓ Full sequence: All time steps in one plot")
    print("✓ Comparison: Side-by-side channel comparison")
    print("✓ Scientific: Which wavelength/time is most important")


if __name__ == '__main__':
    main()