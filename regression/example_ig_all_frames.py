"""
전체 프레임 IG 생성 - All Target Indices Version

기능:
- 모든 input frame × 모든 output target 조합에 대해 IG 계산
- 예: input 24 frames, output 24 targets → 24×24 = 576 combinations
- NPZ 파일로만 저장 (플롯 생성 없음)

NPZ 구조:
- data: (3, seq_len, 64, 64) - 원본 이미지 (target마다 동일)
- ig: (n_targets, 3, seq_len, 64, 64) - 각 target에 대한 IG
- temporal_importance: (n_targets, seq_len) - 각 target별 시간 중요도
- channel_importance: (n_targets, 3) - 각 target별 채널 중요도

Updated: 2025-01-12
Image size: 64x64
Expected file size: ~24 MB per batch (24 targets)
"""

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from networks import create_model
from pipeline import create_dataloader
from saliency_maps import SaliencyExtractor
from utils import setup_device


def load_trained_model(checkpoint_path: str, config: DictConfig, device: str = 'cuda'):
    """훈련된 모델 로드"""
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
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


def generate_ig_all_targets(
    extractor: SaliencyExtractor,
    solar_wind_input: torch.Tensor,
    image_input: torch.Tensor,
    output_path: Path,
    channel_names: list,
    target_variable: int = 0,
    n_steps: int = 30
):
    """
    모든 target_index에 대해 IG 계산 및 NPZ 저장
    
    Args:
        extractor: SaliencyExtractor instance
        solar_wind_input: (batch, seq_len, num_vars)
        image_input: (batch, channels, seq_len, H, W)
        output_path: 저장할 NPZ 파일 경로
        channel_names: 채널 이름 리스트
        target_variable: 예측 변수 (0=Kp, 1=Dst, etc.)
        n_steps: IG interpolation steps
        
    Saves:
        NPZ file with keys:
            - data: (channels, seq_len, H, W) - 원본 이미지
            - ig: (n_targets, channels, seq_len, H, W) - 모든 target의 IG
            - temporal_importance: (n_targets, seq_len) - target별 시간 중요도
            - channel_importance: (n_targets, channels) - target별 채널 중요도
            - metadata: dict with parameters
    """
    
    print("\n" + "=" * 70)
    print("IG GENERATION - ALL TARGETS MODE")
    print("=" * 70)
    
    # Shape 정보
    batch, channels, seq_len, H, W = image_input.shape
    
    # Output shape 확인 (모델 실행해서 확인)
    with torch.no_grad():
        test_output = extractor.model(
            solar_wind_input.to(extractor.device),
            image_input.to(extractor.device)
        )
    n_targets = test_output.shape[1]  # (batch, n_targets, n_variables)
    
    print(f"\nInput shape: {image_input.shape}")
    print(f"  Channels: {channels}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Image size: {H}×{W}")
    print(f"\nOutput shape: {test_output.shape}")
    print(f"  Number of targets: {n_targets}")
    
    # 예상 파일 크기
    size_data = (channels * seq_len * H * W * 4) / (1024**2)  # float32
    size_ig = (n_targets * channels * seq_len * H * W * 4) / (1024**2)
    total_size = size_data + size_ig
    
    print(f"\nExpected size:")
    print(f"  data: {size_data:.1f} MB")
    print(f"  ig: {size_ig:.1f} MB")
    print(f"  Total uncompressed: {total_size:.1f} MB")
    print(f"  Estimated compressed: {total_size * 0.4:.1f} MB")
    
    # ================================================================
    # 1. 원본 데이터 준비
    # ================================================================
    print(f"\n[1/3] Preparing original data...")
    data = image_input[0].detach().cpu().numpy()  # (channels, seq_len, H, W)
    print(f"✓ data.shape = {data.shape}")
    
    # ================================================================
    # 2. 모든 target에 대해 IG 계산 (Batch mode)
    # ================================================================
    print(f"\n[2/3] Computing IG for all targets using BATCH mode (n_steps={n_steps})...")
    print("This will be much faster than sequential computation!")
    
    # 한 번에 모든 target 계산
    all_ig = extractor.integrated_gradients_batch_targets(
        solar_wind_input,
        image_input,
        target_variable=target_variable,
        n_steps=n_steps
    )
    
    # all_ig shape: (n_targets, channels, seq_len, H, W)
    n_targets = all_ig.shape[0]
    
    print(f"✓ IG computed for all {n_targets} targets!")
    print(f"  all_ig.shape = {all_ig.shape}")
    
    # 통계 계산
    temporal_importance_all = np.abs(all_ig).sum(axis=(1, 3, 4))  # (n_targets, seq_len)
    channel_importance_all = np.abs(all_ig).sum(axis=(2, 3, 4))   # (n_targets, channels)
    
    # 전체 IG 통계
    print(f"\nOverall IG Statistics:")
    print(f"  Mean |attribution|: {np.abs(all_ig).mean():.6f}")
    print(f"  Max |attribution|:  {np.abs(all_ig).max():.6f}")
    print(f"  Min attribution:    {all_ig.min():.6f}")
    print(f"  Max attribution:    {all_ig.max():.6f}")
    
    # ================================================================
    # 3. NPZ 저장
    # ================================================================
    print(f"\n[3/3] Saving to NPZ...")
    
    # 메타데이터
    metadata = {
        'target_variable': target_variable,
        'n_steps': n_steps,
        'seq_len': seq_len,
        'n_targets': n_targets,
        'n_channels': channels,
        'image_height': H,
        'image_width': W
    }
    
    # 저장
    np.savez_compressed(
        output_path,
        data=data,                                      # (3, 28, 64, 64)
        ig=all_ig,                                      # (24, 3, 28, 64, 64)
        temporal_importance=temporal_importance_all,    # (24, 28)
        channel_importance=channel_importance_all,      # (24, 3)
        channel_names=np.array(channel_names),
        metadata=metadata
    )
    
    # 실제 파일 크기
    file_size_mb = output_path.stat().st_size / (1024**2)
    
    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # ================================================================
    # 4. 요약
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ IG computation complete for {n_targets} targets!")
    print(f"✓ Data saved to: {output_path.absolute()}")
    
    print(f"\nNPZ Contents:")
    print(f"  'data':                (3, {seq_len}, {H}, {W}) - Original images")
    print(f"  'ig':                  ({n_targets}, 3, {seq_len}, {H}, {W}) - IG for all targets")
    print(f"  'temporal_importance': ({n_targets}, {seq_len}) - Time importance per target")
    print(f"  'channel_importance':  ({n_targets}, 3) - Channel importance per target")
    print(f"  'channel_names':       (3,) - Channel names")
    print(f"  'metadata':            dict - Parameters")
    
    # Target별 평균 중요도
    print(f"\nTarget-wise Summary:")
    for target_idx in range(min(5, n_targets)):  # 처음 5개만 표시
        total_imp = temporal_importance_all[target_idx].sum()
        most_imp_frame = np.argmax(temporal_importance_all[target_idx])
        print(f"  Target {target_idx}: Total={total_imp:.3f}, Peak frame={most_imp_frame}")
    if n_targets > 5:
        print(f"  ... ({n_targets - 5} more targets)")
    
    # 채널별 평균 중요도 (모든 target 평균)
    avg_channel_imp = channel_importance_all.mean(axis=0)
    avg_channel_imp = avg_channel_imp / avg_channel_imp.max()
    
    print(f"\nAverage Channel Importance (across all targets):")
    for i, name in enumerate(channel_names):
        print(f"  {name:15s}: {avg_channel_imp[i]:.3f}")
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print(f"""
# Load the data
import numpy as np

data = np.load('{output_path.name}')

# Access arrays
original_images = data['data']              # (3, {seq_len}, {H}, {W})
ig_all_targets = data['ig']                 # ({n_targets}, 3, {seq_len}, {H}, {W})
temporal_imp = data['temporal_importance']  # ({n_targets}, {seq_len})
channel_imp = data['channel_importance']    # ({n_targets}, 3)

# Example 1: Get IG for target_index=5
ig_target_5 = ig_all_targets[5]  # (3, {seq_len}, {H}, {W})

# Example 2: Get IG for target_index=10, channel=211Å, frame=15
ig_t10_ch1_fr15 = ig_all_targets[10, 1, 15, :, :]  # (64, 64)

# Example 3: Calculate temporal change for each target
for target_idx in range({n_targets}):
    ig_target = ig_all_targets[target_idx]  # (3, {seq_len}, {H}, {W})
    
    changes = []
    for t in range(1, {seq_len}):
        diff = np.abs(ig_target[:, t, :, :] - ig_target[:, t-1, :, :]).mean()
        changes.append(diff)
    
    avg_change = np.mean(changes)
    print(f"Target {{target_idx}}: avg_change = {{avg_change:.6f}}")

# Example 4: Analyze which target uses temporal info most
temporal_changes_per_target = []
for target_idx in range({n_targets}):
    ig_target = ig_all_targets[target_idx]
    changes = []
    for t in range(1, {seq_len}):
        diff = np.abs(ig_target[:, t, :, :] - ig_target[:, t-1, :, :]).mean()
        changes.append(diff)
    temporal_changes_per_target.append(np.mean(changes))

most_temporal_target = np.argmax(temporal_changes_per_target)
print(f"Target {{most_temporal_target}} uses temporal info most")
    """)
    
    return {
        'output_path': output_path,
        'file_size_mb': file_size_mb,
        'n_targets': n_targets,
        'data_shape': data.shape,
        'ig_shape': all_ig.shape,
        'temporal_importance': temporal_importance_all,
        'channel_importance': channel_importance_all
    }


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """메인 실행 함수"""

    # ========================================
    # 설정
    # ========================================
    checkpoint_path = config.saliency.checkpoint_path

    # Device 설정
    device = setup_device(config["environment"]["device"])

    # 출력 디렉토리
    output_dir = Path(config.saliency.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Model type
    model_type = config.model.model_type

    print("=" * 70)
    print("IG GENERATION - ALL TARGETS MODE")
    print("=" * 70)
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image size: 64×64")
    print()
    print("⚠️  NOTE: This will compute IG for ALL target indices")
    print("   For 24 targets × 50 steps, expect ~30-60 minutes per batch (CPU)")
    print()
    
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
    # Saliency Extractor 초기화
    # ========================================
    extractor = SaliencyExtractor(model, device=device)
    print("✓ SaliencyExtractor initialized\n")
    
    # ========================================
    # 설정
    # ========================================
    MAX_BATCHES = len(dataloader)
    # MAX_BATCHES = 3  # 처리할 배치 수

    # 분석 파라미터 (from config)
    target_variable = config.saliency.target_variable
    channel_names = ['193Å', '211Å', 'magnetogram']

    # IG 파라미터 (from config)
    N_STEPS = config.saliency.n_steps
    
    # ========================================
    # 배치 처리
    # ========================================
    results = []
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= MAX_BATCHES:
            print(f"\n✓ Processed {MAX_BATCHES} batches. Stopping.")
            break
        
        print("\n" + "=" * 70)
        print(f"PROCESSING BATCH {batch_idx + 1}/{MAX_BATCHES}")
        print("=" * 70)
        
        # 데이터 준비
        solar_wind_input = batch["inputs"][:1]  # 첫 샘플만
        image_input = batch["sdo"][:1]
        
        # 출력 파일명
        output_path = output_dir / f"ig_all_targets_batch_{batch_idx:04d}.npz"
        
        try:
            result = generate_ig_all_targets(
                extractor=extractor,
                solar_wind_input=solar_wind_input,
                image_input=image_input,
                output_path=output_path,
                channel_names=channel_names,
                target_variable=target_variable,
                n_steps=N_STEPS
            )
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # 최종 요약
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Processed {len(results)} batches successfully")
    print(f"✓ Output directory: {output_dir.absolute()}")
    
    if results:
        total_size = sum(r['file_size_mb'] for r in results)
        n_targets = results[0]['n_targets']
        
        print(f"\nGenerated files:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['output_path'].name} ({result['file_size_mb']:.2f} MB)")
        print(f"\nTotal size: {total_size:.2f} MB")
        
        print(f"\nData shapes:")
        print(f"  data: {results[0]['data_shape']}")
        print(f"  ig:   {results[0]['ig_shape']}")
        print(f"\nNumber of targets: {n_targets}")
        print(f"Total IG combinations: {n_targets} targets × 28 frames = {n_targets * 28}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Load and analyze NPZ files:
   data = np.load('ig_all_targets_outputs/ig_all_targets_batch_0000.npz')
   ig = data['ig']  # (n_targets, 3, 28, 64, 64)
   
2. Analyze target-specific patterns:
   - Which target uses temporal info most?
   - Which target focuses on which channels?
   - How does temporal importance vary across targets?
   
3. Compare early vs late predictions:
   - Early targets (0-7): Short-term prediction
   - Late targets (16-23): Long-term prediction
   - Which uses more temporal information?
   
4. Create target-comparison plots:
   - Temporal importance heatmap (targets × frames)
   - Channel importance evolution across targets
    """)


if __name__ == '__main__':
    main()