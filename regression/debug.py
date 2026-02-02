"""
전체 프레임 IG 생성 - All Target Indices Version (Debug Mode)

기능:
- 모든 input frame × 모든 output target 조합에 대해 IG 계산
- 예: input 24 frames, output 24 targets → 24×24 = 576 combinations
- NPZ 파일로만 저장 (플롯 생성 없음)
- 🐛 Comprehensive debugging enabled

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
    모든 target_index에 대해 IG 계산 및 NPZ 저장 (Debug mode)
    
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
    print("IG GENERATION - ALL TARGETS MODE (DEBUG)")
    print("=" * 70)
    
    # Shape 정보
    batch, channels, seq_len, H, W = image_input.shape
    
    # ================================================================
    # DEBUG 1: Input Statistics
    # ================================================================
    print("\n" + "=" * 70)
    print("DEBUG 1: INPUT STATISTICS")
    print("=" * 70)
    
    print(f"\nImage input:")
    print(f"  Shape: {image_input.shape}")
    print(f"  Mean: {image_input.mean().item():.6f}")
    print(f"  Std: {image_input.std().item():.6f}")
    print(f"  Min: {image_input.min().item():.6f}")
    print(f"  Max: {image_input.max().item():.6f}")
    
    baseline_test = torch.zeros_like(image_input)
    diff = (image_input - baseline_test).abs().mean().item()
    print(f"\n|Input - Baseline(zeros)|: {diff:.6f}")
    
    if diff < 0.01:
        print("\n⚠️  WARNING: Input is too close to zero baseline!")
        print("  → Input may be already normalized to ~0")
        print("  → This will result in zero IG values")
        print("\n  RECOMMENDATION:")
        print("     - Use different baseline (e.g., mean image)")
        print("     - Check data preprocessing")
    
    print("\nSolar wind input:")
    print(f"  Shape: {solar_wind_input.shape}")
    print(f"  Mean: {solar_wind_input.mean().item():.6f}")
    print(f"  Std: {solar_wind_input.std().item():.6f}")
    
    # ================================================================
    # DEBUG 2: Model Output Check
    # ================================================================
    print("\n" + "=" * 70)
    print("DEBUG 2: MODEL OUTPUT")
    print("=" * 70)
    
    # Output shape 확인 (모델 실행해서 확인)
    with torch.no_grad():
        test_output = extractor.model(
            solar_wind_input.to(extractor.device),
            image_input.to(extractor.device)
        )
    n_targets = test_output.shape[1]  # (batch, n_targets, n_variables)
    
    print(f"\nOutput shape: {test_output.shape}")
    print(f"  Number of targets: {n_targets}")
    print(f"  Output sample values:")
    print(f"    Target 0: {test_output[0, 0, 0].item():.6f}")
    if n_targets > 1:
        print(f"    Target {n_targets//2}: {test_output[0, n_targets//2, 0].item():.6f}")
        print(f"    Target {n_targets-1}: {test_output[0, n_targets-1, 0].item():.6f}")
    
    # ================================================================
    # DEBUG 3: Gradient Check
    # ================================================================
    print("\n" + "=" * 70)
    print("DEBUG 3: GRADIENT CHECK")
    print("=" * 70)
    
    print("\nTesting if model uses image input...")
    
    # Test 1: Gradient w.r.t. images
    solar_wind_test = solar_wind_input.to(extractor.device)
    image_test = image_input.to(extractor.device)
    image_test.requires_grad = True
    
    extractor.model.zero_grad()
    output_test = extractor.model(solar_wind_test, image_test)
    target_test = output_test[0, 0, 0]
    
    print(f"  Target value: {target_test.item():.6f}")
    
    target_test.backward()
    
    if image_test.grad is not None:
        grad_mean = image_test.grad.abs().mean().item()
        grad_max = image_test.grad.abs().max().item()
        grad_min = image_test.grad.abs().min().item()
        grad_nonzero = (image_test.grad.abs() > 1e-10).sum().item()
        grad_total = image_test.grad.numel()
        
        print(f"\n✓ Gradient computed!")
        print(f"  Mean |grad|: {grad_mean:.6e}")
        print(f"  Max |grad|: {grad_max:.6e}")
        print(f"  Min |grad|: {grad_min:.6e}")
        print(f"  Non-zero gradients: {grad_nonzero}/{grad_total} ({100*grad_nonzero/grad_total:.1f}%)")
        
        if grad_mean < 1e-10:
            print("\n" + "!" * 70)
            print("❌ CRITICAL ERROR: Gradient is essentially ZERO!")
            print("!" * 70)
            print("\n  → Model is NOT using image input for predictions")
            print("\n  Possible causes:")
            print("     1. ConvLSTM weights are not trained")
            print("     2. Image features are not connected to output")
            print("     3. Model only uses solar wind input")
            print("     4. ConvLSTM output is ignored in final layer")
            print("\n  → IG will be all zeros (as you observed)")
            print("\n  RECOMMENDATIONS:")
            print("     A. Check model architecture:")
            print("        - Print model structure")
            print("        - Verify ConvLSTM is in computation graph")
            print("     B. Test image importance:")
            print("        - Compare predictions with/without images")
            print("     C. Check training:")
            print("        - Was ConvLSTM trained?")
            print("        - Are ConvLSTM weights frozen?")
            print("\n" + "!" * 70)
        elif grad_mean < 1e-6:
            print("\n⚠️  WARNING: Gradient is very small!")
            print("  → Model uses images weakly (~0.01% contribution)")
            print("  → IG values will be very small but non-zero")
            print("\n  This might be expected if:")
            print("     - Solar wind is much more important than images")
            print("     - Model is trained to rely on solar wind primarily")
        else:
            print("\n✓ Gradient looks normal!")
            print("  → Model is using image input properly")
            print("  → IG should work correctly")
    else:
        print("\n" + "!" * 70)
        print("❌ CRITICAL ERROR: No gradient computed!")
        print("!" * 70)
        print("  → Images are completely disconnected from output")
        print("  → Check model.forward() implementation")
        print("!" * 70)
    
    # Test 2: Compare with/without images
    print("\n" + "-" * 70)
    print("Testing model dependency on images...")
    
    with torch.no_grad():
        # Normal output
        output_normal = extractor.model(solar_wind_test, image_test)
        
        # Output with zero images
        image_zero = torch.zeros_like(image_test)
        output_zero_img = extractor.model(solar_wind_test, image_zero)
        
        # Output with random images
        image_random = torch.randn_like(image_test)
        output_random_img = extractor.model(solar_wind_test, image_random)
        
        diff_zero = (output_normal - output_zero_img).abs().mean().item()
        diff_random = (output_normal - output_random_img).abs().mean().item()
        
        print(f"\n  |Output(normal) - Output(zero_images)|: {diff_zero:.6e}")
        print(f"  |Output(normal) - Output(random_images)|: {diff_random:.6e}")
        
        if diff_zero < 1e-6 and diff_random < 1e-6:
            print("\n❌ CRITICAL: Output doesn't change with different images!")
            print("  → Model is NOT using images at all!")
            print("  → Stopping IG computation (would be all zeros)")
            print("\n  ACTION REQUIRED:")
            print("     1. Fix model architecture")
            print("     2. Retrain model to use images")
            print("     3. Verify ConvLSTM is connected properly")
            return None
        elif diff_zero < 1e-3 and diff_random < 1e-3:
            print("\n⚠️  WARNING: Output barely changes with different images")
            print("  → Model uses images very weakly")
            print("  → Proceeding, but expect small IG values")
        else:
            print("\n✓ Output changes significantly with images")
            print("  → Model is using images properly")
    
    # ================================================================
    # DEBUG 4: Model Architecture Check
    # ================================================================
    print("\n" + "=" * 70)
    print("DEBUG 4: MODEL ARCHITECTURE CHECK")
    print("=" * 70)
    
    # 1. ConvLSTM 파라미터 통계
    print("\n1. ConvLSTM Parameters:")
    has_convlstm = False
    convlstm_params = []
    
    for name, param in extractor.model.named_parameters():
        name_lower = name.lower()
        if 'convlstm' in name_lower or ('conv' in name_lower and 'lstm' in name_lower):
            has_convlstm = True
            param_stats = {
                'name': name,
                'shape': param.shape,
                'mean': param.mean().item(),
                'std': param.std().item(),
                'max_abs': param.abs().max().item(),
                'requires_grad': param.requires_grad
            }
            convlstm_params.append(param_stats)
            
            print(f"\n  {name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Mean: {param_stats['mean']:.6e}")
            print(f"    Std: {param_stats['std']:.6e}")
            print(f"    Max |param|: {param_stats['max_abs']:.6e}")
            print(f"    Requires grad: {param_stats['requires_grad']}")
    
    if not has_convlstm:
        print("\n  ⚠️  No ConvLSTM parameters found!")
        print("  → Model may not have ConvLSTM")
        print("  → Or parameter names don't contain 'convlstm'")
    else:
        # ConvLSTM 통계 요약
        avg_std = np.mean([p['std'] for p in convlstm_params])
        avg_max = np.mean([p['max_abs'] for p in convlstm_params])
        
        print(f"\n  Summary:")
        print(f"    Number of ConvLSTM params: {len(convlstm_params)}")
        print(f"    Average std: {avg_std:.6e}")
        print(f"    Average max: {avg_max:.6e}")
        
        if avg_std < 1e-3:
            print("\n  ❌ ConvLSTM parameters are very small!")
            print("  → Likely not trained properly")
            print("  → Weights close to initialization")
        elif avg_std < 0.1:
            print("\n  ⚠️  ConvLSTM parameters are small")
            print("  → May be undertrained")
        else:
            print("\n  ✓ ConvLSTM parameters look trained")
    
    # 2. Forward pass activation check
    print("\n2. Forward Pass Activations:")
    
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            if isinstance(act, torch.Tensor):
                activations[name] = {
                    'mean': act.abs().mean().item(),
                    'std': act.std().item(),
                    'max': act.abs().max().item()
                }
        return hook
    
    # Register hooks for ConvLSTM layers
    handles = []
    for name, module in extractor.model.named_modules():
        name_lower = name.lower()
        if 'convlstm' in name_lower or ('conv' in name_lower and 'lstm' in name_lower):
            handle = module.register_forward_hook(hook_fn(name))
            handles.append(handle)
    
    # Forward pass
    with torch.no_grad():
        _ = extractor.model(solar_wind_test, image_test)
    
    if activations:
        print("\n  Activations found:")
        for name, act in activations.items():
            print(f"\n  {name}:")
            print(f"    Mean: {act['mean']:.6e}")
            print(f"    Std: {act['std']:.6e}")
            print(f"    Max: {act['max']:.6e}")
        
        # 활성화 통계
        avg_act = np.mean([act['mean'] for act in activations.values()])
        print(f"\n  Summary:")
        print(f"    Average activation: {avg_act:.6e}")
        
        if avg_act < 1e-6:
            print("\n  ❌ Activations are essentially zero!")
            print("  → ConvLSTM is not producing meaningful output")
            print("  → Check if inputs are being passed correctly")
        elif avg_act < 1e-3:
            print("\n  ⚠️  Activations are very small")
            print("  → ConvLSTM output is weak")
        else:
            print("\n  ✓ Activations look normal")
    else:
        print("\n  ⚠️  No ConvLSTM activations captured")
        print("  → Module names may not match pattern")
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # 3. 최종 진단
    print("\n" + "-" * 70)
    print("DIAGNOSIS:")
    print("-" * 70)
    
    if not has_convlstm:
        print("\n❌ CRITICAL: No ConvLSTM found in model")
        print("  → Model architecture may not include ConvLSTM")
        print("  → Cannot use IG for image analysis")
    elif grad_mean < 1e-10:
        if avg_std < 1e-3:
            print("\n❌ ROOT CAUSE: ConvLSTM not trained")
            print("  → Parameters are near initialization")
            print("  → Need to retrain model")
        elif not activations or avg_act < 1e-6:
            print("\n❌ ROOT CAUSE: ConvLSTM not activated")
            print("  → Parameters exist but not producing output")
            print("  → Check forward() implementation")
        else:
            print("\n❌ ROOT CAUSE: ConvLSTM disconnected from output")
            print("  → ConvLSTM runs but doesn't affect predictions")
            print("  → Check final layers in forward()")
    else:
        print("\n✓ Model structure looks OK")
        print("  → But gradient is still very small")
        print("  → Solar wind may be much more important than images")
    
    print("\n" + "=" * 70)
    
    print(f"\nInput shape: {image_input.shape}")
    print(f"  Channels: {channels}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Image size: {H}×{W}")
    
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
    print(f"  Mean |attribution|: {np.abs(all_ig).mean():.6e}")
    print(f"  Max |attribution|:  {np.abs(all_ig).max():.6e}")
    print(f"  Min attribution:    {all_ig.min():.6e}")
    print(f"  Max attribution:    {all_ig.max():.6e}")
    
    # 경고
    if np.abs(all_ig).mean() < 1e-10:
        print("\n" + "!" * 70)
        print("⚠️  WARNING: IG values are essentially ZERO!")
        print("!" * 70)
        print("  This confirms the gradient issue detected earlier.")
        print("  The saved NPZ file will contain all zeros.")
        print("!" * 70)
    
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
        print(f"  Target {target_idx}: Total={total_imp:.3e}, Peak frame={most_imp_frame}")
    if n_targets > 5:
        print(f"  ... ({n_targets - 5} more targets)")
    
    # 채널별 평균 중요도 (모든 target 평균)
    avg_channel_imp = channel_importance_all.mean(axis=0)
    if avg_channel_imp.max() > 0:
        avg_channel_imp = avg_channel_imp / avg_channel_imp.max()
    
    print(f"\nAverage Channel Importance (across all targets):")
    for i, name in enumerate(channel_names):
        print(f"  {name:15s}: {avg_channel_imp[i]:.3f}")
    
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
    checkpoint_path = config.validation.checkpoint_path
    
    # Device 설정
    device = "mps"  # or "cpu" or "cuda"
    
    # 출력 디렉토리
    output_dir = Path(config.validation.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("IG GENERATION - ALL TARGETS MODE (DEBUG)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: 64×64")
    print()
    print("🐛 DEBUG MODE ENABLED:")
    print("   - Input statistics check")
    print("   - Model output verification")
    print("   - Gradient computation test")
    print("   - Image dependency analysis")
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
    MAX_BATCHES = 1  # 디버깅 시 1개만
    
    # 분석 파라미터
    target_variable = 0  # Kp, Dst, etc.
    channel_names = ['193Å', '211Å', 'magnetogram']
    
    # IG 파라미터
    N_STEPS = 30
    
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
        output_path = output_dir / f"ig_all_targets_batch_{batch_idx:04d}_debug.npz"
        
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
            
            if result is not None:
                results.append(result)
            else:
                print("\n⚠️  Skipping batch due to critical errors")
            
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Ablation Test
    # ========================================
    print("\n" + "=" * 70)
    print("ABLATION TEST - Checking which input matters")
    print("=" * 70)
    
    print("\nTesting which input (solar wind vs images) affects predictions...")
    
    # Get one batch
    batch = next(iter(dataloader))
    solar_wind = batch["inputs"][:1].to(device)
    images = batch["sdo"][:1].to(device)
    
    with torch.no_grad():
        # 1. Normal (both inputs)
        out_normal = model(solar_wind, images)
        
        # 2. Solar wind only (images = 0)
        out_sw_only = model(solar_wind, torch.zeros_like(images))
        
        # 3. Images only (solar wind = 0)
        out_img_only = model(torch.zeros_like(solar_wind), images)
    
    # Show results for first few targets
    print("\nPredictions for different input combinations:")
    print(f"{'Target':<10} {'Normal':<12} {'SW only':<12} {'Img only':<12}")
    print("-" * 50)
    
    for target_idx in [0, 5, 11, 17, 23]:
        normal_val = out_normal[0, target_idx, 0].item()
        sw_val = out_sw_only[0, target_idx, 0].item()
        img_val = out_img_only[0, target_idx, 0].item()
        
        print(f"{target_idx:<10} {normal_val:<12.6f} {sw_val:<12.6f} {img_val:<12.6f}")
    
    # Calculate differences
    diff_sw = (out_normal - out_sw_only).abs().mean().item()
    diff_img = (out_normal - out_img_only).abs().mean().item()
    
    print(f"\nAverage absolute differences:")
    print(f"  |Normal - SW_only|:  {diff_sw:.6e}")
    print(f"  |Normal - Img_only|: {diff_img:.6e}")
    
    # Diagnosis
    print("\n" + "-" * 70)
    print("ABLATION DIAGNOSIS:")
    print("-" * 70)
    
    if diff_sw < 1e-6:
        print("\n❌ CRITICAL: Output ≈ SW only!")
        print("  → Model completely ignores images")
        print("  → ConvLSTM output is not reaching final layer")
        print("\n  SOLUTION:")
        print("     1. Check model.forward() code")
        print("     2. Verify ConvLSTM features are passed to final FC")
        print("     3. Fix architecture to use both inputs")
        
    elif diff_sw < 1e-3:
        print("\n⚠️  WARNING: Output ≈ SW only with tiny image contribution")
        
        # Calculate contribution percentage
        normal_magnitude = out_normal.abs().mean().item()
        img_contribution = (diff_sw / normal_magnitude) * 100 if normal_magnitude > 0 else 0
        
        print(f"  → Images contribute only ~{img_contribution:.3f}% to predictions")
        print("  → Model heavily relies on solar wind")
        print("\n  POSSIBLE CAUSES:")
        print("     1. Cross-modal fusion weights favor solar wind")
        print("     2. ConvLSTM features are very weak")
        print("     3. Model learned that images are less important")
        print("\n  OPTIONS:")
        print("     A. Accept this (if solar wind is truly more important)")
        print("     B. Retrain with image importance loss")
        print("     C. Use different baseline for IG (not zeros)")
        
    else:
        print("\n✓ Both inputs matter!")
        print(f"  → Solar wind change affects output by {diff_sw:.6e}")
        print(f"  → Image change affects output by {diff_img:.6e}")
        
        if diff_img < diff_sw * 0.01:
            print("\n  Note: Images matter, but much less than solar wind")
            print("  This is why IG gradients are small but not zero")
        else:
            print("\n  Both inputs have reasonable contributions")
    
    # ========================================
    # 최종 요약
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if results:
        print(f"\n✓ Processed {len(results)} batches successfully")
        print(f"✓ Output directory: {output_dir.absolute()}")
        
        total_size = sum(r['file_size_mb'] for r in results)
        n_targets = results[0]['n_targets']
        
        print(f"\nGenerated files:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['output_path'].name} ({result['file_size_mb']:.2f} MB)")
        print(f"\nTotal size: {total_size:.2f} MB")
        
        print(f"\nData shapes:")
        print(f"  data: {results[0]['data_shape']}")
        print(f"  ig:   {results[0]['ig_shape']}")
    else:
        print("\n❌ No results generated")
        print("   Check the debug output above for issues")
    
    print("\n" + "=" * 70)
    print("DEBUG SUMMARY")
    print("=" * 70)
    print("""
If IG values are all zeros, check the debug output for:

1. "CRITICAL ERROR: Gradient is essentially ZERO"
   → Model is not using images
   → Fix: Check model architecture, verify ConvLSTM connection

2. "WARNING: Input is too close to zero baseline"
   → Input normalized incorrectly
   → Fix: Use different baseline or check preprocessing

3. "Output doesn't change with different images"
   → Images disconnected from output
   → Fix: Verify forward pass uses ConvLSTM features

See debug output above for specific recommendations.
    """)


if __name__ == '__main__':
    main()