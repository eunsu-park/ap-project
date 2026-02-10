"""
Attention Analysis - All Batches

Extract and analyze attention weights from all validation batches.
Similar workflow to example_ig_all_frames.py but much faster (forward-only).

기능:
- 모든 validation batch에 대해 attention weights 추출
- NPZ 파일로 저장 (원본 데이터 + attention + 통계)
- 시각화 (optional)

NPZ 구조:
- solar_wind_data: (seq_len, num_vars) - 원본 OMNI 데이터
- sdo_data: (3, seq_len, 64, 64) - 원본 SDO 이미지
- attention_weights: (num_layers, num_heads, seq_len, seq_len) - 모든 layer의 attention
- temporal_importance: (num_layers, seq_len) - layer별 시간 중요도
- predictions: (n_targets, num_vars) - 모델 예측
- targets: (n_targets, num_vars) - Ground truth
- metadata: dict - 설정 정보

Updated: 2025-01-22
Expected file size: ~5-10 MB per batch (much smaller than IG)
"""

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
import sys

# Import custom modules
sys.path.append('.')
from networks import create_model
from pipeline import create_dataloader
from attention_analysis import AttentionExtractor


def load_trained_model(checkpoint_path: str, config: DictConfig, device: str = 'cuda'):
    """
    훈련된 모델 로드 (example_ig_all_frames.py와 동일)
    """
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


def generate_attention_analysis(
    extractor: AttentionExtractor,
    solar_wind_input: torch.Tensor,
    image_input: torch.Tensor,
    targets: torch.Tensor,
    output_path: Path,
    config: DictConfig,
    create_plots: bool = False
):
    """
    단일 배치에 대한 attention 분석 및 저장
    
    Args:
        extractor: AttentionExtractor instance
        solar_wind_input: (batch, seq_len, num_vars)
        image_input: (batch, channels, seq_len, H, W)
        targets: (batch, target_seq_len, num_vars)
        output_path: 저장할 NPZ 파일 경로
        config: Hydra config
        create_plots: 시각화 생성 여부
        
    Saves:
        NPZ file with attention weights and statistics
    """
    
    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS - SINGLE BATCH")
    print("=" * 70)
    
    # Shape 정보
    batch, seq_len, num_vars = solar_wind_input.shape
    _, channels, sdo_seq_len, H, W = image_input.shape
    _, n_targets, n_target_vars = targets.shape
    
    print(f"\nInput shapes:")
    print(f"  Solar wind: {solar_wind_input.shape}")
    print(f"  SDO images: {image_input.shape}")
    print(f"  Targets: {targets.shape}")
    
    # ================================================================
    # 1. Attention Weights 추출
    # ================================================================
    print(f"\n[1/3] Extracting attention weights...")
    
    attention_weights, predictions = extractor.extract_attention_manual_forward(
        solar_wind_input, image_input
    )
    
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]
    
    print(f"✓ Extracted attention from {num_layers} layers")
    print(f"  Each layer: (batch={batch}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len})")
    
    # ================================================================
    # 2. Temporal Importance 계산
    # ================================================================
    print(f"\n[2/3] Computing temporal importance...")
    
    temporal_importance_all = []
    
    for layer_idx, attn in enumerate(attention_weights):
        temporal_imp = extractor.compute_temporal_importance(
            attn[0],  # First batch
            method='incoming'
        )
        temporal_importance_all.append(temporal_imp)
        
        print(f"  Layer {layer_idx}: Most important timestep = {temporal_imp.argmax()} "
              f"(score={temporal_imp.max():.4f})")
    
    temporal_importance_all = np.array(temporal_importance_all)  # (num_layers, seq_len)
    
    # ================================================================
    # 3. NPZ 저장
    # ================================================================
    print(f"\n[3/3] Saving to NPZ...")
    
    # Convert attention weights to numpy
    attention_weights_np = [attn[0].cpu().numpy() for attn in attention_weights]
    attention_weights_np = np.array(attention_weights_np)  # (num_layers, num_heads, seq_len, seq_len)
    
    # 메타데이터
    metadata = {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'num_vars': num_vars,
        'n_targets': n_targets,
        'sdo_channels': channels,
        'sdo_seq_len': sdo_seq_len,
        'image_size': H,
        'd_model': config.model.transformer_d_model,
        'checkpoint': config.validation.checkpoint_path
    }
    
    # 저장
    np.savez_compressed(
        output_path,
        solar_wind_data=solar_wind_input[0].cpu().numpy(),         # (seq_len, num_vars)
        sdo_data=image_input[0].cpu().numpy(),                     # (channels, sdo_seq_len, H, W)
        attention_weights=attention_weights_np,                     # (num_layers, num_heads, seq_len, seq_len)
        temporal_importance=temporal_importance_all,                # (num_layers, seq_len)
        predictions=predictions[0].cpu().numpy(),                   # (n_targets, n_target_vars)
        targets=targets[0].cpu().numpy(),                           # (n_targets, n_target_vars)
        metadata=metadata
    )
    
    # 파일 크기
    file_size_mb = output_path.stat().st_size / (1024**2)
    
    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # ================================================================
    # 4. 시각화 (Optional)
    # ================================================================
    if create_plots:
        print(f"\n[4/4] Creating visualizations...")
        
        plot_dir = output_path.parent / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        batch_name = output_path.stem
        
        # Attention heatmap (마지막 layer)
        extractor.visualize_attention_heatmap(
            attention_weights[-1][0],  # Last layer, first batch
            save_path=plot_dir / f"{batch_name}_heatmap.png",
            title=f"Attention Heatmap - {batch_name}"
        )
        
        # Attention matrix (마지막 layer, 모든 head)
        extractor.visualize_attention_matrix(
            attention_weights[-1][0],  # Last layer, first batch
            save_path=plot_dir / f"{batch_name}_matrix.png",
            title=f"Attention Matrix - {batch_name}"
        )
        
        # Layer comparison
        all_layers_batch0 = [attn[0] for attn in attention_weights]
        extractor.visualize_layer_comparison(
            all_layers_batch0,
            save_path=plot_dir / f"{batch_name}_layers.png"
        )
        
        print(f"✓ Plots saved to: {plot_dir}")
    
    # ================================================================
    # 5. 요약
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Attention analysis complete!")
    print(f"✓ Data saved to: {output_path.absolute()}")
    
    print(f"\nNPZ Contents:")
    print(f"  'solar_wind_data':       ({seq_len}, {num_vars})")
    print(f"  'sdo_data':              ({channels}, {sdo_seq_len}, {H}, {W})")
    print(f"  'attention_weights':     ({num_layers}, {num_heads}, {seq_len}, {seq_len})")
    print(f"  'temporal_importance':   ({num_layers}, {seq_len})")
    print(f"  'predictions':           ({n_targets}, {n_target_vars})")
    print(f"  'targets':               ({n_targets}, {n_target_vars})")
    print(f"  'metadata':              dict")
    
    # Temporal importance 요약
    print(f"\nTemporal Importance Summary:")
    print(f"  {'Layer':<8} {'Peak Frame':<12} {'Peak Score':<12} {'Mean Score':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for layer_idx in range(num_layers):
        peak_frame = temporal_importance_all[layer_idx].argmax()
        peak_score = temporal_importance_all[layer_idx].max()
        mean_score = temporal_importance_all[layer_idx].mean()
        print(f"  {layer_idx:<8} {peak_frame:<12} {peak_score:<12.4f} {mean_score:<12.4f}")
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print(f"""
# Load the data
import numpy as np

data = np.load('{output_path.name}')

# Access arrays
solar_wind = data['solar_wind_data']            # ({seq_len}, {num_vars})
sdo_images = data['sdo_data']                   # ({channels}, {sdo_seq_len}, {H}, {W})
attention = data['attention_weights']           # ({num_layers}, {num_heads}, {seq_len}, {seq_len})
temporal_imp = data['temporal_importance']      # ({num_layers}, {seq_len})
predictions = data['predictions']               # ({n_targets}, {n_target_vars})
targets = data['targets']                       # ({n_targets}, {n_target_vars})

# Example 1: Get attention from last layer
last_layer_attention = attention[-1]  # ({num_heads}, {seq_len}, {seq_len})

# Example 2: Average attention over all heads
avg_attention = attention[-1].mean(axis=0)  # ({seq_len}, {seq_len})

# Example 3: Find most important timestep
most_important = temporal_imp[-1].argmax()
print(f"Most important timestep: {{most_important}}")

# Example 4: Compare prediction vs target
mse = ((predictions - targets) ** 2).mean()
print(f"MSE: {{mse:.4f}}")

# Example 5: Analyze attention focus
# Higher values = more focused attention
attention_entropy = -(attention * np.log(attention + 1e-10)).sum(axis=-1).mean()
print(f"Attention entropy: {{attention_entropy:.4f}}")
    """)
    
    return {
        'output_path': output_path,
        'file_size_mb': file_size_mb,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'temporal_importance': temporal_importance_all,
        'predictions': predictions[0].cpu().numpy(),
        'targets': targets[0].cpu().numpy()
    }


@hydra.main(config_path="./configs", config_name="attention_0", version_base=None)
def main(config: DictConfig):
    """메인 실행 함수"""

    # ========================================
    # 설정
    # ========================================
    checkpoint_path = config.attention.checkpoint_path

    # Device 설정
    device = config["environment"]["device"]

    # 출력 디렉토리
    output_dir = Path(config.attention.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Model type
    model_type = config.model.model_type

    # ConvLSTM-only models don't have transformer attention
    if model_type == 'convlstm':
        print("=" * 70)
        print("⚠️  ATTENTION ANALYSIS NOT AVAILABLE FOR CONVLSTM-ONLY MODEL")
        print("=" * 70)
        print("ConvLSTM models don't have Transformer attention weights.")
        print("Use saliency analysis (IG) instead for ConvLSTM interpretability.")
        print()
        return

    print("=" * 70)
    print("ATTENTION ANALYSIS - ALL BATCHES MODE")
    print("=" * 70)
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    print("⚡ NOTE: This is MUCH faster than IG (~50-100x)")
    print("   Forward pass only, no gradients needed!")
    print()
    
    # ========================================
    # 모델 로드
    # ========================================
    model = load_trained_model(checkpoint_path, config, device)
    
    # ========================================
    # DataLoader
    # ========================================
    dataloader = create_dataloader(config, phase="validation")
    print(f"✓ DataLoader loaded (total batches: {len(dataloader)})\n")
    
    # ========================================
    # AttentionExtractor 초기화
    # ========================================
    extractor = AttentionExtractor(model, device=device)
    print("✓ AttentionExtractor initialized\n")
    
    # ========================================
    # 설정
    # ========================================
    MAX_BATCHES = len(dataloader)
    # MAX_BATCHES = 3  # 테스트용

    # 시각화 생성 여부 (from config)
    CREATE_PLOTS = config.attention.create_plots
    
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
        targets = batch["targets"][:1]
        
        # 출력 파일명
        output_path = output_dir / f"attention_batch_{batch_idx:04d}.npz"
        
        try:
            result = generate_attention_analysis(
                extractor=extractor,
                solar_wind_input=solar_wind_input,
                image_input=image_input,
                targets=targets,
                output_path=output_path,
                config=config,
                create_plots=CREATE_PLOTS
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
        num_layers = results[0]['num_layers']
        num_heads = results[0]['num_heads']
        
        print(f"\nGenerated files:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['output_path'].name} ({result['file_size_mb']:.2f} MB)")
        print(f"\nTotal size: {total_size:.2f} MB")
        
        print(f"\nModel architecture:")
        print(f"  Transformer layers: {num_layers}")
        print(f"  Attention heads: {num_heads}")
        
        # 평균 temporal importance
        avg_temporal_imp = np.mean([r['temporal_importance'] for r in results], axis=0)
        print(f"\nAverage Temporal Importance (across all batches):")
        for layer_idx in range(num_layers):
            peak_frame = avg_temporal_imp[layer_idx].argmax()
            print(f"  Layer {layer_idx}: Peak at timestep {peak_frame} "
                  f"(score={avg_temporal_imp[layer_idx, peak_frame]:.4f})")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Load and analyze NPZ files:
   data = np.load('attention_batch_0000.npz')
   attention = data['attention_weights']  # (num_layers, num_heads, seq_len, seq_len)
   
2. Compare with IG results:
   - IG shows gradient-based importance
   - Attention shows what model directly attends to
   - Both should align for well-trained models
   
3. Analyze attention patterns:
   - Which timesteps get most attention?
   - How does attention change across layers?
   - Is attention focused or uniform?
   
4. Compare with paper findings (DeepHalo):
   - Positive predictions: Uniform attention (progressive process)
   - Negative predictions: Early-focused attention
    """)


if __name__ == '__main__':
    main()
