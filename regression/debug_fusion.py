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
    # 모델 로드 (기존 코드)
    model = load_trained_model(checkpoint_path, config, device)
    model.eval()

    dataloader = create_dataloader(config, phase="validation")
    print("✓ DataLoader loaded\n")

    # 데이터 로드
    batch = next(iter(dataloader))
    solar_wind = batch["inputs"][:1].to(device)
    images = batch["sdo"][:1].to(device)

    print("=" * 70)
    print("FUSION DEBUGGING")
    print("=" * 70)

    with torch.no_grad():
        # 1. 각 모달리티 특징
        transformer_feat = model.transformer_model(solar_wind)
        convlstm_feat = model.convlstm_model(images)
        
        print("\n1. Feature Statistics:")
        print(f"  Transformer features:")
        print(f"    Shape: {transformer_feat.shape}")
        print(f"    Mean: {transformer_feat.mean().item():.6f}")
        print(f"    Std: {transformer_feat.std().item():.6f}")
        print(f"    Max: {transformer_feat.max().item():.6f}")
        
        print(f"\n  ConvLSTM features:")
        print(f"    Shape: {convlstm_feat.shape}")
        print(f"    Mean: {convlstm_feat.mean().item():.6f}")
        print(f"    Std: {convlstm_feat.std().item():.6f}")
        print(f"    Max: {convlstm_feat.max().item():.6f}")
        
        # Magnitude 비교
        tf_magnitude = transformer_feat.abs().mean().item()
        conv_magnitude = convlstm_feat.abs().mean().item()
        ratio = tf_magnitude / (conv_magnitude + 1e-8)
        
        print(f"\n  Magnitude ratio (Transformer/ConvLSTM): {ratio:.2f}×")
        
        if ratio > 100:
            print("  ⚠️  Transformer features are 100× larger!")
            print("  → ConvLSTM will be drowned out in fusion")
        
        # 2. Fusion 출력
        fused_feat = model.cross_modal_fusion(transformer_feat, convlstm_feat)
        
        print(f"\n2. Fused Features:")
        print(f"    Shape: {fused_feat.shape}")
        print(f"    Mean: {fused_feat.mean().item():.6f}")
        print(f"    Std: {fused_feat.std().item():.6f}")
        
        # 3. Fusion이 어느 입력에 더 가까운가?
        diff_tf = (fused_feat - transformer_feat).abs().mean().item()
        diff_conv = (fused_feat - convlstm_feat).abs().mean().item()
        
        print(f"\n3. Fusion Similarity:")
        print(f"    |Fused - Transformer|: {diff_tf:.6f}")
        print(f"    |Fused - ConvLSTM|: {diff_conv:.6f}")
        
        if diff_tf < diff_conv * 0.1:
            print("\n  ❌ Fused ≈ Transformer!")
            print("  → Fusion is ignoring ConvLSTM features")
        
        # 4. Ablation on Fusion
        print("\n4. Fusion Ablation:")
        
        # Normal
        output_normal = model(solar_wind, images)
        
        # Zero ConvLSTM features in fusion
        # (이건 직접 호출할 수 없으므로 우회)
        # 대신 convlstm output을 조작
        
        print("  Testing impact of zeroing ConvLSTM features...")
        
        # ConvLSTM을 0으로 만들고 fusion
        zero_conv = torch.zeros_like(convlstm_feat)
        fused_zero_conv = model.cross_modal_fusion(transformer_feat, zero_conv)
        
        diff_fused = (fused_feat - fused_zero_conv).abs().mean().item()
        print(f"    |Fused(normal) - Fused(zero_conv)|: {diff_fused:.6e}")
        
        if diff_fused < 1e-6:
            print("\n  ❌ CRITICAL: Fusion doesn't change when ConvLSTM=0!")
            print("  → cross_modal_fusion is ignoring convlstm_feat")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if ratio > 100:
        print("\n❌ ROOT CAUSE: Feature Magnitude Imbalance")
        print(f"  → Transformer features are {ratio:.0f}× larger than ConvLSTM")
        print("  → In fusion, ConvLSTM contribution is negligible")
        print("\n  SOLUTION:")
        print("    1. Normalize features before fusion:")
        print("       transformer_feat = F.normalize(transformer_feat, dim=-1)")
        print("       convlstm_feat = F.normalize(convlstm_feat, dim=-1)")
        print("    2. Or add learnable scaling:")
        print("       fused = alpha * transformer_feat + beta * convlstm_feat")
        print("       where alpha, beta are learnable")

    elif diff_fused < 1e-6:
        print("\n❌ ROOT CAUSE: Fusion Ignores ConvLSTM")
        print("  → cross_modal_fusion implementation has bug")
        print("  → Check cross_modal_fusion code")

    print("\n" + "=" * 70)

    # 기존 코드 끝에 추가

    print("\n" + "=" * 70)
    print("REGRESSION HEAD DEBUGGING")
    print("=" * 70)

    with torch.no_grad():
        # 1. Normal prediction
        pred_normal = model.regression_head(fused_feat)
        
        # 2. Prediction with zero ConvLSTM
        pred_zero_conv = model.regression_head(fused_zero_conv)
        
        print("\n1. Regression Head Output:")
        print(f"  Normal prediction:")
        print(f"    Shape: {pred_normal.shape}")
        print(f"    Mean: {pred_normal.mean().item():.6f}")
        print(f"    Sample: {pred_normal[0, :5].tolist()}")
        
        print(f"\n  With zero ConvLSTM:")
        print(f"    Mean: {pred_zero_conv.mean().item():.6f}")
        print(f"    Sample: {pred_zero_conv[0, :5].tolist()}")
        
        # 3. Difference
        diff_pred = (pred_normal - pred_zero_conv).abs().mean().item()
        
        print(f"\n2. Prediction Difference:")
        print(f"    |Pred(normal) - Pred(zero_conv)|: {diff_pred:.6e}")
        
        # 4. Reshaped output
        output_normal = pred_normal.reshape(1, 24, 1)
        output_zero_conv = pred_zero_conv.reshape(1, 24, 1)
        
        diff_output = (output_normal - output_zero_conv).abs().mean().item()
        
        print(f"\n3. Final Output Difference:")
        print(f"    |Output(normal) - Output(zero_conv)|: {diff_output:.6e}")
        
        # Compare with full ablation
        print(f"\n4. Comparison:")
        print(f"    Fusion level difference:     {diff_fused:.6e}")
        print(f"    Regression head difference:  {diff_pred:.6e}")
        print(f"    Final output difference:     {diff_output:.6e}")
        print(f"    Full ablation difference:    {8.501423e-07:.6e}")
        
        # 5. Regression head weights analysis
        print(f"\n5. Regression Head Parameters:")
        
        for name, param in model.regression_head.named_parameters():
            print(f"\n  {name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Mean: {param.mean().item():.6e}")
            print(f"    Std: {param.std().item():.6e}")
            print(f"    Max |param|: {param.abs().max().item():.6e}")

    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS")
    print("=" * 70)

    if diff_output < 1e-6:
        print("\n❌ CRITICAL: Regression head nullifies ConvLSTM contribution!")
        print(f"  Fusion difference: {diff_fused:.6e} → Output difference: {diff_output:.6e}")
        print("\n  Possible causes:")
        print("    1. Regression head weights for ConvLSTM features ≈ 0")
        print("    2. Regression head has learned to ignore ConvLSTM component")
        print("    3. Architecture issue in regression head")
        
        print("\n  Check regression_head code!")

    elif diff_pred > 1e-3 and diff_output < 1e-6:
        print("\n❌ CRITICAL: Reshape or post-processing nullifies difference!")
        print(f"  Pred difference: {diff_pred:.6e} → Output difference: {diff_output:.6e}")
        print("\n  Issue is in the reshape or final processing step")

    else:
        print("\n⚠️  Unknown issue - investigate further")
        print("  Differences don't match expected pattern")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()