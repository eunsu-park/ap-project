"""
Cross-Modal Fusion의 가중치 확인

문제:
ConvLSTM features가 너무 작거나 fusion에서 무시되는가?
"""

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from networks import create_model
from pipeline import create_dataloader


def diagnose_fusion(model, dataloader, device='cpu'):
    """Cross-Modal Fusion 진단"""
    
    print("\n" + "="*70)
    print("CROSS-MODAL FUSION DIAGNOSTIC")
    print("="*70)
    
    batch = next(iter(dataloader))
    solar_wind = batch['inputs'][:1].to(device)
    images = batch['sdo'][:1].to(device)
    
    model.eval()
    
    # 1. 각 모듈의 출력 크기 비교
    print("\n1. Feature Magnitude Comparison")
    print("-" * 70)
    
    with torch.no_grad():
        # Transformer features
        transformer_feat = model.transformer_model(solar_wind)
        
        # ConvLSTM features
        convlstm_feat = model.convlstm_model(images)
        
        # Fused features
        fused_feat = model.cross_modal_fusion(transformer_feat, convlstm_feat)
        
        print(f"Transformer output:")
        print(f"  Shape: {transformer_feat.shape}")
        print(f"  Mean:  {transformer_feat.abs().mean():.6f}")
        print(f"  Std:   {transformer_feat.std():.6f}")
        print(f"  Max:   {transformer_feat.abs().max():.6f}")
        
        print(f"\nConvLSTM output:")
        print(f"  Shape: {convlstm_feat.shape}")
        print(f"  Mean:  {convlstm_feat.abs().mean():.6f}")
        print(f"  Std:   {convlstm_feat.std():.6f}")
        print(f"  Max:   {convlstm_feat.abs().max():.6f}")
        
        print(f"\nFused output:")
        print(f"  Shape: {fused_feat.shape}")
        print(f"  Mean:  {fused_feat.abs().mean():.6f}")
        print(f"  Std:   {fused_feat.std():.6f}")
        
        # 크기 비율
        t_mag = transformer_feat.abs().mean().item()
        c_mag = convlstm_feat.abs().mean().item()
        
        print(f"\n📊 Magnitude Ratio:")
        print(f"  Transformer / ConvLSTM = {t_mag / (c_mag + 1e-10):.2f}x")
        
        if t_mag > c_mag * 100:
            print("\n🔴 CRITICAL: Transformer features 100x larger!")
            print("   → Cross-modal fusion will ignore ConvLSTM")
            print("   → Need to normalize or rescale features")
        elif t_mag > c_mag * 10:
            print("\n⚠️  WARNING: Transformer features 10x larger")
            print("   → ConvLSTM has minimal influence")
        else:
            print("\n✓ Feature magnitudes are balanced")
    
    # 2. Fusion 가중치 확인 (attention 기반인 경우)
    print("\n2. Fusion Mechanism Analysis")
    print("-" * 70)
    
    try:
        # Fusion layer의 파라미터 확인
        fusion = model.cross_modal_fusion
        
        # Attention weights 추출 (구현에 따라 다름)
        if hasattr(fusion, 'attention'):
            print("✓ Fusion uses attention mechanism")
            
            with torch.no_grad():
                # Attention 계산
                attn = fusion.attention(transformer_feat, convlstm_feat)
                
                print(f"\nAttention weights:")
                print(f"  Transformer: {attn[0].mean():.4f}")
                print(f"  ConvLSTM:    {attn[1].mean():.4f}")
                
                if attn[1].mean() < 0.1:
                    print("\n🔴 CRITICAL: ConvLSTM attention < 10%!")
                    print("   → Model is ignoring visual features")
                    
        elif hasattr(fusion, 'gate'):
            print("✓ Fusion uses gating mechanism")
            
            # Gate 출력 확인
            with torch.no_grad():
                gate = fusion.gate(torch.cat([transformer_feat, convlstm_feat], dim=-1))
                print(f"\nGate values: {gate.mean():.4f} ± {gate.std():.4f}")
                
                if gate.mean() < 0.1 or gate.mean() > 0.9:
                    print("\n⚠️  Gate is saturated (close to 0 or 1)")
                    print("   → One modality dominates")
                    
        else:
            print("⚠️  Fusion type unknown")
            print("   → Check fusion implementation")
            
    except Exception as e:
        print(f"⚠️  Error analyzing fusion: {e}")
    
    # 3. Ablation test
    print("\n3. Ablation Test")
    print("-" * 70)
    
    with torch.no_grad():
        # Full model
        output_full = model(solar_wind, images)
        
        # Without ConvLSTM (zero images)
        output_no_conv = model(solar_wind, torch.zeros_like(images))
        
        # Without Transformer (zero OMNI)
        output_no_trans = model(torch.zeros_like(solar_wind), images)
        
        diff_no_conv = (output_full - output_no_conv).abs().mean().item()
        diff_no_trans = (output_full - output_no_trans).abs().mean().item()
        
        print(f"Prediction change when removing:")
        print(f"  ConvLSTM:    {diff_no_conv:.6f}")
        print(f"  Transformer: {diff_no_trans:.6f}")
        
        print(f"\n📊 Contribution Ratio:")
        print(f"  Transformer / ConvLSTM = {diff_no_trans / (diff_no_conv + 1e-10):.2f}x")
        
        if diff_no_conv < 1e-4:
            print("\n🔴 CRITICAL: Removing ConvLSTM has NO effect!")
            print("   → Model is NOT using visual features")
        elif diff_no_conv < diff_no_trans * 0.1:
            print("\n⚠️  WARNING: ConvLSTM contributes < 10%")
            print("   → Transformer dominates prediction")
    
    # 4. 최종 진단
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print("\n💡 RECOMMENDED ACTIONS:")
    
    if c_mag < t_mag * 0.01:
        print("\n1. Feature Normalization:")
        print("   - Add LayerNorm before fusion")
        print("   - Or manually rescale ConvLSTM output")
        
    if diff_no_conv < 1e-4:
        print("\n2. Retrain with:")
        print("   - Higher learning rate for ConvLSTM")
        print("   - Separate optimizer for each branch")
        print("   - Auxiliary loss on ConvLSTM features")
        
    print("\n3. Architecture Changes:")
    print("   - Use learnable fusion weights")
    print("   - Add skip connections")
    print("   - Increase ConvLSTM hidden size")


@hydra.main(config_path="./configs", config_name="saliency", version_base=None)
def main(config: DictConfig):
    
    checkpoint_path = "/Users/eunsupark/checkpoints/SINGLE_7_1_05/checkpoint/model_epoch0100.pth"
    device = "cpu"
    
    print("="*70)
    print("CROSS-MODAL FUSION DIAGNOSTIC TOOL")
    print("="*70)
    print(f"\nCheckpoint: {checkpoint_path}")
    
    # Load model
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print("✓ Model loaded")
    
    # Load data
    dataloader = create_dataloader(config, phase='validation')
    print("✓ DataLoader loaded")
    
    # Run diagnostics
    diagnose_fusion(model, dataloader, device)


if __name__ == '__main__':
    main()
