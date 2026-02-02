"""
ConvLSTM 작동 여부를 확인하는 진단 스크립트

문제:
1. Hidden state가 전달되는가?
2. 시간에 따라 feature가 변하는가?
3. Gradient가 흐르는가?
"""

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from networks import create_model
from pipeline import create_dataloader


def diagnose_convlstm(model, dataloader, device='cpu'):
    """ConvLSTM의 작동 상태 진단"""
    
    print("\n" + "="*70)
    print("CONVLSTM DIAGNOSTIC")
    print("="*70)
    
    batch = next(iter(dataloader))
    images = batch['sdo'][:1].to(device)  # (1, C, T, H, W)
    
    seq_len = images.shape[2]
    
    # 1. 각 시점의 output 확인
    print("\n1. Temporal Feature Evolution")
    print("-" * 70)
    
    model.eval()
    with torch.no_grad():
        # ConvLSTM 통과
        features = model.convlstm_model(images)  # (1, hidden_dim)
        
        print(f"Final feature shape: {features.shape}")
        print(f"Final feature mean: {features.mean():.6f}")
        print(f"Final feature std: {features.std():.6f}")
        
    # 2. 각 시점별로 개별 처리
    print("\n2. Per-Timestep Analysis")
    print("-" * 70)
    
    timestep_features = []
    for t in range(seq_len):
        single_frame = images[:, :, t:t+1, :, :]  # (1, C, 1, H, W)
        
        with torch.no_grad():
            feat = model.convlstm_model(single_frame)
            timestep_features.append(feat.cpu().numpy())
            
        if t % 5 == 0:
            print(f"t={t:2d}: mean={feat.mean():.6f}, std={feat.std():.6f}")
    
    timestep_features = np.array(timestep_features)
    
    # 3. 시간별 변화 분석
    print("\n3. Temporal Variation Analysis")
    print("-" * 70)
    
    # 각 시점 간 차이
    diffs = []
    for t in range(1, seq_len):
        diff = np.abs(timestep_features[t] - timestep_features[t-1]).mean()
        diffs.append(diff)
    
    mean_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    
    print(f"Mean temporal change: {mean_diff:.6f}")
    print(f"Max temporal change: {max_diff:.6f}")
    
    if mean_diff < 1e-6:
        print("\n🔴 CRITICAL: Features do NOT change over time!")
        print("   → ConvLSTM is producing identical outputs")
        print("   → Hidden state is not being updated")
    elif mean_diff < 1e-3:
        print("\n⚠️  WARNING: Very small temporal changes")
        print("   → ConvLSTM barely uses temporal information")
    else:
        print("\n✓ Features change over time (Good)")
    
    # 4. Gradient 확인
    print("\n4. Gradient Flow Check")
    print("-" * 70)
    
    model.train()
    images.requires_grad = True
    
    # Forward
    output = model.convlstm_model(images)
    loss = output.sum()
    
    # Backward
    loss.backward()
    
    grad_magnitude = images.grad.abs().mean().item()
    print(f"Gradient magnitude: {grad_magnitude:.6f}")
    
    if grad_magnitude < 1e-6:
        print("\n🔴 CRITICAL: No gradient flow!")
        print("   → ConvLSTM is not learning")
    elif grad_magnitude < 1e-3:
        print("\n⚠️  WARNING: Very small gradients")
        print("   → Vanishing gradient problem")
    else:
        print("\n✓ Gradients are flowing (Good)")
    
    # 5. Hidden state 확인 (LSTM 내부)
    print("\n5. Hidden State Analysis")
    print("-" * 70)
    
    # ConvLSTM의 hidden state에 접근
    try:
        # 모델 구조에 따라 경로 조정 필요
        convlstm = model.convlstm_model
        
        # Reset hidden state
        if hasattr(convlstm, 'reset_hidden_state'):
            convlstm.reset_hidden_state()
        
        # 첫 프레임 처리
        h_init = None
        if hasattr(convlstm, 'hidden_state'):
            first_frame = images[:, :, 0:1, :, :]
            _ = convlstm(first_frame)
            h_init = convlstm.hidden_state
            
            # 두 번째 프레임 처리
            second_frame = images[:, :, 1:2, :, :]
            _ = convlstm(second_frame)
            h_after = convlstm.hidden_state
            
            # 변화 확인
            if h_init is not None and h_after is not None:
                h_change = (h_after - h_init).abs().mean().item()
                print(f"Hidden state change: {h_change:.6f}")
                
                if h_change < 1e-6:
                    print("\n🔴 CRITICAL: Hidden state is NOT updating!")
                else:
                    print("\n✓ Hidden state is updating (Good)")
        else:
            print("⚠️  Cannot access hidden state directly")
            
    except Exception as e:
        print(f"⚠️  Error accessing hidden state: {e}")
    
    # 6. 최종 진단
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    problems = []
    
    if mean_diff < 1e-6:
        problems.append("ConvLSTM outputs are identical across time")
    if grad_magnitude < 1e-6:
        problems.append("No gradient flow through ConvLSTM")
    
    if problems:
        print("\n🔴 PROBLEMS DETECTED:")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        
        print("\n💡 RECOMMENDED ACTIONS:")
        print("  1. Check ConvLSTM implementation")
        print("  2. Verify input shape: (batch, channels, time, H, W)")
        print("  3. Check if hidden state is being passed correctly")
        print("  4. Increase learning rate for ConvLSTM")
        print("  5. Add gradient clipping")
        
    else:
        print("\n✓ ConvLSTM appears to be working")
        print("\nBut saliency maps show the model is NOT using temporal info!")
        print("This suggests:")
        print("  1. ConvLSTM works but has very low weight in fusion")
        print("  2. Transformer dominates the prediction")
        print("  3. Need to increase ConvLSTM influence")


@hydra.main(config_path="./configs", config_name="saliency", version_base=None)
def main(config: DictConfig):
    
    checkpoint_path = checkpoint_path = "/Users/eunsupark/checkpoints/SINGLE_7_1_05/checkpoint/model_epoch0100.pth"
    device = "cpu"
    
    print("="*70)
    print("CONVLSTM DIAGNOSTIC TOOL")
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
    diagnose_convlstm(model, dataloader, device)


if __name__ == '__main__':
    main()
