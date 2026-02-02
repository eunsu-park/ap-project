"""
Saliency/Attribution Maps for AIA (SDO) Image Inputs

훈련된 모델에서 SDO 이미지의 어느 부분이 예측에 중요한지 분석하는 도구.

지원하는 방법:
1. Grad-CAM: ConvLSTM의 spatial activation + gradient
2. Integrated Gradients: 픽셀별 기여도
3. Occlusion Sensitivity: 영역 가리기 실험
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List, Tuple, Dict
from pathlib import Path


class SaliencyExtractor:
    """훈련된 모델에서 saliency/attribution maps 추출"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Gradient와 activation 저장용
        self.gradients = None
        self.activations = None
        
    # ================================================================
    # Grad-CAM for ConvLSTM
    # ================================================================
    
    def _register_hooks(self):
        """ConvLSTM 마지막 레이어에 hook 등록"""
        
        def forward_hook(module, input, output):
            # ConvLSTMCell의 output: (hidden, cell)
            self.activations = output[0].detach()  # hidden state
        
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: hidden에 대한 gradient
            self.gradients = grad_output[0].detach()
        
        # ConvLSTM 마지막 레이어에 hook 등록
        target_layer = self.model.convlstm_model.convlstm_layers[-1]
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    def _remove_hooks(self):
        """Hook 제거"""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
    
    def grad_cam(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0
    ) -> np.ndarray:
        """
        Grad-CAM으로 spatial saliency map 생성
        
        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: 어느 시점의 예측을 분석할지 (target sequence 내)
            target_variable: 어느 변수를 분석할지 (e.g., ap_Index)
            
        Returns:
            saliency_maps: (seq_len, H, W) - 각 time step별 saliency map
        """
        self._register_hooks()
        
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)
        image_input.requires_grad = True
        
        batch_size, channels, seq_len, H, W = image_input.shape
        saliency_maps = []
        
        # 각 time step마다 Grad-CAM 계산
        for t in range(seq_len):
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(solar_wind_input, image_input)
            
            # Target: 특정 시점, 특정 변수의 예측값
            target = output[0, target_index, target_variable]
            
            # Backward
            target.backward(retain_graph=(t < seq_len - 1))
            
            # Grad-CAM 계산
            if self.gradients is not None and self.activations is not None:
                # Global average pooling of gradients
                weights = self.gradients[0].mean(dim=[1, 2], keepdim=True)  # (channels, 1, 1)
                
                # Weighted sum of activations
                cam = (weights * self.activations[0]).sum(dim=0)  # (H, W)
                
                # ReLU
                cam = F.relu(cam)
                
                # Normalize
                cam = cam.cpu().numpy()
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                saliency_maps.append(cam)
            else:
                saliency_maps.append(np.zeros((H, W)))
        
        self._remove_hooks()
        
        return np.array(saliency_maps)  # (seq_len, H, W)
    
    # ================================================================
    # Integrated Gradients
    # ================================================================
    
    def integrated_gradients(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Integrated Gradients로 픽셀별 기여도 계산
        
        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: 예측 시점
            target_variable: 예측 변수
            n_steps: Interpolation steps
            baseline: Baseline 이미지 (None이면 zeros)
            
        Returns:
            attributions: (channels, seq_len, H, W) - 각 픽셀의 기여도
        """
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)
        
        # Baseline (검은 이미지)
        if baseline is None:
            baseline = torch.zeros_like(image_input)
        else:
            baseline = baseline.to(self.device)
        
        # Integrated gradients 계산
        attributions = torch.zeros_like(image_input)
        
        for step in range(n_steps):
            # Interpolation - clone()과 detach() 사용
            alpha = step / n_steps
            interpolated = (baseline + alpha * (image_input - baseline)).clone().detach()
            interpolated.requires_grad = True  # leaf variable로 만들기
            
            # Forward
            self.model.zero_grad()
            output = self.model(solar_wind_input, interpolated)
            target = output[0, target_index, target_variable]
            
            # Backward
            target.backward()
            
            # Accumulate gradients
            if interpolated.grad is not None:
                attributions += interpolated.grad.detach() / n_steps
        
        # Scale by (input - baseline)
        attributions = attributions * (image_input - baseline)
        
        return attributions[0].detach().cpu().numpy()  # (channels, seq_len, H, W)
    
    def integrated_gradients_batch_targets(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_variable: int = 0,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Integrated Gradients - 모든 target을 한 번에 계산 (Batch version)
        
        Sequential version보다 약 5-6배 빠름:
        - Sequential: n_targets × n_steps 번의 forward/backward
        - Batch: n_steps 번만 (각 step에서 모든 target 동시 계산)
        
        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_variable: 예측 변수
            n_steps: Interpolation steps
            baseline: Baseline 이미지 (None이면 zeros)
            
        Returns:
            attributions: (n_targets, channels, seq_len, H, W) - 각 target별 픽셀 기여도
        """
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)
        
        # Baseline (검은 이미지)
        if baseline is None:
            baseline = torch.zeros_like(image_input)
        else:
            baseline = baseline.to(self.device)
        
        # Output shape 확인 (n_targets 개수)
        with torch.no_grad():
            test_output = self.model(solar_wind_input, image_input)
        n_targets = test_output.shape[1]  # (batch, n_targets, n_variables)
        
        batch_size, channels, seq_len, H, W = image_input.shape
        
        # 모든 target의 attribution 저장
        # Shape: (n_targets, channels, seq_len, H, W)
        all_attributions = torch.zeros(n_targets, channels, seq_len, H, W, device=self.device)
        
        print(f"Computing IG for {n_targets} targets with {n_steps} steps...")
        print("Using batch mode (all targets simultaneously)...")
        
        # Step별로 모든 target 계산
        for step in range(n_steps):
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Step {step+1}/{n_steps}...", end='\r')
            
            # Interpolation
            alpha = step / n_steps
            interpolated = (baseline + alpha * (image_input - baseline)).clone().detach()
            interpolated.requires_grad = True
            
            # Forward
            self.model.zero_grad()
            output = self.model(solar_wind_input, interpolated)
            # output shape: (batch, n_targets, n_variables)
            
            # 각 target에 대해 gradient 계산
            for target_idx in range(n_targets):
                target = output[0, target_idx, target_variable]
                
                # Backward (마지막 target 제외하고는 graph 유지)
                target.backward(retain_graph=(target_idx < n_targets - 1))
                
                # Gradient 저장
                if interpolated.grad is not None:
                    all_attributions[target_idx] += interpolated.grad[0].detach() / n_steps
                    
                    # Gradient 초기화 (다음 target을 위해)
                    interpolated.grad.zero_()
        
        print()  # 줄바꿈
        
        # Scale by (input - baseline) for each target
        input_diff = (image_input - baseline)[0]  # (channels, seq_len, H, W)
        
        # Broadcasting: (n_targets, channels, seq_len, H, W) * (channels, seq_len, H, W)
        all_attributions = all_attributions * input_diff.unsqueeze(0)
        
        print("✓ Batch IG computation complete!")
        
        return all_attributions.detach().cpu().numpy()  # (n_targets, channels, seq_len, H, W)
    
    # ================================================================
    # Occlusion Sensitivity
    # ================================================================
    
    def occlusion_sensitivity(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        patch_size: int = 16,
        stride: int = 8
    ) -> np.ndarray:
        """
        Occlusion Sensitivity로 중요 영역 찾기
        
        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: 예측 시점
            target_variable: 예측 변수
            patch_size: 가릴 패치 크기
            stride: 슬라이딩 간격
            
        Returns:
            sensitivity_maps: (seq_len, H, W) - 각 영역의 중요도
        """
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)
        
        batch_size, channels, seq_len, H, W = image_input.shape
        
        # Baseline prediction
        with torch.no_grad():
            baseline_output = self.model(solar_wind_input, image_input)
            baseline_pred = baseline_output[0, target_index, target_variable].item()
        
        sensitivity_maps = []
        
        # 각 time step에 대해
        for t in range(seq_len):
            sensitivity_map = np.zeros((H, W))
            
            # Sliding window
            for i in range(0, H - patch_size + 1, stride):
                for j in range(0, W - patch_size + 1, stride):
                    # 패치를 0으로 가림
                    occluded = image_input.clone()
                    occluded[:, :, t, i:i+patch_size, j:j+patch_size] = 0
                    
                    # 예측
                    with torch.no_grad():
                        output = self.model(solar_wind_input, occluded)
                        occluded_pred = output[0, target_index, target_variable].item()
                    
                    # 예측 변화 = 중요도
                    importance = abs(baseline_pred - occluded_pred)
                    sensitivity_map[i:i+patch_size, j:j+patch_size] = np.maximum(
                        sensitivity_map[i:i+patch_size, j:j+patch_size],
                        importance
                    )
            
            # Normalize
            if sensitivity_map.max() > 0:
                sensitivity_map = sensitivity_map / sensitivity_map.max()
            
            sensitivity_maps.append(sensitivity_map)
        
        return np.array(sensitivity_maps)  # (seq_len, H, W)
    
    # ================================================================
    # Temporal Importance
    # ================================================================
    
    def temporal_importance(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0
    ) -> np.ndarray:
        """
        각 time step의 중요도 계산 (제거했을 때 예측 변화)
        
        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: 예측 시점
            target_variable: 예측 변수
            
        Returns:
            importance: (seq_len,) - 각 시간의 중요도
        """
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)
        
        seq_len = image_input.shape[2]
        
        # Baseline prediction
        with torch.no_grad():
            baseline_output = self.model(solar_wind_input, image_input)
            baseline_pred = baseline_output[0, target_index, target_variable].item()
        
        importance = []
        
        for t in range(seq_len):
            # t번째 time step을 0으로 만듦
            masked = image_input.clone()
            masked[:, :, t, :, :] = 0
            
            with torch.no_grad():
                output = self.model(solar_wind_input, masked)
                masked_pred = output[0, target_index, target_variable].item()
            
            # 예측 변화
            importance.append(abs(baseline_pred - masked_pred))
        
        importance = np.array(importance)
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return importance
    
    # ================================================================
    # Visualization
    # ================================================================
    
    def visualize_grad_cam(
        self,
        saliency_maps: np.ndarray,
        original_images: torch.Tensor,
        channel_idx: int = 0,
        time_steps: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Grad-CAM 결과를 원본 이미지에 오버레이하여 시각화
        
        Args:
            saliency_maps: (seq_len, H, W)
            original_images: (batch, channels, seq_len, H, W)
            channel_idx: 표시할 채널
            time_steps: 표시할 시간 (None이면 일부 선택)
            save_path: 저장 경로
        """
        seq_len = saliency_maps.shape[0]
        
        if time_steps is None:
            # 처음, 중간, 마지막
            time_steps = [0, seq_len // 2, seq_len - 1]
        
        num_steps = len(time_steps)
        fig, axes = plt.subplots(3, num_steps, figsize=(4*num_steps, 10))
        
        if num_steps == 1:
            axes = axes[:, np.newaxis]
        
        for idx, t in enumerate(time_steps):
            # 원본 이미지
            orig_img = original_images[0, channel_idx, t].detach().cpu().numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            
            # Saliency map
            sal_map = saliency_maps[t]
            
            # Resize saliency to match image
            if sal_map.shape != orig_img.shape:
                sal_map = cv2.resize(sal_map, (orig_img.shape[1], orig_img.shape[0]))
            
            # Heatmap 생성
            heatmap = cv2.applyColorMap(np.uint8(255 * sal_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            # 1. 원본 이미지
            axes[0, idx].imshow(orig_img, cmap='gray')
            axes[0, idx].set_title(f'Original (t={t})')
            axes[0, idx].axis('off')
            
            # 2. Saliency map
            im = axes[1, idx].imshow(sal_map, cmap='hot')
            axes[1, idx].set_title(f'Saliency Map (t={t})')
            axes[1, idx].axis('off')
            plt.colorbar(im, ax=axes[1, idx], fraction=0.046)
            
            # 3. 오버레이
            overlay = orig_img[..., np.newaxis] * 0.6 + heatmap * 0.4
            axes[2, idx].imshow(overlay)
            axes[2, idx].set_title(f'Overlay (t={t})')
            axes[2, idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_temporal_importance(
        self,
        importance: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        시간별 중요도 시각화
        
        Args:
            importance: (seq_len,)
            save_path: 저장 경로
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.bar(range(len(importance)), importance, color='steelblue', alpha=0.7)
        ax.plot(range(len(importance)), importance, 'r-', linewidth=2, marker='o')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title('Temporal Importance of Each Time Step', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 가장 중요한 시점 표시
        max_idx = np.argmax(importance)
        ax.axvline(max_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(max_idx, importance[max_idx], f'  Most Important\n  t={max_idx}',
                verticalalignment='bottom', fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_comprehensive_saliency_map(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        channel_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        모든 saliency 방법을 종합하여 한 번에 시각화
        
        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: 예측 시점
            target_variable: 예측 변수
            channel_idx: 표시할 채널
            save_path: 저장 경로
        """
        print("Computing saliency maps...")
        
        # 1. Grad-CAM
        print("  [1/3] Grad-CAM...")
        grad_cam_maps = self.grad_cam(
            solar_wind_input, image_input,
            target_index, target_variable
        )
        
        # 2. Temporal Importance
        print("  [2/3] Temporal Importance...")
        temporal_imp = self.temporal_importance(
            solar_wind_input, image_input,
            target_index, target_variable
        )
        
        # 3. Integrated Gradients (간단 버전)
        print("  [3/3] Integrated Gradients...")
        ig_maps = self.integrated_gradients(
            solar_wind_input, image_input,
            target_index, target_variable,
            n_steps=20  # 빠르게
        )
        
        # 시각화
        seq_len = grad_cam_maps.shape[0]
        time_steps = [0, seq_len // 2, seq_len - 1]
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Grad-CAM
        for idx, t in enumerate(time_steps):
            ax = fig.add_subplot(gs[0, idx])
            orig_img = image_input[0, channel_idx, t].detach().cpu().numpy()
            sal_map = grad_cam_maps[t]
            
            # Resize
            if sal_map.shape != orig_img.shape:
                sal_map = cv2.resize(sal_map, (orig_img.shape[1], orig_img.shape[0]))
            
            # Overlay
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * sal_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = orig_img[..., np.newaxis] * 0.6 + heatmap * 0.4
            
            ax.imshow(overlay)
            ax.set_title(f'Grad-CAM (t={t})', fontsize=12)
            ax.axis('off')
        
        # Row 1, Col 4: Temporal Importance
        ax = fig.add_subplot(gs[0, 3])
        ax.bar(range(len(temporal_imp)), temporal_imp, color='steelblue', alpha=0.7)
        ax.plot(range(len(temporal_imp)), temporal_imp, 'r-', linewidth=2)
        ax.set_title('Temporal Importance', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Importance')
        ax.grid(True, alpha=0.3)
        
        # Row 2: Integrated Gradients (spatial avg)
        for idx, t in enumerate(time_steps):
            ax = fig.add_subplot(gs[1, idx])
            orig_img = image_input[0, channel_idx, t].detach().cpu().numpy()
            
            # IG map: channel별 평균
            ig_map = np.abs(ig_maps[:, t, :, :]).mean(axis=0)
            ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
            
            # Overlay
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * ig_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = orig_img[..., np.newaxis] * 0.6 + heatmap * 0.4
            
            ax.imshow(overlay)
            ax.set_title(f'Integrated Gradients (t={t})', fontsize=12)
            ax.axis('off')
        
        # Row 2, Col 4: Channel Importance
        ax = fig.add_subplot(gs[1, 3])
        channel_imp = np.abs(ig_maps).sum(axis=(1, 2, 3))
        channel_imp = channel_imp / channel_imp.max()
        channel_names = ['193Å', '211Å', '304Å'][:image_input.shape[1]]
        ax.bar(channel_names, channel_imp, color=['red', 'green', 'blue'][:len(channel_names)])
        ax.set_title('Channel Importance', fontsize=12)
        ax.set_ylabel('Importance')
        ax.grid(True, alpha=0.3)
        
        # Row 3: Original images
        for idx, t in enumerate(time_steps):
            ax = fig.add_subplot(gs[2, idx])
            orig_img = image_input[0, channel_idx, t].detach().cpu().numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            ax.imshow(orig_img, cmap='gray')
            ax.set_title(f'Original (t={t})', fontsize=12)
            ax.axis('off')
        
        # Row 3, Col 4: Prediction info
        ax = fig.add_subplot(gs[2, 3])
        ax.axis('off')
        
        with torch.no_grad():
            output = self.model(solar_wind_input.to(self.device), image_input.to(self.device))
            pred = output[0, target_index, target_variable].item()
        
        info_text = f"""
        Target Prediction:
        - Time Index: {target_index}
        - Variable: {target_variable}
        - Value: {pred:.2f}
        
        Most Important:
        - Time Step: {np.argmax(temporal_imp)}
        - Channel: {channel_names[np.argmax(channel_imp)]}
        """
        ax.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Comprehensive Saliency Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {save_path}")
        else:
            plt.show()
        
        plt.close()


    def visualize_full_sequence_analysis(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        channel_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        입력 시퀀스 전체를 한 눈에 볼 수 있는 종합 분석 플롯
        
        전체 시간 스텝에 대해:
        1. 원본 이미지 시퀀스 (작게)
        2. Grad-CAM 시퀀스
        3. Integrated Gradients 시퀀스 (NEW!)
        4. Temporal importance curve
        5. Prediction output
        
        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: 예측 시점
            target_variable: 예측 변수
            channel_idx: 표시할 채널
            save_path: 저장 경로
        """
        print("\nGenerating full sequence analysis...")
        
        # 1. Grad-CAM 계산
        print("  Computing Grad-CAM for all time steps...")
        grad_cam_maps = self.grad_cam(
            solar_wind_input, image_input,
            target_index, target_variable
        )
        
        # 2. Integrated Gradients 계산
        print("  Computing Integrated Gradients...")
        ig_maps = self.integrated_gradients(
            solar_wind_input, image_input,
            target_index, target_variable,
            n_steps=50
        )
        
        # 3. Temporal importance 계산
        print("  Computing temporal importance...")
        temporal_imp = self.temporal_importance(
            solar_wind_input, image_input,
            target_index, target_variable
        )
        
        # 4. 예측값 계산
        with torch.no_grad():
            output = self.model(
                solar_wind_input.to(self.device), 
                image_input.to(self.device)
            )
            predictions = output[0, :, target_variable].cpu().numpy()
        
        # 시퀀스 길이
        seq_len = grad_cam_maps.shape[0]
        
        # Figure 생성 (5 rows)
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 1, height_ratios=[2, 2, 2, 1.5, 1], hspace=0.3)
        
        # ================================================================
        # Row 1: 원본 이미지 시퀀스
        # ================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Original SDO Image Sequence', fontsize=14, fontweight='bold', pad=10)
        
        # 모든 시간 스텝의 이미지를 나란히 배치
        image_sequence = []
        for t in range(seq_len):
            img = image_input[0, channel_idx, t].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            image_sequence.append(img)
        
        # 이미지들을 가로로 연결
        concat_images = np.concatenate(image_sequence, axis=1)
        ax1.imshow(concat_images, cmap='gray', aspect='auto')
        ax1.set_ylabel('Spatial\nDimension', fontsize=10)
        ax1.set_xlabel('')
        
        # X축: 시간 스텝 표시
        img_width = image_sequence[0].shape[1]
        tick_positions = [img_width * t + img_width // 2 for t in range(seq_len)]
        ax1.set_xticks(tick_positions[::max(1, seq_len // 10)])  # 10개만 표시
        ax1.set_xticklabels([f't={t}' for t in range(seq_len)][::max(1, seq_len // 10)])
        ax1.tick_params(axis='x', labelsize=8)
        
        # 중요한 시간 스텝 강조
        most_important_t = np.argmax(temporal_imp)
        ax1.axvline(x=most_important_t * img_width + img_width // 2, 
                   color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(most_important_t * img_width + img_width // 2, 
                ax1.get_ylim()[0] * 0.95, 
                f'Most Important\nt={most_important_t}',
                ha='center', va='top', color='red', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # ================================================================
        # Row 2: Grad-CAM 시퀀스
        # ================================================================
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title('Grad-CAM Saliency Sequence', fontsize=14, fontweight='bold', pad=10)
        
        # Grad-CAM을 원본 이미지에 오버레이
        saliency_sequence = []
        for t in range(seq_len):
            # 원본 이미지
            orig = image_input[0, channel_idx, t].detach().cpu().numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
            
            # Saliency map
            sal = grad_cam_maps[t]
            if sal.shape != orig.shape:
                sal = cv2.resize(sal, (orig.shape[1], orig.shape[0]))
            
            # Heatmap 생성
            heatmap = cv2.applyColorMap(np.uint8(255 * sal), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            # Overlay
            overlay = orig[..., np.newaxis] * 0.5 + heatmap * 0.5
            saliency_sequence.append(overlay)
        
        # 연결
        concat_saliency = np.concatenate(saliency_sequence, axis=1)
        ax2.imshow(concat_saliency, aspect='auto')
        ax2.set_ylabel('Spatial\nDimension', fontsize=10)
        ax2.set_xlabel('')
        ax2.set_xticks(tick_positions[::max(1, seq_len // 10)])
        ax2.set_xticklabels([f't={t}' for t in range(seq_len)][::max(1, seq_len // 10)])
        ax2.tick_params(axis='x', labelsize=8)
        
        # 중요한 시간 스텝 강조
        ax2.axvline(x=most_important_t * img_width + img_width // 2, 
                   color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap='jet', norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', pad=0.01, fraction=0.02)
        cbar.set_label('Grad-CAM\nIntensity', fontsize=9)
        
        # ================================================================
        # Row 3: Integrated Gradients 시퀀스
        # ================================================================
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.set_title('Integrated Gradients Sequence', fontsize=14, fontweight='bold', pad=10)
        
        # IG를 원본 이미지에 오버레이
        ig_sequence = []
        for t in range(seq_len):
            # 원본 이미지
            orig = image_input[0, channel_idx, t].detach().cpu().numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
            
            # IG map: 해당 채널만 선택 + 절대값
            ig_map = np.abs(ig_maps[channel_idx, t, :, :])
            
            # Normalize
            ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
            
            # Resize if needed
            if ig_map.shape != orig.shape:
                ig_map = cv2.resize(ig_map, (orig.shape[1], orig.shape[0]))
            
            # Heatmap 생성
            heatmap = cv2.applyColorMap(np.uint8(255 * ig_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            # Overlay
            overlay = orig[..., np.newaxis] * 0.5 + heatmap * 0.5
            ig_sequence.append(overlay)
        
        # 연결
        concat_ig = np.concatenate(ig_sequence, axis=1)
        ax3.imshow(concat_ig, aspect='auto')
        ax3.set_ylabel('Spatial\nDimension', fontsize=10)
        ax3.set_xlabel('')
        ax3.set_xticks(tick_positions[::max(1, seq_len // 10)])
        ax3.set_xticklabels([f't={t}' for t in range(seq_len)][::max(1, seq_len // 10)])
        ax3.tick_params(axis='x', labelsize=8)
        
        # 중요한 시간 스텝 강조
        ax3.axvline(x=most_important_t * img_width + img_width // 2, 
                   color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Colorbar
        sm_ig = ScalarMappable(cmap='jet', norm=Normalize(vmin=0, vmax=1))
        sm_ig.set_array([])
        cbar_ig = plt.colorbar(sm_ig, ax=ax3, orientation='vertical', pad=0.01, fraction=0.02)
        cbar_ig.set_label('IG\nIntensity', fontsize=9)
        
        # ================================================================
        # Row 4: Temporal Importance
        # ================================================================
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.set_title('Temporal Importance', fontsize=14, fontweight='bold', pad=10)
        
        # Bar plot
        colors = ['red' if i == most_important_t else 'steelblue' for i in range(seq_len)]
        bars = ax4.bar(range(seq_len), temporal_imp, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Line plot
        ax4.plot(range(seq_len), temporal_imp, 'k-', linewidth=2, marker='o', markersize=4)
        
        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('Importance', fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(-0.5, seq_len - 0.5)
        
        # 최대값 표시
        ax4.axvline(most_important_t, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax4.text(most_important_t, temporal_imp[most_important_t] * 1.05, 
                f'Peak\nt={most_important_t}\n({temporal_imp[most_important_t]:.3f})',
                ha='center', va='bottom', fontsize=9, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # X축 틱 조정
        if seq_len > 20:
            ax4.set_xticks(range(0, seq_len, max(1, seq_len // 10)))
        
        # ================================================================
        # Row 5: Prediction Output
        # ================================================================
        ax5 = fig.add_subplot(gs[4, 0])
        ax5.set_title('Model Predictions', fontsize=14, fontweight='bold', pad=10)
        
        time_axis = range(len(predictions))
        ax5.plot(time_axis, predictions, 'b-', linewidth=2, marker='s', markersize=6, label='Predicted')
        
        # Target 시점 강조
        ax5.axvline(target_index, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax5.plot(target_index, predictions[target_index], 'go', markersize=10, label=f'Target (t={target_index})')
        
        ax5.set_xlabel('Prediction Time Step', fontsize=11)
        ax5.set_ylabel('Prediction Value', fontsize=11)
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.legend(loc='upper right', fontsize=9)
        
        # 예측값 통계 표시
        stats_text = f'Mean: {predictions.mean():.3f}\nStd: {predictions.std():.3f}\nMin: {predictions.min():.3f}\nMax: {predictions.max():.3f}'
        ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
        
        # ================================================================
        # 전체 타이틀
        # ================================================================
        fig.suptitle(
            f'Full Sequence Analysis (Grad-CAM + IG) - Channel {channel_idx} (Target: t={target_index}, var={target_variable})',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 통계 요약 출력
        print("\n" + "="*60)
        print("SEQUENCE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total time steps: {seq_len}")
        print(f"Most important time step: t={most_important_t} (importance: {temporal_imp[most_important_t]:.4f})")
        print(f"\nPrediction statistics:")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std:  {predictions.std():.4f}")
        print(f"  Min:  {predictions.min():.4f}")
        print(f"  Max:  {predictions.max():.4f}")
        print(f"\nGrad-CAM statistics:")
        print(f"  Mean saliency: {grad_cam_maps.mean():.4f}")
        print(f"  Max saliency:  {grad_cam_maps.max():.4f}")
        print(f"\nIntegrated Gradients statistics:")
        print(f"  Mean |attribution|: {np.abs(ig_maps).mean():.4f}")
        print(f"  Max |attribution|:  {np.abs(ig_maps).max():.4f}")
        print("="*60)


    def visualize_all_channels_analysis(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        channel_names: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ):
        """
        모든 채널에 대해 전체 분석을 수행하고 저장
        
        각 채널별로:
        1. Grad-CAM
        2. Full Sequence Analysis
        3. Comprehensive Saliency
        
        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: 예측 시점
            target_variable: 예측 변수
            channel_names: 채널 이름 리스트 (e.g., ['193Å', '211Å', '304Å'])
            output_dir: 저장 디렉토리
        """
        num_channels = image_input.shape[1]
        
        if channel_names is None:
            channel_names = [f'Channel_{i}' for i in range(num_channels)]
        
        if output_dir is None:
            output_dir = Path('.')
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "=" * 70)
        print(f"MULTI-CHANNEL ANALYSIS: {num_channels} channels")
        print("=" * 70)
        
        for ch_idx, ch_name in enumerate(channel_names[:num_channels]):
            print(f"\n{'─' * 70}")
            print(f"Processing Channel {ch_idx}: {ch_name}")
            print('─' * 70)
            
            ch_dir = output_dir / f"channel_{ch_idx}_{ch_name.replace('Å', 'A')}"
            ch_dir.mkdir(exist_ok=True, parents=True)
            
            # 1. Grad-CAM
            print(f"\n[1/4] Grad-CAM for {ch_name}...")
            try:
                grad_cam_maps = self.grad_cam(
                    solar_wind_input, image_input,
                    target_index, target_variable
                )
                
                self.visualize_grad_cam(
                    grad_cam_maps, image_input,
                    channel_idx=ch_idx,
                    save_path=ch_dir / f"grad_cam_{ch_name}.png"
                )
                print(f"  ✓ Saved: grad_cam_{ch_name}.png")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            # 2. Full Sequence Analysis
            print(f"\n[2/4] Full Sequence for {ch_name}...")
            try:
                self.visualize_full_sequence_analysis(
                    solar_wind_input, image_input,
                    target_index, target_variable,
                    channel_idx=ch_idx,
                    save_path=ch_dir / f"full_sequence_{ch_name}.png"
                )
                print(f"  ✓ Saved: full_sequence_{ch_name}.png")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            # 3. Comprehensive Saliency
            print(f"\n[3/4] Comprehensive Saliency for {ch_name}...")
            try:
                self.create_comprehensive_saliency_map(
                    solar_wind_input, image_input,
                    target_index, target_variable,
                    channel_idx=ch_idx,
                    save_path=ch_dir / f"comprehensive_{ch_name}.png"
                )
                print(f"  ✓ Saved: comprehensive_{ch_name}.png")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            # 4. Temporal Importance (채널 공통)
            if ch_idx == 0:  # 첫 채널에서만 계산 (모든 채널 동일)
                print(f"\n[4/4] Temporal Importance (all channels)...")
                try:
                    temporal_imp = self.temporal_importance(
                        solar_wind_input, image_input,
                        target_index, target_variable
                    )
                    
                    self.visualize_temporal_importance(
                        temporal_imp,
                        save_path=output_dir / "temporal_importance_all_channels.png"
                    )
                    print(f"  ✓ Saved: temporal_importance_all_channels.png")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        
        # 채널 비교 플롯 생성
        print(f"\n{'─' * 70}")
        print("Creating Channel Comparison Plot...")
        print('─' * 70)
        
        try:
            self._create_channel_comparison_plot(
                solar_wind_input, image_input,
                target_index, target_variable,
                channel_names, output_dir
            )
            print(f"  ✓ Saved: channel_comparison.png")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print("\n" + "=" * 70)
        print("MULTI-CHANNEL ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nAll results saved to: {output_dir}")
        print(f"\nGenerated {num_channels} channel directories:")
        for ch_idx, ch_name in enumerate(channel_names[:num_channels]):
            ch_dir_name = f"channel_{ch_idx}_{ch_name.replace('Å', 'A')}"
            print(f"  - {ch_dir_name}/")
    
    def _create_channel_comparison_plot(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int,
        target_variable: int,
        channel_names: List[str],
        output_dir: Path
    ):
        """
        채널 간 비교 플롯 생성
        
        각 채널의 대표 시점 이미지와 Grad-CAM을 나란히 표시
        """
        num_channels = image_input.shape[1]
        seq_len = image_input.shape[2]
        
        # Grad-CAM 계산
        grad_cam_maps = self.grad_cam(
            solar_wind_input, image_input,
            target_index, target_variable
        )
        
        # Temporal importance
        temporal_imp = self.temporal_importance(
            solar_wind_input, image_input,
            target_index, target_variable
        )
        most_important_t = np.argmax(temporal_imp)
        
        # 3개 시점 선택: 처음, 가장 중요한 시점, 마지막
        time_points = [0, most_important_t, seq_len - 1]
        
        # Figure 생성
        fig, axes = plt.subplots(
            num_channels, len(time_points) * 2,  # 각 시점마다 원본 + Grad-CAM
            figsize=(4 * len(time_points) * 2, 4 * num_channels)
        )
        
        # 1D array로 변환 (단일 채널 대응)
        if num_channels == 1:
            axes = axes.reshape(1, -1)
        
        for ch_idx in range(num_channels):
            ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'Ch{ch_idx}'
            
            for t_idx, t in enumerate(time_points):
                col_orig = t_idx * 2
                col_sal = t_idx * 2 + 1
                
                # 원본 이미지
                orig_img = image_input[0, ch_idx, t].detach().cpu().numpy()
                orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
                
                axes[ch_idx, col_orig].imshow(orig_img, cmap='gray')
                axes[ch_idx, col_orig].set_title(
                    f'{ch_name} - t={t}\n{"(Most Important)" if t == most_important_t else ""}',
                    fontsize=10, fontweight='bold' if t == most_important_t else 'normal'
                )
                axes[ch_idx, col_orig].axis('off')
                
                # Grad-CAM overlay
                sal_map = grad_cam_maps[t]
                if sal_map.shape != orig_img.shape:
                    sal_map = cv2.resize(sal_map, (orig_img.shape[1], orig_img.shape[0]))
                
                heatmap = cv2.applyColorMap(np.uint8(255 * sal_map), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                overlay = orig_img[..., np.newaxis] * 0.5 + heatmap * 0.5
                
                axes[ch_idx, col_sal].imshow(overlay)
                axes[ch_idx, col_sal].set_title(f'Grad-CAM', fontsize=10)
                axes[ch_idx, col_sal].axis('off')
        
        plt.suptitle(
            f'Channel Comparison - Target: t={target_index}, var={target_variable}',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(output_dir / 'channel_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    print("Saliency Maps Module")
    print("Use this module to extract saliency/attribution maps from trained models")