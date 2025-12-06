"""
Contrastive loss functions for multimodal alignment.

This module provides loss functions for aligning features from different
modalities (transformer and ConvLSTM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalContrastiveLoss(nn.Module):
    """Contrastive loss for aligning multimodal representations.
    
    Uses InfoNCE-style loss to align transformer (solar wind) and ConvLSTM (image)
    features in the same embedding space. Features from the same sample are treated
    as positive pairs, while features from different samples in the batch serve as
    negative pairs (in-batch negatives).
    
    Args:
        temperature: Temperature parameter for scaling similarity scores.
                    Higher values (0.3-0.5) recommended for small batch sizes.
                    Lower values (0.07-0.1) for large batch sizes.
        normalize: Whether to L2-normalize features before computing similarity.
                  True is strongly recommended for stable training.
    """
    
    def __init__(self, temperature: float = 0.3, normalize: bool = True):
        super().__init__()
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two sets of features.
        
        Args:
            features1: Features from first modality of shape (batch_size, feature_dim).
            features2: Features from second modality of shape (batch_size, feature_dim).
            
        Returns:
            Contrastive loss scalar.
            
        Raises:
            ValueError: If input tensors have mismatched dimensions.
        """
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")
        
        if features1.size(0) != features2.size(0):
            raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")
        
        if features1.size(1) != features2.size(1):
            raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")
        
        batch_size = features1.size(0)
        
        # L2 normalization for cosine similarity
        if self.normalize:
            features1 = F.normalize(features1, p=2, dim=1)
            features2 = F.normalize(features2, p=2, dim=1)
        
        # Concatenate features from both modalities
        # Shape: (2 * batch_size, feature_dim)
        features = torch.cat([features1, features2], dim=0)
        
        # Compute similarity matrix
        # Shape: (2 * batch_size, 2 * batch_size)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for positive pairs
        # For each sample i in features1, its positive pair is sample i in features2
        # which is at index (i + batch_size) in the concatenated tensor
        labels = torch.arange(batch_size, device=features1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Create mask to exclude self-similarity (diagonal elements)
        # We don't want to compare a sample with itself
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Compute cross-entropy loss
        # This encourages high similarity with positive pairs and low similarity with negatives
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class MultiModalMSELoss(nn.Module):
    """MSE-based consistency loss for multimodal alignment.
    
    Directly minimizes the Euclidean distance between features from two modalities.
    This approach encourages the same sample's features from different modalities
    to have similar representations in the embedding space.
    
    Unlike contrastive learning approaches (InfoNCE), MSE loss does not use negative
    samples, making it more suitable for small batch sizes where negative samples
    would be limited.
    
    Args:
        reduction: Specifies the reduction to apply to the output.
                  'mean' (default): average of all elements
                  'sum': sum of all elements
                  'none': no reduction
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'")
        
        self.reduction = reduction
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between two sets of features.
        
        Args:
            features1: Features from first modality of shape (batch_size, feature_dim).
            features2: Features from second modality of shape (batch_size, feature_dim).
            
        Returns:
            MSE loss scalar (or tensor if reduction='none').
            
        Raises:
            ValueError: If input tensors have mismatched dimensions.
        """
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")
        
        if features1.size(0) != features2.size(0):
            raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")
        
        if features1.size(1) != features2.size(1):
            raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")
        
        # Compute MSE loss - directly aligns features from both modalities
        loss = F.mse_loss(features1, features2, reduction=self.reduction)
        
        return loss
