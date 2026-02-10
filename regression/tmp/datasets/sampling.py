"""
Sampling strategies for handling class imbalance.

This module provides undersampling and oversampling strategies
for imbalanced datasets.
"""

from typing import List, Tuple


class SamplingStrategy:
    """
    Handle dataset sampling strategies (undersampling, oversampling).
    Uses index-based approach instead of file name manipulation.
    """
    
    @staticmethod
    def split_by_class(file_list: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Split file list into positive and negative samples.
        
        Returns:
            (positive_samples, negative_samples)
        """
        positive = []
        negative = []
        
        for file_name, label in file_list:
            if label == 0:
                negative.append((file_name, label))
            else:
                positive.append((file_name, label))
        
        # Sort for reproducibility (independent of input order)
        positive.sort(key=lambda x: x[0])
        negative.sort(key=lambda x: x[0])
        
        return positive, negative
    
    @staticmethod
    def undersample(
        file_list: List[Tuple[str, int]],
        num_folds: int,
        fold_index: int
    ) -> Tuple[List[Tuple[str, int]], int, int]:
        """
        Undersample negative class by splitting into folds.
        
        Args:
            file_list: List of (file_name, label) tuples
            num_folds: Number of folds to split negative samples
            fold_index: Which fold to use (0 to num_folds-1)
            
        Returns:
            (sampled_file_list, num_positive, num_negative_total)
            
        Note:
            Uses global random state (should be set by train.py)
        """
        positive, negative = SamplingStrategy.split_by_class(file_list)
        
        # Shuffle negative samples using global random state
        import random
        random.shuffle(negative)
        
        # Split into folds
        n = len(negative)
        base_size = n // num_folds
        remainder = n % num_folds
        
        start = 0
        for i in range(num_folds):
            size = base_size + (1 if i < remainder else 0)
            if i == fold_index:
                selected_negative = negative[start:start + size]
                break
            start += size
        
        # Combine positive and selected negative
        sampled_list = positive + selected_negative
        
        return sampled_list, len(positive), len(negative)
    
    @staticmethod
    def oversample(
        file_list: List[Tuple[str, int]],
        oversample_factor: int
    ) -> Tuple[List[int], List[Tuple[str, int]], int, int]:
        """
        Oversample positive class by creating index mapping.
        
        Args:
            file_list: List of (file_name, label) tuples
            oversample_factor: How many times to replicate positive samples
            
        Returns:
            (indices, file_list, num_positive, num_negative)
            indices: List of indices into file_list (with repetitions)
        """
        positive, negative = SamplingStrategy.split_by_class(file_list)
        
        # Build index mapping
        indices = []
        
        # Add negative indices (once each)
        for i, (file_name, label) in enumerate(file_list):
            if label == 0:
                indices.append(i)
        
        # Add positive indices (multiple times)
        for i, (file_name, label) in enumerate(file_list):
            if label == 1:
                for _ in range(oversample_factor):
                    indices.append(i)
        
        return indices, file_list, len(positive), len(negative)
    
    @staticmethod
    def get_pos_weight(file_list: List[Tuple[str, int]]) -> float:
        """
        Calculate positive class weight for BCEWithLogitsLoss.
        
        Returns:
            Weight = num_negative / num_positive
        """
        positive, negative = SamplingStrategy.split_by_class(file_list)
        
        if len(positive) == 0:
            raise ValueError("No positive samples found")
        
        return len(negative) / len(positive)
