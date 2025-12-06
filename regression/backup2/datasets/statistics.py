"""
Statistics computation and normalization for multimodal dataset.

This module handles:
- Data normalization (SDO and OMNI)
- Online statistics computation (Welford's algorithm)
- Statistics caching
"""

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import h5py


# ============================================================================
# Normalization
# ============================================================================

class Normalizer:
    """Handles data normalization for SDO and OMNI data."""
    
    def __init__(self, stats: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Args:
            stats: Dictionary of statistics {variable: {'mean': x, 'std': y}}
        """
        self.stats = stats or {}
    
    def normalize_sdo(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize SDO data from [0, 255] to [-1, 1].
        
        Args:
            data: Raw SDO data in range [0, 255]
            
        Returns:
            Normalized data in range [-1, 1]
        """
        return (data * (2.0 / 255.0)) - 1.0
    
    def normalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        """
        Z-score normalization for OMNI data.
        
        Args:
            data: Raw OMNI data
            variable: Variable name for statistics lookup
            
        Returns:
            Normalized data (z-score)
        """
        if variable not in self.stats:
            raise KeyError(f"Statistics not found for variable: {variable}")
        
        mean = self.stats[variable]['mean']
        std = self.stats[variable]['std']
        
        return (data - mean) / std
    
    def denormalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Reverse z-score normalization."""
        if variable not in self.stats:
            raise KeyError(f"Statistics not found for variable: {variable}")
        
        mean = self.stats[variable]['mean']
        std = self.stats[variable]['std']
        
        return data * std + mean


# ============================================================================
# Statistics Computation
# ============================================================================

class OnlineStatistics:
    """
    Compute mean and std using Welford's online algorithm.
    Memory efficient - O(1) space complexity.
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def update(self, batch: np.ndarray):
        """
        Update statistics with a new batch of data.
        
        Args:
            batch: Data array of any shape
        """
        # Flatten and filter finite values
        values = batch.flatten()
        valid_values = values[np.isfinite(values)]
        
        for x in valid_values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
    
    @property
    def std(self) -> float:
        """Return standard deviation."""
        if self.n < 2:
            return 1.0  # Fallback for insufficient data
        return float(np.sqrt(self.M2 / self.n))
    
    def get_stats(self) -> Dict[str, float]:
        """Return statistics as dictionary."""
        return {
            'mean': float(self.mean),
            'std': self.std
        }


def compute_statistics(
    data_root: str,
    dataset_dir_name: str,
    data_file_list: List[str],
    variables: List[str],
    stat_file_path: str,
    overwrite: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Compute and cache statistics for OMNI variables.
    
    Args:
        data_root: Root directory containing data files
        data_file_list: List of data file names
        variables: List of OMNI variable names
        stat_file_path: Path to save/load statistics pickle file
        overwrite: Whether to recompute even if cache exists
        
    Returns:
        Dictionary of statistics {variable: {'mean': x, 'std': y}}
    """
    # Load existing statistics if available
    if os.path.exists(stat_file_path) and not overwrite:
        try:
            with open(stat_file_path, 'rb') as f:
                loaded_stats = pickle.load(f)
            
            # Verify all variables are present
            if all(var in loaded_stats for var in variables):
                print(f"Loaded statistics from {stat_file_path}")
                return {var: loaded_stats[var] for var in variables}
            else:
                print("Incomplete statistics, recomputing...")
        except (pickle.PickleError, KeyError) as e:
            print(f"Failed to load statistics: {e}, recomputing...")
    
    # Filter for h5 files only
    h5_files = [f"{data_root}/{dataset_dir_name}/{f}" for f in data_file_list if f.endswith('.h5')]
    
    if not h5_files:
        raise ValueError("No valid .h5 files found in data file list")
    
    # Initialize online statistics for each variable
    stats_computers = {var: OnlineStatistics() for var in variables}
    
    # Process files
    valid_files = 0
    for file_path in h5_files:
        if not os.path.exists(file_path):
            continue
        
        try:
            with h5py.File(file_path, 'r') as f:
                for variable in variables:
                    dataset_name = f"omni_{variable}"
                    if dataset_name in f:
                        data = f[dataset_name][:]
                        stats_computers[variable].update(data)
            
            valid_files += 1
            
            if valid_files % 100 == 0:
                print(f"Processed {valid_files}/{len(h5_files)} files...")
                
        except (OSError, KeyError) as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            continue
    
    if valid_files == 0:
        raise ValueError("No valid data files found for statistics computation")
    
    # Compile final statistics
    stat_dict = {}
    for variable in variables:
        stat_dict[variable] = stats_computers[variable].get_stats()
        print(f"{variable}: mean={stat_dict[variable]['mean']:.3f}, "
              f"std={stat_dict[variable]['std']:.3f}")
    
    # Save statistics
    try:
        os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
        with open(stat_file_path, 'wb') as f:
            pickle.dump(stat_dict, f)
        print(f"Statistics saved to {stat_file_path}")
    except (OSError, pickle.PickleError) as e:
        print(f"Warning: Failed to save statistics: {e}")
    
    return stat_dict
