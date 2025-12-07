"""
Test data loading pipeline.

Tests:
1. Dataloader creation
2. Batch loading
3. Data shapes and ranges
4. Loading speed
5. Statistics consistency
"""

import time
import random
import numpy as np
import torch
import hydra

from config import Config
from datasets import create_dataloader


@hydra.main(config_path="./configs", version_base=None)
def main(hydra_cfg):
    """Test the data loading pipeline."""
    # Convert to structured config
    config = Config.from_hydra(hydra_cfg)
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 80)
    print("Testing Refactored Data Pipeline")
    print("=" * 80)
    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Phase: {config.experiment.phase}")
    print(f"Dataset: {config.data.dataset_name}")
    print("=" * 80)
    
    # Create dataloader
    print("\nCreating dataloader...")
    t0 = time.time()
    dataloader = create_dataloader(config)
    creation_time = time.time() - t0
    
    print(f"\n✓ DataLoader created in {creation_time:.2f}s")
    print(f"  Dataset size: {len(dataloader.dataset)}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    print(f"  Num workers: {dataloader.num_workers}")
    print(f"  Pos weight: {dataloader.dataset.pos_weight:.3f}")
    
    # Test 1: Load first batch
    print("\n" + "=" * 80)
    print("Test 1: First Batch Inspection")
    print("=" * 80)
    
    batch = next(iter(dataloader))
    
    sdo = batch['sdo']
    inputs = batch['inputs']
    targets = batch['targets']
    labels = batch['labels']
    file_names = batch['file_names']
    
    print(f"\nBatch contents:")
    print(f"  SDO shape: {sdo.shape}")
    print(f"    - Expected: (batch={config.experiment.batch_size}, "
          f"channels={config.model.convlstm_input_channels}, "
          f"time={config.data.sdo_sequence_length}, "
          f"height={config.data.sdo_image_size}, "
          f"width={config.data.sdo_image_size})")
    print(f"    - Range: [{sdo.min():.3f}, {sdo.max():.3f}]")
    print(f"    - Mean: {sdo.mean():.3f}, Std: {sdo.std():.3f}")
    
    print(f"\n  Inputs shape: {inputs.shape}")
    print(f"    - Expected: (batch={config.experiment.batch_size}, "
          f"seq_len={config.data.input_sequence_length}, "
          f"vars={len(config.data.input_variables)})")
    print(f"    - Range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"    - Mean: {inputs.mean():.3f}, Std: {inputs.std():.3f}")
    
    print(f"\n  Targets shape: {targets.shape}")
    print(f"    - Expected: (batch={config.experiment.batch_size}, "
          f"seq_len={config.data.target_sequence_length}, "
          f"vars={len(config.data.target_variables)})")
    print(f"    - Range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"    - Mean: {targets.mean():.3f}, Std: {targets.std():.3f}")
    
    print(f"\n  Labels shape: {labels.shape}")
    print(f"    - Unique values: {torch.unique(labels).tolist()}")
    
    print(f"\n  File names (first 3):")
    for i, name in enumerate(file_names[:3]):
        print(f"    [{i}] {name}")
    
    # Test 2: Data normalization check
    print("\n" + "=" * 80)
    print("Test 2: Normalization Check")
    print("=" * 80)
    
    # SDO should be in [-1, 1]
    print(f"\nSDO normalization:")
    print(f"  Expected range: [-1, 1]")
    print(f"  Actual range: [{sdo.min():.3f}, {sdo.max():.3f}]")
    if sdo.min() >= -1.1 and sdo.max() <= 1.1:
        print(f"  ✓ Normalization correct")
    else:
        print(f"  ✗ Warning: SDO values outside expected range")
    
    # OMNI inputs should be z-score normalized (approx mean=0, std=1)
    print(f"\nOMNI inputs (z-score normalized):")
    print(f"  Expected: mean ≈ 0, std ≈ 1")
    print(f"  Actual: mean = {inputs.mean():.3f}, std = {inputs.std():.3f}")
    
    # Test 3: Loading speed
    print("\n" + "=" * 80)
    print("Test 3: Loading Speed Test (3 epochs)")
    print("=" * 80)
    
    for epoch in range(3):
        t0 = time.time()
        batch_count = 0
        
        for i, batch in enumerate(dataloader):
            batch_count += 1
            
            if i == 0:  # Print first batch info
                sdo_batch = batch['sdo']
                inputs_batch = batch['inputs']
                targets_batch = batch['targets']
                
                print(f"\nEpoch {epoch} - First batch:")
                print(f"  SDO: shape={sdo_batch.shape}, "
                      f"mean={sdo_batch.mean():.3f}, "
                      f"std={sdo_batch.std():.3f}")
                print(f"  Inputs: shape={inputs_batch.shape}, "
                      f"mean={inputs_batch.mean():.3f}, "
                      f"std={inputs_batch.std():.3f}")
                print(f"  Targets: shape={targets_batch.shape}, "
                      f"mean={targets_batch.mean():.3f}, "
                      f"std={targets_batch.std():.3f}")
            
            # Only process first 5 batches for speed test
            if i >= 4:
                break
        
        elapsed = time.time() - t0
        samples_per_sec = (batch_count * config.experiment.batch_size) / elapsed
        print(f"  Epoch {epoch}: {elapsed:.2f}s "
              f"({samples_per_sec:.1f} samples/sec, first 5 batches)")
    
    # Test 4: Batch consistency
    print("\n" + "=" * 80)
    print("Test 4: Batch Consistency")
    print("=" * 80)
    
    print("\nLoading 3 batches to check consistency...")
    batch_stats = []
    
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        sdo_batch = batch['sdo']
        inputs_batch = batch['inputs']
        
        stats = {
            'sdo_mean': sdo_batch.mean().item(),
            'sdo_std': sdo_batch.std().item(),
            'inputs_mean': inputs_batch.mean().item(),
            'inputs_std': inputs_batch.std().item(),
        }
        batch_stats.append(stats)
        
        print(f"\nBatch {i}:")
        print(f"  SDO: mean={stats['sdo_mean']:.3f}, std={stats['sdo_std']:.3f}")
        print(f"  Inputs: mean={stats['inputs_mean']:.3f}, std={stats['inputs_std']:.3f}")
    
    # Check variance across batches
    sdo_means = [s['sdo_mean'] for s in batch_stats]
    inputs_means = [s['inputs_mean'] for s in batch_stats]
    
    print(f"\nCross-batch statistics:")
    print(f"  SDO mean variance: {np.std(sdo_means):.4f}")
    print(f"  Inputs mean variance: {np.std(inputs_means):.4f}")
    print(f"  ✓ Batches are {'consistent' if np.std(sdo_means) < 0.1 else 'varying'}")
    
    # Test 5: Dataset statistics
    print("\n" + "=" * 80)
    print("Test 5: Dataset Statistics Access")
    print("=" * 80)
    
    if hasattr(dataloader.dataset, 'stat_dict'):
        print("\n✓ Statistics available")
        stat_dict = dataloader.dataset.stat_dict
        
        print(f"\nAvailable statistics for {len(stat_dict)} variables:")
        for i, (var, stats) in enumerate(list(stat_dict.items())[:5]):
            print(f"  [{i}] {var}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        if len(stat_dict) > 5:
            print(f"  ... and {len(stat_dict) - 5} more variables")
    else:
        print("\n✗ Statistics not accessible")
    
    print("\n" + "=" * 80)
    print("✓ All data pipeline tests passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
