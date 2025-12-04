import time
import random
import numpy as np
import torch
import hydra
from data import create_dataloader

@hydra.main(config_path="./configs", version_base=None)
def main(config):
    """Test the pipeline."""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("="*80)
    print("Testing Refactored Pipeline")
    print("="*80)
    
    # Create dataloader
    dataloader = create_dataloader(config)
    
    print(f"\nDataLoader created:")
    print(f"  Dataset size: {len(dataloader.dataset)}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    print(f"  Pos weight: {dataloader.dataset.pos_weight:.3f}")
    
    # Test loading speed
    print(f"\nTesting loading speed (3 epochs):")
    for epoch in range(3):
        t0 = time.time()
        for i, batch in enumerate(dataloader):
            sdo = batch['sdo'].numpy()
            inputs = batch['inputs'].numpy()
            targets = batch['targets'].numpy()
            labels = batch['labels'].numpy()
            
            if i == 0:  # Print first batch info
                print(f"\nEpoch {epoch} - First batch:")
                print(f"  SDO shape: {sdo.shape}, mean: {sdo.mean():.3f}, std: {sdo.std():.3f}")
                print(f"  Inputs shape: {inputs.shape}, mean: {inputs.mean():.3f}, std: {inputs.std():.3f}")
                print(f"  Targets shape: {targets.shape}, mean: {targets.mean():.3f}, std: {targets.std():.3f}")
                print(f"  Labels shape: {labels.shape}, mean: {labels.mean():.3f}")
            
            # Only process a few batches for speed test
            if i >= 2:
                break
        
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}: {elapsed:.2f}s (first 3 batches)")
    
    print("\n" + "="*80)
    print("Pipeline test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()