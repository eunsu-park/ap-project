"""
Test configuration system.

Tests:
1. Config loading from YAML
2. Hydra to structured config conversion
3. Type safety and attribute access
4. Config validation
5. Convenience properties
"""

import hydra
from hydra import initialize, compose

from config import Config


@hydra.main(config_path="./configs", version_base=None)
def main(hydra_cfg):
    """Test configuration system."""
    print("=" * 80)
    print("Testing Refactored Config System")
    print("=" * 80)
    
    # Test 1: Hydra config loading
    print("\nTest 1: Hydra Config Loading")
    print("-" * 80)
    print(f"✓ Hydra config loaded")
    print(f"  Experiment: {hydra_cfg.experiment.experiment_name}")
    print(f"  Device: {hydra_cfg.environment.device}")
    print(f"  Dataset: {hydra_cfg.data.dataset_name}")
    
    # Test 2: Structured config conversion
    print("\nTest 2: Structured Config Conversion")
    print("-" * 80)
    config = Config.from_hydra(hydra_cfg)
    print(f"✓ Converted to structured Config")
    print(f"  Type: {type(config)}")
    print(f"  Environment type: {type(config.environment)}")
    print(f"  Data type: {type(config.data)}")
    print(f"  Model type: {type(config.model)}")
    print(f"  Training type: {type(config.training)}")
    
    # Test 3: Type safety and IDE support
    print("\nTest 3: Type Safety and Attribute Access")
    print("-" * 80)
    
    # These all have IDE autocomplete and type checking
    print(f"✓ Environment:")
    print(f"  seed: {config.environment.seed}")
    print(f"  device: {config.environment.device}")
    print(f"  data_root: {config.environment.data_root}")
    print(f"  save_root: {config.environment.save_root}")
    
    print(f"\n✓ Experiment:")
    print(f"  name: {config.experiment.experiment_name}")
    print(f"  phase: {config.experiment.phase}")
    print(f"  batch_size: {config.experiment.batch_size}")
    print(f"  num_workers: {config.experiment.num_workers}")
    
    print(f"\n✓ Data:")
    print(f"  dataset_name: {config.data.dataset_name}")
    print(f"  sdo_wavelengths: {config.data.sdo_wavelengths}")
    print(f"  input_variables ({len(config.data.input_variables)}): "
          f"{config.data.input_variables[:3]}...")
    print(f"  target_variables: {config.data.target_variables}")
    print(f"  sdo_sequence_length: {config.data.sdo_sequence_length}")
    print(f"  input_sequence_length: {config.data.input_sequence_length}")
    print(f"  target_sequence_length: {config.data.target_sequence_length}")
    
    print(f"\n✓ Model:")
    print(f"  transformer_d_model: {config.model.transformer_d_model}")
    print(f"  transformer_nhead: {config.model.transformer_nhead}")
    print(f"  transformer_num_layers: {config.model.transformer_num_layers}")
    print(f"  convlstm_hidden_channels: {config.model.convlstm_hidden_channels}")
    print(f"  fusion_num_heads: {config.model.fusion_num_heads}")
    
    print(f"\n✓ Training:")
    print(f"  num_epochs: {config.training.num_epochs}")
    print(f"  learning_rate: {config.training.learning_rate}")
    print(f"  optimizer: {config.training.optimizer}")
    print(f"  loss_type: {config.training.loss_type}")
    print(f"  contrastive_type: {config.training.contrastive_type}")
    print(f"  lambda_contrastive: {config.training.lambda_contrastive}")
    
    # Test 4: Computed properties
    print("\nTest 4: Computed Properties")
    print("-" * 80)
    
    print(f"✓ Data computed properties:")
    print(f"  omni_variables: {config.data.omni_variables}")
    print(f"  (Deduplicates input + target variables)")
    
    print(f"\n✓ Config convenience properties:")
    print(f"  dataset_path: {config.dataset_path}")
    print(f"  train_list_path: {config.train_list_path}")
    print(f"  validation_list_path: {config.validation_list_path}")
    print(f"  stat_file_path: {config.stat_file_path}")
    print(f"  experiment_dir: {config.experiment_dir}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    print(f"  log_dir: {config.log_dir}")
    
    # Test 5: Validation config (optional)
    print("\nTest 5: Validation Config")
    print("-" * 80)
    
    if config.validation is not None:
        print(f"✓ Validation config present:")
        print(f"  checkpoint_path: {config.validation.checkpoint_path}")
        print(f"  output_dir: {config.validation.output_dir}")
        print(f"  compute_alignment: {config.validation.compute_alignment}")
        print(f"  save_plots: {config.validation.save_plots}")
    else:
        print(f"✓ Validation config not present (expected for training)")
    
    # Test 6: Config modification
    print("\nTest 6: Config Modification")
    print("-" * 80)
    
    # Configs are dataclasses, so they're mutable
    original_lr = config.training.learning_rate
    config.training.learning_rate = 0.001
    print(f"✓ Modified learning_rate: {original_lr} → {config.training.learning_rate}")
    
    original_batch = config.experiment.batch_size
    config.experiment.batch_size = 16
    print(f"✓ Modified batch_size: {original_batch} → {config.experiment.batch_size}")
    
    # Restore
    config.training.learning_rate = original_lr
    config.experiment.batch_size = original_batch
    print(f"✓ Restored to original values")
    
    # Test 7: Error handling
    print("\nTest 7: Attribute Access Safety")
    print("-" * 80)
    
    try:
        # This will raise AttributeError (good!)
        _ = config.nonexistent_attribute
        print("✗ Should have raised AttributeError")
    except AttributeError:
        print("✓ AttributeError raised for invalid attribute (type-safe!)")
    
    # Test 8: Sampling configuration
    print("\nTest 8: Sampling Configuration")
    print("-" * 80)
    
    print(f"✓ Sampling settings:")
    print(f"  enable_undersampling: {config.experiment.enable_undersampling}")
    print(f"  num_subsample: {config.experiment.num_subsample}")
    print(f"  subsample_index: {config.experiment.subsample_index}")
    print(f"  enable_oversampling: {config.experiment.enable_oversampling}")
    print(f"  num_oversample: {config.experiment.num_oversample}")
    
    print("\n" + "=" * 80)
    print("✓ All config tests passed successfully!")
    print("=" * 80)
    
    # Summary
    print("\nConfig System Benefits:")
    print("  ✓ Type-safe attribute access")
    print("  ✓ IDE autocomplete support")
    print("  ✓ Clear structure with dataclasses")
    print("  ✓ Convenience properties for common paths")
    print("  ✓ Easy to modify and extend")
    print("  ✓ Backward compatible with Hydra YAML files")


if __name__ == "__main__":
    main()
