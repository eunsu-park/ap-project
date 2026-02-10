# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal deep learning system for solar wind and geomagnetic activity prediction. Combines:
- **Transformer encoder** for solar wind time series (OMNI data)
- **ConvLSTM** for SDO image sequences (AIA 193, AIA 211, HMI magnetogram)
- **Cross-modal attention fusion** to combine both modalities

Target variable: AP index (geomagnetic activity indicator)

## Common Commands

### Training
```bash
python train.py --config-name wulver      # Server (Wulver HPC)
python train.py --config-name local       # Local development
```

### Validation
```bash
python validation.py --config-name <config>
```

### Monte Carlo Dropout (uncertainty estimation)
```bash
python monte_carlo_dropout.py --config-name <config>
```

### XAI / Saliency Analysis
```bash
python example_saliency.py --config-name saliency_0
python example_ig_all_frames.py --config-name saliency_0      # Integrated Gradients
python example_attention_all_targets.py --config-name attention_0
```

### Testing individual components
```bash
python networks.py --config-name local    # Test model forward pass
python pipeline.py --config-name local    # Test data loading
```

## Configuration System

Uses **Hydra** for configuration. All configs in `configs/` directory.

Key config sections:
- `experiment`: seed, batch_size, undersampling settings
- `environment`: device, data_root, save_root, num_workers
- `data`: SDO wavelengths, OMNI variables, index ranges
- `model`: Transformer/ConvLSTM/Fusion hyperparameters
- `training`: loss types, optimizer, learning rate
- `validation`/`mcd`: checkpoint_path, output_dir

Override config values via CLI:
```bash
python train.py --config-name local training.num_epochs=1 experiment.batch_size=2
```

## Architecture

### Model (`networks.py`)

세 가지 모델 타입 지원 (`config.model.model_type`):

**1. `fusion` (기본값)** - MultiModalModel
```
MultiModalModel
├── TransformerEncoderModel  # OMNI time series -> (batch, d_model)
├── ConvLSTMModel            # SDO image sequences -> (batch, d_model)
├── CrossModalFusion         # Bidirectional cross-attention
└── regression_head          # -> (batch, target_seq_len, num_targets)
```

**2. `transformer`** - TransformerOnlyModel (OMNI only)
```
TransformerOnlyModel
├── TransformerEncoderModel  # OMNI time series -> (batch, d_model)
└── regression_head          # -> (batch, target_seq_len, num_targets)
```

**3. `convlstm`** - ConvLSTMOnlyModel (SDO only)
```
ConvLSTMOnlyModel
├── ConvLSTMModel            # SDO image sequences -> (batch, d_model)
└── regression_head          # -> (batch, target_seq_len, num_targets)
```

CLI에서 모델 타입 변경:
```bash
python train.py --config-name local model.model_type=transformer
python train.py --config-name local model.model_type=convlstm
```

### Data Pipeline (`pipeline.py`)
- `HDF5Reader`: Reads HDF5 files with SDO images and OMNI data
- `Normalizer`: SDO to [-1,1], OMNI via z-score
- `BaseDataset` → `TrainDataset` / `ValidationDataset`
- Undersampling support for class imbalance

### Data Format
HDF5 files with structure:
- `sdo/aia_193`, `sdo/aia_211`, `sdo/hmi_magnetogram`: Image arrays (T, 1, H, W)
- `omni/<variable>`: Time series arrays

Index conventions (3-hour resolution):
- SDO indices: sdo_start_index to sdo_end_index
- Input indices: input_start_index to input_end_index
- Target indices: target_start_index to target_end_index

## SLURM Job Submission

`run_train.py`, `run_validation.py`, `run_monte_carlo_dropout.py` contain `WulverSubmitter` class for batch job submission.

```python
# Example: Generate and submit training jobs
python run_train.py  # Creates SLURM scripts, use dry_run=False to submit
```

## Key Dependencies

- PyTorch, Hydra
- h5py (data I/O)
- captum (for saliency maps, XAI)

## Known Issues

ConvLSTM temporal information utilization issue documented in `ANALYSIS_AND_IMPROVEMENT_PLAN.md`:
- Grad-CAM patterns too static across time
- Middle timesteps underweighted
- Potential fixes: feature normalization, auxiliary loss, temporal attention
