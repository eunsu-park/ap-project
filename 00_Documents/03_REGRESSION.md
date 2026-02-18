# 03_Regression Module Documentation

Multi-modal deep learning system for geomagnetic Ap Index prediction. Combines temporal data (OMNI solar wind parameters) with spatial data (SDO images) using cross-modal attention fusion.

## Directory Structure

```
03_Regression/
├── src/                              # Core source modules
│   ├── networks.py                   # Model architectures (6 model types)
│   ├── losses.py                     # Loss functions (12 types)
│   ├── pipeline.py                   # Data loading & normalization
│   ├── trainers.py                   # Training loop, checkpointing, early stopping
│   ├── validators.py                 # Validation pipeline & metrics
│   ├── testers.py                    # Inference without ground truth
│   └── utils.py                      # Helpers (seed, device, plotting)
├── scripts/                          # Entry points
│   ├── train.py                      # Training
│   ├── validate.py                   # Validation with metrics
│   └── test.py                       # Inference
├── analysis/                         # Interpretability tools
│   ├── saliency_maps.py             # Grad-CAM, Integrated Gradients, Occlusion
│   ├── attention_analysis.py        # Cross-modal attention visualization
│   ├── monte_carlo_dropout.py       # Uncertainty estimation
│   ├── ablation_analysis.py         # Modality contribution analysis
│   └── cross_modal_analysis.py      # Gate weight & feature norm analysis
├── experiments/                      # Experiment management
│   ├── config_generator.py          # Generate experiment configs
│   ├── experiment_tracker.py        # Track metadata
│   ├── result_aggregator.py         # Aggregate results
│   └── submit_jobs.py               # SLURM job submission
├── configs/                          # Hydra configuration
│   ├── base.yaml                    # All default parameters
│   ├── local.yaml                   # macOS/MPS development
│   ├── wulver.yaml                  # HPC cluster (CUDA)
│   └── experiments/                 # Experiment-specific overrides
├── tests/                            # Unit tests
│   ├── test_networks.py             # 15+ model architecture tests
│   ├── test_losses.py               # 28 loss function tests
│   └── test_pipeline.py             # 10+ data pipeline tests
├── EXPERIMENTS.md                    # Experiment log (13 versions)
├── IMPROVEMENT_PLAN.md               # Development roadmap
└── README.md                         # Quick start guide
```

## Model Types

| Type | Architecture | Input | Description |
|------|-------------|-------|-------------|
| `fusion` | Transformer + ConvLSTM + CrossModalFusion | SDO + OMNI | Cross-modal attention fusion |
| `baseline` | Conv3D + Linear + concatenation | SDO + OMNI | Son et al. 2023 reproduction |
| `transformer` | Transformer encoder | OMNI only | Temporal sequence model |
| `tcn` | Temporal Convolutional Network | OMNI only | Dilated causal convolutions |
| `linear` | Linear encoder | OMNI only | Simple baseline |
| `convlstm` | ConvLSTM | SDO only | Spatial-temporal model |

### Architecture (fusion mode)

```
OMNI (12 vars, 56 timesteps) → TransformerEncoder → 128-d features
                                                          ↓
                                                  CrossModalFusion ← Bidirectional attention
                                                          ↓
SDO (3 ch, 28 frames, 64×64) → ConvLSTM → 128-d features
                                                          ↓
                                                  RegressionHead → Ap Index (24 timesteps)
```

## Configuration System

Hydra-based with inheritance:

```bash
# Local development
python scripts/train.py --config-name=local

# HPC cluster
python scripts/train.py --config-name=wulver

# CLI overrides
python scripts/train.py --config-name=local model.model_type=transformer training.epochs=50
```

**Key config sections**:
- `model.model_type`: Model selection (fusion, baseline, transformer, tcn, linear, convlstm)
- `training.regression_loss_type`: Loss function (mse, mae, huber, weighted_mse, solar_wind_weighted)
- `training.lambda_contrastive`: Contrastive loss weight (0.0 to disable)
- `data.dataset_name` / `data.dataset_suffix`: Dataset from 02_Acquisition

## Loss Functions

| Category | Types |
|----------|-------|
| **Regression** | MSE, MAE, Huber, WeightedMSE, SolarWindWeightedLoss |
| **Contrastive** | InfoNCE (MultiModalContrastiveLoss), MSE consistency |
| **Advanced** | AdaptiveWeight, GradientBased, Quantile, MultiTask |

**SolarWindWeightedLoss**: NOAA G-Scale tier weighting (none→1.0, G1→2.0, G2→4.0, G3+→8.0) with denormalization and temporal emphasis.

## Data Input

Consumes HDF5 files from 02_Acquisition:

```
{filename}.h5
├── omni/               # 12 solar wind variables, (80,) each
│   ├── [0:56] → input (T-7d to T)
│   └── [56:80] → target (T to T+3d)
└── sdo/
    ├── aia_193         # (28, 64, 64) uint8
    ├── aia_211         # (28, 64, 64) uint8
    └── hmi_magnetogram # (28, 64, 64) uint8
```

CSV file lists from 02_Acquisition Stage 3 specify train/validation splits.

## Common Commands

```bash
# Training
python scripts/train.py --config-name=local
python scripts/train.py --config-name=wulver

# Batch training
bash train.sh

# Validation
python scripts/validate.py --config-name=local experiment.name=baseline_v7 validation.epoch=best

# Batch validation
./validation.sh 7

# Analysis
./run_analysis.sh baseline best baseline_v7

# Tests
pytest tests/
pytest tests/test_networks.py -v
```

## Key Findings (from EXPERIMENTS.md)

- **Baseline (Conv3D+Linear) outperforms fusion model** (MAE 0.672 vs 0.701)
- **OMNI-only achieves best results** (MAE 0.666, R² +0.074)
- **SDO images contribute minimally** — SDO features are 4× weaker than OMNI
- **Heavy overfitting**: Best performance at epoch 1-2
- **Two-stage training does not improve** over joint training

## Dependencies

- `torch` (PyTorch)
- `hydra-core` / `omegaconf` (configuration)
- `h5py` (HDF5 data loading)
- `numpy`, `pandas`, `matplotlib`
- No egghouse dependency (standalone module)
