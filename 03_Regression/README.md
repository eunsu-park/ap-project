# Multi-Modal Solar Wind Prediction

Multi-modal deep learning system for predicting Ap Index using solar wind data. Combines OMNI time series data (Transformer) and SDO solar images (ConvLSTM) through cross-modal attention fusion.

## Features

- **Multi-modal learning**: Combines temporal (OMNI) and spatial (SDO) data
- **Flexible model selection**: ConvLSTM-only, Transformer-only, Fusion, or Baseline mode
- **Cross-modal attention**: Bidirectional attention for modality fusion
- **SolarWindWeightedLoss**: NOAA G-Scale based weighted loss for geomagnetic storms
- **Two-stage training**: Separate contrastive pre-training and regression fine-tuning
- **Hydra configuration**: Easy experiment management with config inheritance

## Quick Start

```bash
cd /opt/projects/10_Harim/01_AP/02_Regression

# Verify model setup
python src/networks.py

# Train (local)
python scripts/train.py --config-name=local

# Train (HPC cluster)
python scripts/train.py --config-name=wulver

# Train (batch experiments)
bash train.sh

# Validate
python scripts/validate.py --config-name=local experiment.name=baseline_v4 validation.epoch=best
```

## Project Structure

```
02_Regression/
├── configs/           # Hydra configuration files
├── src/               # Core modules (networks, pipeline, trainers, etc.)
├── scripts/           # Entry points (train.py, validate.py, test.py)
├── analysis/          # Interpretability tools (Grad-CAM, Attention, etc.)
├── experiments/       # Large-scale experiment management
├── tests/             # Unit tests
├── docs/              # Detailed documentation
├── EXPERIMENTS.md     # Experiment log and results
└── Legacy/            # Archived reference code
```

## Model Types

| Model | Description | Input |
|-------|-------------|-------|
| `convlstm` | ConvLSTM-only | SDO images |
| `transformer` | Transformer-only | OMNI time series |
| `fusion` | Cross-modal attention fusion (default) | SDO + OMNI |
| `baseline` | Conv3D + Linear fusion (Son et al. 2023) | SDO + OMNI |

```bash
# Change model type via CLI
python scripts/train.py --config-name=local model.model_type=baseline
```

## Loss Functions

| Loss | Description |
|------|-------------|
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |
| `huber` | Huber Loss |
| `weighted_mse` | Target-value-based weighted MSE |
| `solar_wind_weighted` | NOAA G-Scale based weighted loss (default) |
| `none` | No regression loss (for two-stage training Stage 1) |

## Two-Stage Training

Separate contrastive pre-training and regression fine-tuning:

```bash
# Stage 1: Contrastive only
python scripts/train.py --config-name=local \
  experiment.name=fusion_twostage_s1 \
  training.regression_loss_type=none \
  training.lambda_contrastive=1.0

# Stage 2: Regression only (load Stage 1 weights)
python scripts/train.py --config-name=local \
  experiment.name=fusion_twostage_s2 \
  training.lambda_contrastive=0.0 \
  training.pretrained_checkpoint=fusion_twostage_s1/checkpoint/model_best.pth
```

## Current Best Results

| Model | MAE | RMSE | R² | Notes |
|-------|-----|------|-----|-------|
| baseline_v2 | 0.673 | 0.881 | +0.052 | Best baseline |
| fusion_v2 | 0.670 | 0.884 | +0.044 | Best fusion |
| baseline_v4 | 0.672 | 0.884 | +0.046 | Lightweight |

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed experiment log.

## Documentation

| Document | Description |
|----------|-------------|
| [Configuration](docs/configuration.md) | Hydra config system and parameters |
| [Models](docs/models.md) | Model architectures and components |
| [Data](docs/data.md) | Data format, pipeline, and time indexing |
| [Experiments](docs/experiments.md) | Large-scale experiment management (SLURM) |
| [Analysis](docs/analysis.md) | Interpretability tools and visualization |
| [Plot Guide](docs/plot_guide.md) | Plot interpretation guide |
| [API Reference](docs/api.md) | Module reference and usage |

## Installation

```bash
pip install torch torchvision
pip install hydra-core omegaconf
pip install h5py numpy pandas matplotlib
pip install opencv-python  # for saliency visualization
pip install pytest  # for testing
```

## Testing

```bash
pytest tests/ -v
```

## License

Internal research project.
