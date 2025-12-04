# Solar Wind Prediction - Modularized Codebase

This directory contains the modularized implementation of the multimodal solar wind prediction system.

## Project Structure

```
.
├── datasets/          # Data pipeline (8 files, ~34KB)
│   ├── __init__.py           # Main exports
│   ├── config.py             # DataConfig
│   ├── statistics.py         # Normalizer, OnlineStatistics, compute_statistics
│   ├── io.py                 # HDF5Reader
│   ├── preprocessing.py      # DataProcessor
│   ├── sampling.py           # SamplingStrategy
│   ├── dataset.py            # BaseDataset, TrainDataset, ValidationDataset
│   └── dataloader.py         # create_dataloader
│
├── models/            # Neural network models (6 files, ~35KB)
│   ├── __init__.py           # Main exports
│   ├── convlstm.py           # ConvLSTMCell, ConvLSTMModel
│   ├── transformer.py        # PositionalEncoding, TransformerEncoderModel
│   ├── fusion.py             # CrossModalAttention, CrossModalFusion
│   ├── multimodal.py         # MultiModalModel
│   └── factory.py            # create_model
│
├── losses/            # Loss functions (5 files, ~40KB)
│   ├── __init__.py           # Main exports
│   ├── contrastive.py        # MultiModalContrastiveLoss, MultiModalMSELoss
│   ├── regression.py         # WeightedMSELoss, HuberMultiCriteriaLoss, MAEOutlierFocusedLoss
│   ├── advanced.py           # AdaptiveWeightLoss, GradientBasedWeightLoss, QuantileLoss, MultiTaskLoss
│   └── factory.py            # create_loss_functions, create_loss
│
├── utils/             # Utility functions (8 files, ~20KB)
│   ├── __init__.py           # Main exports
│   ├── seed.py               # set_seed
│   ├── logging_utils.py      # setup_logger, log_message
│   ├── device.py             # setup_device
│   ├── model_io.py           # load_model
│   ├── visualization.py      # save_plot, plotting functions
│   ├── metrics.py            # calculate_metrics
│   └── slurm.py              # WulverSubmitter
│
├── trainers.py        # Training components (Trainer, MetricsTracker, CheckpointManager)
├── validators.py      # Validation components (Validator, MetricsAggregator, ResultsWriter)
├── train.py           # Main training script
└── validation.py      # Main validation script
```

## Migration Guide

### Original Code
```python
from pipeline import create_dataloader
from networks import create_model
from losses import create_loss_functions
from utils import set_seed, setup_logger, setup_device, load_model
```

### New Code
```python
from datasets import create_dataloader
from models import create_model
from losses import create_loss_functions
from utils import set_seed, setup_logger, setup_device, load_model
```

**Required changes:**
- `pipeline` → `datasets`
- `networks` → `models`
- `losses` → `losses` (same, but now modularized)
- `utils` → `utils` (same, but now modularized)

## Usage Examples

### 0. Training and Validation (Main Scripts)
```python
# Training
python train.py

# Validation
python validation.py

# Both scripts use Hydra for configuration management
# Configuration files should be in ./configs/ directory
```

### 1. Data Pipeline
```python
from datasets import TrainDataset, ValidationDataset, create_dataloader

# Using factory function (recommended)
train_loader = create_dataloader(config)  # phase='train'
val_loader = create_dataloader(config)    # phase='validation'

# Or directly
from datasets import DataConfig, compute_statistics

config = DataConfig(...)
stats = compute_statistics(...)
train_dataset = TrainDataset(config, stats)
val_dataset = ValidationDataset(config, stats)
```

### 2. Model Creation
```python
from models import create_model

# Using factory function (recommended)
model = create_model(config, logger)

# Or directly
from models import MultiModalModel

model = MultiModalModel(
    num_input_variables=10,
    input_sequence_length=120,
    num_target_variables=3,
    num_target_groups=24,
    ...
)
```

### 3. Loss Functions
```python
from losses import create_loss_functions

# Get both regression and contrastive losses
regression_loss, contrastive_loss = create_loss_functions(config, logger)

# Or use specific losses
from losses import HuberMultiCriteriaLoss, MultiModalContrastiveLoss

reg_loss = HuberMultiCriteriaLoss(beta=0.3)
cont_loss = MultiModalContrastiveLoss(temperature=0.3)
```

### 4. Utilities
```python
from utils import set_seed, setup_logger, setup_device, load_model

# Setup environment
set_seed(42)
logger = setup_logger(__name__, log_dir='./logs')
device = setup_device(config, logger)

# Load model
model = load_model(model, checkpoint_path, device, logger)

# Visualization
from utils import save_plot
save_plot(targets, outputs, var_names, stats, 'plot_path', 'Title')
```

## Key Features

### datasets/
- **Modular design**: Separate Train/Validation datasets
- **Clean separation**: Config, statistics, I/O, preprocessing, sampling
- **Reproducibility**: Explicit seeding, deterministic sampling
- **Memory efficient**: Online statistics computation

### models/
- **Clear architecture**: ConvLSTM, Transformer, Fusion separated
- **Type safety**: Comprehensive input validation
- **Flexibility**: Easy to modify individual components
- **Documentation**: Detailed docstrings for all classes

### losses/
- **Variety**: 10+ loss functions for different scenarios
- **Organized**: Contrastive, regression, advanced categories
- **Factory pattern**: Easy loss creation from config
- **Research-friendly**: Easy to add new loss functions

### utils/
- **Comprehensive**: All utility functions in one place
- **Well-organized**: Separate modules for different concerns
- **Logging**: Consistent logging throughout
- **HPC support**: SLURM job submission utilities

## Benefits of Modularization

1. **Maintainability**: Easy to locate and modify specific functionality
2. **Testability**: Each module can be tested independently
3. **Reusability**: Import only what you need
4. **Readability**: Clear file names indicate functionality
5. **Collaboration**: Multiple people can work on different modules
6. **Scalability**: Easy to add new features without breaking existing code

## Backward Compatibility

The modularized code maintains 100% backward compatibility with the original monolithic files:
- All function signatures unchanged
- All class interfaces unchanged
- Only import statements need updating

## Notes for Researchers

- Each package has clear `__init__.py` with usage examples
- All original functionality preserved
- No performance impact from modularization
- Easy to extend for new experiments
- Clean separation makes paper writing easier (clear architecture diagrams)

## Version Information

- **Created**: December 2024
- **Python**: 3.8+
- **PyTorch**: 1.x+
- **Dependencies**: numpy, pandas, h5py, matplotlib, hydra-core

---

For questions or issues, please refer to individual module docstrings or contact the development team.