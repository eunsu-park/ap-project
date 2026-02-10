# python standard library
import os

# third-party library
import hydra
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# custom library
from pipeline import create_dataloader
from networks import create_model
from utils import setup_experiment


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    device = setup_experiment(config)

    output_dir = config.mcd.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Model type
    model_type = config.model.model_type

    print("=" * 70)
    print("MONTE CARLO DROPOUT - UNCERTAINTY ESTIMATION")
    print("=" * 70)
    print(f"Model type: {model_type}")
    print(f"Output directory: {output_dir}")

    # Create validation dataloader
    validation_dataloader = create_dataloader(config, 'validation')
    print(
        f"Validation dataloader: {len(validation_dataloader.dataset)} samples, "
        f"{len(validation_dataloader)} batches"
    )

    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    checkpoint_path = config.mcd.checkpoint_path
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    n_samples = 100

    for i, data_dict in enumerate(validation_dataloader):
        sdo = data_dict["sdo"].to(device)
        inputs = data_dict["inputs"].to(device)
        targets = data_dict["targets"].to(device)
        labels = data_dict["labels"].to(device)
        file_names = data_dict["file_names"]

        for n in range(len(file_names)):
            _sdo = sdo[n:n+1]
            _input = inputs[n:n+1]
            _target = targets[n:n+1]
            file_name = file_names[n]
            file_path = f"{output_dir}/{file_name}.npz"
            if not os.path.exists(file_path):
                predictions = []
                with torch.no_grad():
                    for _ in range(n_samples):
                        output, transformer_features, convlstm_features = model(
                            _input, _sdo, return_features=True
                        )
                        output = output.cpu().numpy()
                        output = output[:,:,0]
                        predictions.append(output)

                predictions = np.concatenate(predictions, 0)
                predictions = validation_dataloader.dataset.normalizer.denormalize_omni(predictions, 'ap_index_nt')
                mean = predictions.mean(0)
                uncertainty = predictions.std(0)

                np.savez(file_path, mean=mean, std=uncertainty)

                print(file_path)


if __name__ == "__main__" :
    main()



