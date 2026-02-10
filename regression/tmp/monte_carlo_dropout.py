# python standard library
import os

# third-party library
import hydra
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# custom library
from config import Config
from datasets import create_dataloader
from models import create_model


@hydra.main(config_path="./configs", version_base=None)
def main(hydra_config):
    config = Config.from_hydra(hydra_config)
    print(config)

    output_dir = config.validation.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config.experiment.phase = 'validation'
    config.experiment.batch_size = 1
    
    # Create validation dataloader
    validation_dataloader = create_dataloader(config)
    print(
        f"Validation dataloader: {len(validation_dataloader.dataset)} samples, "
        f"{len(validation_dataloader)} batches"
    )



    model = create_model(config).to(config.environment.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    checkpoint_path = config.validation.checkpoint_path
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=config.environment.device)
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
        sdo = data_dict["sdo"].to("mps")
        inputs = data_dict["inputs"].to("mps")
        targets = data_dict["targets"].to("mps")
        labels = data_dict["labels"].to("mps")
        file_names = data_dict["file_names"]

        print(sdo.shape)
        print(inputs.shape)
        print(targets.shape)
        print(labels.shape)
        print(file_names)
    
        predictions = []
    
        with torch.no_grad():
            for _ in range(n_samples):
                outputs, transformer_features, convlstm_features = model(
                    inputs, sdo, return_features=True
                )
                outputs = outputs.cpu().numpy()
                outputs = outputs[:,:,0]
                predictions.append(outputs)

        print(outputs.shape)
        print(len(predictions))
        predictions = np.concatenate(predictions, 0)

        mean = predictions.mean(0)
        uncertainty = predictions.std(0)
        print(mean)
        print(uncertainty)

        break

    plt.figure()
    x = range(8)
    plt.plot(x, targets.cpu().numpy()[0,:,0], label="target")
    plt.plot(x, mean, label="output")
    # plt.fill_between(mean - uncertainty, mean + uncertainty, alpha=0.25, label="output ± uncertainty")
    plt.errorbar(x, mean, yerr=uncertainty, ms=3, lw=1, capsize=2, label="output ± 1σ")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()    



if __name__ == "__main__" :
    main()



