# python standard library
import os
from multiprocessing import freeze_support

# third-party library
import torch.optim as optim
import hydra

# custom library
from pipeline import create_dataloader
from networks import create_model
from losses import create_loss_functions
from utils import setup_experiment
from trainers import Trainer, save_training_history, plot_training_curves


# 향후 따로 빼야할까?
def create_optimizer(config, model):
    """Create optimizer from config."""
    if config.training.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=0.9)
    else:
        return optim.Adam(model.parameters(), lr=config.training.learning_rate)


# 향후 따로 빼야할까?
def create_scheduler(optimizer):
    """Create learning rate scheduler."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    device = setup_experiment(config)

    # 이 부분은 setup_experiment 안으로?
    # 디렉터리 관리하는 기능이 하나 있어야 할 것 같은데
    save_root = config.environment.save_root
    experiment_name = config.experiment.experiment_name
    experiment_dir = os.path.join(save_root, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(experiment_dir, "log")

    # print(or logger) 수정?
    dataloader = create_dataloader(config, phase="train")
    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(optimizer)
    criterion, contrastive_criterion = create_loss_functions(config)

    print(f"Optimizer: {config.training.optimizer.upper()}, LR: {config.training.learning_rate}")

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        contrastive_criterion=contrastive_criterion,
        device=device,
        logger=None
    )

    try:
        history = trainer.fit(dataloader, config.training.num_epochs)
        
        # Save results
        save_training_history(history, config, None)
        plot_training_curves(history, config, None)
        
        print("Training completed successfully")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__" :
    freeze_support()
    main()
