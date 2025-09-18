import os
import time
import yaml


if __name__ == "__main__" :
    loss_map = [
        # 'mse',
        'mae',
        'huber',
        'huber_multi_criteria',
        'mae_outlier_focused',
        'adaptive_weight',
        'gradient_based_weight',
        # 'quantile', # Error
        'multi_task',
        'weighted_mse'
    ]

    for loss in loss_map :

        yaml_path = "./configs/config_validation_base.yaml"
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config_dict["experiment_name"] = f"wulver_mm_{loss}"
        config_dict["loss_type"] = loss
        yaml_path_run = f"/Users/eunsupark/configs/config_validation_wulver_mm_{loss}.yaml"
        with open(yaml_path_run, 'w') as f:
            yaml.dump(config_dict, f)

        save_root = '/Volumes/work/Ap/results'
        # save_root = '/Users/eunsupark/ap_project/results'
        checkpoint_dir = os.path.join(save_root, config_dict["experiment_name"], 'checkpoint')
        
        for _epoch in range(10):
            epoch = (_epoch + 1) * 10

            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
                continue
            else :
                print(f"Found checkpoint: {checkpoint_path}")
            validation_dir = os.path.join(save_root, config_dict["experiment_name"], 'validation', f'epoch{epoch}')
            os.makedirs(validation_dir, exist_ok=True)
            print(checkpoint_path, validation_dir)

            command = f"nohup python validation.py --config {yaml_path_run} --checkpoint {checkpoint_path} --output_dir {validation_dir} &"
            print(command)

            os.system(command)
            time.sleep(600)
