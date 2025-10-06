import os
import time
import yaml


HOME = os.path.expanduser('~')

fixed = [
    "#!/bin/bash -l",
    "",
    f"#SBATCH --output={HOME}/outs/%x.%j.out # %x.%j expands to slurm JobName.JobID",
    f"#SBATCH --error={HOME}/errs/%x.%j.err # prints the error message",
    "#SBATCH --partition=gpu",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --gres=gpu:1",
    "#SBATCH --mem-per-cpu=4000M # Maximum allowable mempry per CPU 4G",
    "#SBATCH --qos=standard",
    "#SBATCH --account=wangj # Replace PI_ucid which the NJIT UCID of PI",
    "#SBATCH --time=71:59:59  # D-HH:MM:SS",
    " ",
    "# Purge any module loaded by default",
    "module purge > /dev/null 2>&1",
    "module load wulver # Load slurm, easybuild",
    "conda activate ap",
]

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
        yaml_path_run = f"{HOME}/configs/config_wulver_mm_{loss}.yaml"
        with open(yaml_path_run, 'w') as f:
            yaml.dump(config_dict, f)

        save_root = f'{HOME}/ap/results'
        checkpoint_dir = os.path.join(save_root, config_dict["experiment_name"], 'checkpoint')
        
        for _epoch in range(10):

            epoch = (_epoch + 1) * 50

            lines = fixed.copy()
            job_name = f"val_wulver_mm_{loss}_epoch{epoch}"
            print(f"Submitting job: {job_name}")
            lines.insert(2, f"#SBATCH --job-name={job_name}")


            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
                continue
            else :
                print(f"Found checkpoint: {checkpoint_path}")
            validation_dir = os.path.join(save_root, config_dict["experiment_name"], 'validation', f'epoch{epoch}')
            os.makedirs(validation_dir, exist_ok=True)
            print(checkpoint_path, validation_dir)

            # command = f"nohup python validation.py --config {yaml_path_run} --checkpoint {checkpoint_path} --output_dir {validation_dir} &"
            command = f"srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --config {yaml_path_run}"
            print(command)
            

            os.system(command)
            time.sleep(300)
