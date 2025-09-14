import os
import yaml

loss_map = [
    'mse',
    'mae',
    'huber',
    'huber_multi_criteria',
    'mae_outlier_focused',
    'adaptive_weight',
    'gradient_based_weight',
    'quantile',
    'multi_task',
    'weighted_mse'
]

lines = [
    "#!/bin/bash -l",
    "",
    "#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID",
    "#SBATCH --error=%x.%j.err # prints the error message",
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

for loss in loss_map :
    yaml_path = "./configs/config_wulver_base.yaml"
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config_dict["experiment_name"] = f"wulver_{loss}"
    config_dict["loss_type"] = loss
    yaml_path_run = f"./configs/config_wulver_{loss}.yaml"
    with open(yaml_path_run, 'w') as f:
        yaml.dump(config_dict, f)

    save_root = '/mmfs1/home/hl545/ap/results'
    checkpoint_dir = os.path.join(save_root, config_dict["experiment_name"], 'checkpoint')
    
    for _epoch in range(10):
        epoch = (_epoch + 1) * 10
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
            continue
        validation_dir = os.path.join(save_root, config_dict["experiment_name"], 'validation', f'epoch{epoch}')
        os.makedirs(validation_dir, exist_ok=True)
        print(checkpoint_path, validation_dir)
        # command = f"python validation.py --config tmp.yaml --checkpoint {checkpoint_path} --output_dir {validation_dir}"
        line = f"srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --config {yaml_path_run} --checkpoint {checkpoint_path} --output_dir {validation_dir}"
        lines = lines + [line]

        with open("tmp.sh", "w") as f:
            f.write("\n".join(lines))

        os.system(f"sbatch tmp.sh")
        # os.system(f"cat tmp.sh")  # For testing purpose, print out the script instead of submitting it
        os.remove("tmp.sh")
    # os.system(f"cat tmp.yaml")  # For testing purpose, print out the script instead of submitting it

