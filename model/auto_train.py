import os
import yaml
import time

fixed = [
    "#!/bin/bash -l",
    "",
    "#SBATCH --output=./outs/%x.%j.out # %x.%j expands to slurm JobName.JobID",
    "#SBATCH --error=./errs/%x.%j.err # prints the error message",
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
        'mse',
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

        lines = fixed.copy()
        job_name = f"wulver_{loss}"
        print(f"Submitting job: {job_name}")
        lines.insert(2, f"#SBATCH --job-name={job_name}")
        yaml_path = "./configs/config_wulver_base.yaml"
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config_dict["experiment_name"] = f"wulver_{loss}"
        config_dict["loss_type"] = loss
        yaml_path_run = f"./configs/config_wulver_{loss}.yaml"
        if os.path.exists(yaml_path_run):
            os.remove(yaml_path_run)
        with open(yaml_path_run, 'w') as f:
            yaml.dump(config_dict, f)

        command = f"srun /home/hl545/miniconda3/envs/ap/bin/python train.py --config {yaml_path_run}"
        lines.append(command)

        with open("tmp.sh", "w") as f:
            f.write("\n".join(lines))    
            
        os.system(f"sbatch tmp.sh")
        # os.system(f"python train.py --config {yaml_path_run}")
        # os.system(f"cat tmp.sh")  # For testing purpose, print out the script instead of submitting it
        # os.system(f"cat tmp.yaml")  # For testing purpose, print out the script instead of submitting it
        time.sleep(10)
        os.remove("tmp.sh")
