import os
import time
import yaml
from multiprocessing import Pool, freeze_support

from typing import Union, List

## System info
HOME = os.path.expanduser('~')
PYTHON_PATH = "/Users/eunsupark/Softwares/miniconda3/envs/ap/bin/python"

# Default parameters
CONTRASTIVE = [
    {"type": "mse", "temperature": 0.3, "lambda": 0.1},
]   


INPUT_DAYS = [1, 2, 3, 4, 5, 6, 7]
TARGET_DAYS = [[1], [2], [3]]
NUM_SUBSAMPLING = 14

default_config_path = "./configs/local.yaml"
with open(default_config_path, 'r') as f:
    default_config = yaml.safe_load(f)


# def main(T, S, epoch, dry_run=True):
def main(args):
    T, S, epoch, dry_run = args

    config = default_config.copy()
    experiment_name = f"SINGLE_{T+1}_{S:02d}"
    config["experiment"]["experiment_name"] = experiment_name
    config["experiment"]["target_days"] = [T+1]
    config["experiment"]["subsample_index"] = S

    config["data"]["target_start_index"] = 80 + (T * 8)
    config["data"]["target_end_index"] = 80 + ((T+1) * 8)

    checkpoint_path = f'/opt/projects/10_Harim/01_AP/04_Result/{experiment_name}/checkpoint/model_epoch{epoch:04d}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"no checkpoint: {checkpoint_path}")
        return False
    config["validation"]["checkpoint_path"] = checkpoint_path

    # /Volumes/work/NJIT/01_AP/04_Results
    # output_dir = f'/opt/projects/10_Harim/01_AP/04_Result/{experiment_name}/validation/epoch_{epoch:04d}'
    # /Volumes/EUNSU-T9/10_Harim/01_AP/04_Result
    # output_dir = f'/Volumes/work/NJIT/01_AP/04_Results/{experiment_name}/validation/epoch_{epoch:04d}'
    output_dir = f'/Volumes/EUNSU-T9/10_Harim/01_AP/04_Result/{experiment_name}/validation/epoch_{epoch:04d}'
    if os.path.exists(f"{output_dir}/validation_results.txt"):
        return False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config["validation"]["output_dir"] = output_dir

    config_name = f"AUTO-VALIDATION_{experiment_name}"
    config_path = f"./configs/{config_name}.yaml"
    config_path = os.path.abspath(config_path)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    command = f"{PYTHON_PATH} validation.py --config-name {config_name}"
    print(command)

    if dry_run is False :
        os.system(command)
    
if __name__ == "__main__" :
    freeze_support()

    list_args = []

    for T in range(3): # TARGET DAYS
        for S in range(NUM_SUBSAMPLING) : 
            for epoch in range(20, 101, 20):
                # main(T, S, epoch, dry_run=False)
                list_args.append((T, S, epoch, False))

    pool = Pool(8)
    pool.map(main, list_args)
    pool.close()



# T = 2
# S = 7
# for epoch in range(20, 101, 20):
#     main(T, S, epoch, dry_run=False)
