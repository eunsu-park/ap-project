import os
import time
import yaml

from typing import Union, List

## System info
HOME = os.path.expanduser('~')
PYTHON_PATH = "/home/hl545/miniconda3/envs/ap/bin/python"

# Default parameters
CONTRASTIVE = [
    {"type": "mse", "temperature": 0.3, "lambda": 0.1},
]   

class WulverSubmitter:
    def __init__(self, config: dict):
        lines = [
            "#!/bin/bash -l",
            ""
        ]
        lines += [f"#SBATCH --output={config['OUT_DIR']}/%x.%j.out"]
        lines += [f"#SBATCH --error={config['ERR_DIR']}/%x.%j.err"]
        lines += [f"#SBATCH --partition={config['PARTITION']}"]
        lines += [f"#SBATCH --nodes={config['NUM_NODE']}"]
        lines += [f"#SBATCH --ntasks-per-node={config['NUM_CPU_CORE']}"]
        lines += [f"#SBATCH --gres={config['GPU']}:{config['NUM_GPU']}"]
        # # GPU configuration
        # if config.get("MIG", False):
        #     lines += [f"#SBATCH --gres=gpu:a100_10g:{config['NUM_GPU']}"]
        # else:
        #     lines += [f"#SBATCH --gres=gpu:{config['NUM_GPU']}"]
        
        lines += [f"#SBATCH --mem={config['MEM']:d}M"]
        
        # Validate QOS
        valid_qos = ("standard", f"high_{config['PI']}", "low")
        if config["QOS"] not in valid_qos:
            raise ValueError(f"Invalid QOS: {config['QOS']}. Must be one of {valid_qos}")
        
        lines += [f"#SBATCH --qos={config['QOS']}"]
        lines += [f"#SBATCH --account={config['PI']}"]
        lines += [f"#SBATCH --time={config['TIME']}"]
        lines += [""]
        lines += ["module purge > /dev/null 2>&1"]
        lines += ["module load wulver # Load slurm, easybuild"]
        lines += ["conda activate ap"]
        
        self.lines = lines

    def submit(self, job_name: str, commands: Union[str, List[str]], 
               script_path: str, dry_run: bool = True) -> None:
        lines = self.lines.copy()
        lines.insert(2, f"#SBATCH --job-name={job_name}")

        # Add commands
        if isinstance(commands, str):
            lines.append(commands)
        elif isinstance(commands, list):
            for command in commands:
                lines.append(command)
        else:
            raise TypeError(f"Commands must be str or list, got {type(commands)}")
        
        # Write script
        with open(script_path, "w") as f:
            f.write("\n".join(lines))
        
        # Submit if not dry run
        if not dry_run:
            time.sleep(5)
            os.system(f"sbatch {script_path}")
            time.sleep(5)

default_config_path = "./configs/wulver.yaml"
with open(default_config_path, 'r') as f:
    default_config = yaml.safe_load(f)

def get_info(I, T, epoch):
    config = default_config.copy()
    experiment_name = f"ORIGINAL_{I}_{T+1}"
    config["experiment"]["experiment_name"] = experiment_name

    config["experiment"]["enable_undersampling"] = False

    config["data"]["sdo_start_index"] = 40 - (I*4)
    config["data"]["input_start_index"] = 80 - (I*8) # 80-48=32

    config["data"]["target_days"] = [T+1]
    config["data"]["target_start_index"] = 80 + (T * 8)
    config["data"]["target_end_index"] = 80 + ((T+1) * 8)

    checkpoint_path = f'/home/hl545/ap/renew/results/{experiment_name}/checkpoint/model_epoch{epoch:04d}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"no checkpoint: {checkpoint_path}")
        return False
    config["validation"]["checkpoint_path"] = checkpoint_path
    config["mcd"]["checkpoint_path"] = checkpoint_path

    output_dir = f'/home/hl545/ap/renew/results/{experiment_name}/mcd/epoch_{epoch:04d}'
    if os.path.exists(f"{output_dir}/2024122300.h5.npz"):
        print(f"already done: {checkpoint_path}")
        return False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config["mcd"]["output_dir"] = output_dir
    config["validation"]["output_dir"] = output_dir

    config_name = f"AUTO-MCD-{experiment_name}"
    config_path = f"./configs/{config_name}.yaml"
    config_path = os.path.abspath(config_path)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    command = f"{PYTHON_PATH} monte_carlo_dropout.py --config-name {config_name}"
    return command


if __name__ == "__main__" :

    WULVER_CONFIG = {
        "OUT_DIR": f"{HOME}/ap/renew/mcd_outs",
        "ERR_DIR": f"{HOME}/ap/renew/mcd_errs",
        "PARTITION": "gpu",
        "NUM_NODE": 1,
        "NUM_CPU_CORE": 8,
        "NUM_GPU": 1,
        "GPU": "gpu",
        "MEM": 8000,
        "QOS": "high_wangj",
        "PI": "wangj",
        "TIME": "3-00:00:00" # D-HH:MM:SS"
    }

    submitter = WulverSubmitter(WULVER_CONFIG)

    def run(N, commands):
        job_name = f"AUTO-MCD-{N:03d}"
        script_path = f"{job_name}.sh"
        submitter.submit(job_name = job_name,
                        commands = commands,
                        script_path = script_path,
                        dry_run=False)  
        
    NUM_PACKAGE = 5
    INPUT_DAYS = [1, 2, 3, 4, 5, 6, 7]
    commands = []
    N = 0
    for I in INPUT_DAYS:
        for T in range(3): # TARGET DAYS
            for epoch in range(20, 101, 20):
                command = get_info(I, T, epoch)
                if command :
                    commands.append(command)
                if len(commands) == NUM_PACKAGE :
                    run(N, commands)
                    commands = []
                    N += 1

    if len(commands) > 0 :
        run(N, commands)
