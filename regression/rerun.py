import os
import time
import yaml

from typing import Union, List

## System info
HOME = os.path.expanduser('~')
PYTHON_PATH = "/home/hl545/miniconda3/envs/ap/bin/python"

# Default parameters
SYSTEM = "wulver" # "local" or "wulver"
CONTRASTIVE = [
    {"type": "mse", "temperature": 0.3, "lambda": 0.1},
]   


"""
SLURM job submission utilities.

This module provides utilities for submitting jobs to SLURM
workload manager (specifically for Wulver cluster).
"""


class WulverSubmitter:
    """SLURM job submitter for Wulver cluster.
    
    This class generates and submits SLURM batch scripts for
    job execution on the Wulver HPC cluster.
    """
    
    def __init__(self, config: dict):
        """Initialize SLURM submitter with configuration.
        
        Args:
            config: Dictionary containing SLURM configuration parameters.
                   Required keys: OUT_DIR, ERR_DIR, PARTITION, NUM_NODE,
                   NUM_CPU_CORE, NUM_GPU, MEM, QOS, PI, TIME
        """
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
        """Submit a job to SLURM.
        
        Args:
            job_name: Name of the job.
            commands: Command(s) to execute. Can be a single string or list of strings.
            script_path: Path to save the SLURM script.
            dry_run: If True, only create the script without submitting.
        """
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


INFO = [
    (2, 1),
]
NUM_SUBSAMPLING = 14

WULVER_CONFIG = {
    "OUT_DIR": f"{HOME}/ap/renew/train_outs",
    "ERR_DIR": f"{HOME}/ap/renew/train_errs",
    "PARTITION": "gpu",
    "NUM_NODE": 1,
    "NUM_CPU_CORE": 8,
    "NUM_GPU": 1,
    "GPU": "gpu",
    # "GPU": "gpu:a100_10g",
    "MEM": 8000,
    # "QOS": "standard",
    # "QOS": "low",
    # "QOS": "high_wangj",
    "QOS": "standard",
    "PI": "wangj",
    "TIME": "3-00:00:00" # D-HH:MM:SS"
}

submitter = WulverSubmitter(WULVER_CONFIG)

commands = []
for I, T in INFO:
    for S in range(NUM_SUBSAMPLING) : 
        config = default_config.copy()
        experiment_name = f"SINGLE_{I}_{T+1}_{S:02d}"
        config["experiment"]["experiment_name"] = experiment_name

        config["experiment"]["enable_undersampling"] = True
        config["experiment"]["subsample_index"] = S

        config["data"]["sdo_start_index"] = 40 - (I*4)
        config["data"]["input_start_index"] = 80 - (I*8) # 80-48=32

        config["data"]["target_days"] = [T+1]
        config["data"]["target_start_index"] = 80 + (T * 8)
        config["data"]["target_end_index"] = 80 + ((T+1) * 8)

        config_name = f"AUTO-TRAIN_{experiment_name}.yaml"
        config_path = f"./configs/{config_name}"
        config_path = os.path.abspath(config_path)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        commands.append(f"{PYTHON_PATH} train.py --config-name {config_name}")

    job_name = f"AUTO-TRAIN_{I}_{T+1}_UNDER"
    script_path = f"{job_name}.sh"
    submitter.submit(job_name = job_name,
                    commands = commands,
                    script_path = script_path,
                    dry_run=False)
