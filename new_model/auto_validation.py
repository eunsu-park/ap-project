import os
import time
import yaml
import sys


HOME = os.path.expanduser('~')

fixed = [
    "#!/bin/bash -l",
    "",
    f"#SBATCH --output={HOME}/TEMP/%x.%j.out # %x.%j expands to slurm JobName.JobID",
    f"#SBATCH --error={HOME}/TEMP/%x.%j.err # prints the error message",
    "#SBATCH --partition=gpu",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --gres=gpu:1",
    "#SBATCH --mem-per-cpu=4000M # Maximum allowable mempry per CPU 4G",
    "#SBATCH --qos=standard",
    "#SBATCH --account=wangj # Replace PI_ucid which the NJIT UCID of PI",
    "#SBATCH --time=00:59:59  # D-HH:MM:SS",
    " ",
    "# Purge any module loaded by default",
    "module purge > /dev/null 2>&1",
    "module load wulver # Load slurm, easybuild",
    "conda activate ap",
]


if __name__ == "__main__" :

    base_config_names = [
        "G1_3_to_1",
        # "G1_3_to_1_1month",
        "G1_3_to_2",
        "G1_3_to_3",

        "G1_4_to_1",
        # "G1_4_to_1_1month",
        "G1_4_to_2",
        "G1_4_to_3",

        "G1_5_to_1",
        # "G1_5_to_1_1month",
        "G1_5_to_2",
        "G1_5_to_3"
        ]
    
    for base_config_name in base_config_names :

        base_config_path = f"./configs/val_{base_config_name}.yaml"
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f) 

        save_root = base_config["environment"]["save_root"]
        experiment_name = base_config["experiment"]["experiment_name"]
        experiment_dir = f"{save_root}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoint"
        output_root = f"{experiment_dir}/output"

        for n in range(10):

            lines = fixed.copy()
            epoch = (n+1)*50

            job_name = f"{experiment_name}_{epoch:03d}"
            print(f"Submitting job: {job_name}")
            lines.insert(2, f"#SBATCH --job-name={job_name}")

            checkpoint_path = f"{checkpoint_dir}/model_epoch{epoch}.pth"
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
                continue
            else :
                print(f"Found checkpoint: {checkpoint_path}")

            output_dir = f"{output_root}/epoch_{epoch:03d}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")

            config = base_config.copy()
            config["validation"] = {
                "checkpoint_path": checkpoint_path,
                "output_dir": output_dir
            }

            config_name = f"{experiment_name}_epoch{epoch:03d}"
            config_path = f"./configs/{config_name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                print(f"Saved config to: {config_path}")

            lines.append( f"/home/hl545/miniconda3/envs/ap/bin/python validation.py --config-name {config_name}")

            with open("tmp.sh", "w") as f:
                f.write("\n".join(lines))

            command = "sbatch tmp.sh"
            os.system(command)

            time.sleep(10)
            os.remove("tmp.sh")
            time.sleep(10)

        # break