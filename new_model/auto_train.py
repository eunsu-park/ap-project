import os
import time
import yaml


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
    "#SBATCH --time=23:59:59  # D-HH:MM:SS",
    " ",
    "# Purge any module loaded by default",
    "module purge > /dev/null 2>&1",
    "module load wulver # Load slurm, easybuild",
    "conda activate ap",
]


if __name__ == "__main__" :

    config_names = [
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

    for config_name in config_names:

        script_path = f"./{config_name}.sh"

        lines = fixed.copy()
        lines.insert(2, f"#SBATCH --job-name={config_name}")

        command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
        lines.append(command)

        with open(script_path, "w") as f:
            f.write("\n".join(lines))

        # time.sleep(5)
        os.system(f"sbatch {script_path}")
        time.sleep(5)
        os.system(f"rm {script_path}")

        del lines, config_name, script_path