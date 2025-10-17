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

    task_list = ["G1", "G2", "G3", "G1_full"]

    for task in task_list :

        config_name = f"wulver_{task}"
        job_name = f"class_{task}"
        script_path = f"./{task}.sh"

        lines = fixed.copy()
        lines.insert(2, f"#SBATCH --job-name={job_name}")

        command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
        lines.append(command)

        with open(script_path, "w") as f:
            f.write("\n".join(lines))

        time.sleep(5)
        os.system(f"sbatch {script_path}")
        time.sleep(5)
        os.system(f"rm {script_path}")

        del lines, config_name, job_name, script_path