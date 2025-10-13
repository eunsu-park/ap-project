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
    "#SBATCH --time=3:59:59  # D-HH:MM:SS",
    " ",
    "# Purge any module loaded by default",
    "module purge > /dev/null 2>&1",
    "module load wulver # Load slurm, easybuild",
    "conda activate ap",
]


if __name__ == "__main__" :


    # lines = fixed.copy()
    # lines.insert(2, f"#SBATCH --job-name=class_G1")
    # config_name = "wulver_G1"
    # command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    # lines.append(command)
    # with open("tmp.sh", "w") as f:
    #     f.write("\n".join(lines))

    # command = "sbatch tmp.sh"
    # os.system(command)

    # time.sleep(5)
    # os.remove("tmp.sh")
    # time.sleep(5)


    lines = fixed.copy()
    lines.insert(2, f"#SBATCH --job-name=class_G2")
    config_name = "wulver_G2"
    command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    lines.append(command)
    with open("tmp.sh", "w") as f:
        f.write("\n".join(lines))

    command = "sbatch tmp.sh"
    os.system(command)

    time.sleep(5)
    os.remove("tmp.sh")
    time.sleep(5)


    # lines = fixed.copy()
    # lines.insert(2, f"#SBATCH --job-name=class_G3")
    # config_name = "wulver_G3"
    # command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    # lines.append(command)
    # with open("tmp.sh", "w") as f:
    #     f.write("\n".join(lines))

    # command = "sbatch tmp.sh"
    # os.system(command)

    # time.sleep(5)
    # os.remove("tmp.sh")
    # time.sleep(5)