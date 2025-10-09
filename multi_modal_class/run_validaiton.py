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

    for n in range(10):
        lines = fixed.copy()
        epoch = (n + 1) * 50
        job_name = f"val_class_epoch_{epoch:03d}"

        checkpoint_path = f"/home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch{epoch}.pth"
        output_dir = f"/home/hl545/ap/results/wulver_mm_class/epoch_{epoch:03d}"

        command = f"srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint {checkpoint_path} --output_dir {output_dir}"
        lines.insert(2, f"#SBATCH --job-name={job_name}")
        lines.append(command)

        with open(f"tmp.sh", "w") as f:
            f.write("\n".join(lines))
        
        os.system(f"tmp.sh")
        time.sleep(10)
        os.remove("tmp.sh")
