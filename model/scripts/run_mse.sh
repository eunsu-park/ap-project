#!/bin/bash -l
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err # prints the error message
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000M # Maximum allowable mempry per CPU 4G
#SBATCH --qos=standard
#SBATCH --account=wangj # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --time=71:59:59  # D-HH:MM:SS

# Purge any module loaded by default
module purge > /dev/null 2>&1
module load wulver # Load slurm, easybuild
conda activate ap
srun /home/hl545/miniconda3/envs/ap/bin/python train.py --config ./configs/config_mse.yaml
