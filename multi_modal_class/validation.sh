#!/bin/bash -l
#SBATCH --output=/home/hl545/outs/%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=/home/hl545/errs/%x.%j.err # prints the error message
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000M # Maximum allowable mempry per CPU 4G
#SBATCH --qos=standard
#SBATCH --account=wangj # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --time=71:59:59  # D-HH:MM:SS

# Purge any module loaded by default",
module purge > /dev/null 2>&1
module load wulver # Load slurm, easybuild
conda activate ap

srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch50.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_050
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch100.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_100
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch150.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_150
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch200.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_200
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch250.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_250
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch300.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_300
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch350.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_350
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch400.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_400
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch450.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_450
srun /home/hl545/miniconda3/envs/ap/bin/python validation.py --checkpoint /home/hl545/ap/results/wulver_mm_class/checkpoint/model_epoch500.pth --output_dir /home/hl545/ap/results/wulver_mm_class/epoch_500
