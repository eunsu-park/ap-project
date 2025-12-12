#!/bin/bash -l

#SBATCH --job-name=AP_TRAIN_REG_MSE_UNDER-ALL
#SBATCH --output=/Users/eunsupark/ap/renew/train_outs/%x.%j.out
#SBATCH --error=/Users/eunsupark/ap/renew/train_errs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --qos=standard
#SBATCH --account=wangj
#SBATCH --time=0-00:59:59

module purge > /dev/null 2>&1
module load wulver # Load slurm, easybuild
conda activate ap
/home/hl545/miniconda3/envs/ap/bin/python test_data.py --config-name AUTO-TRAIN_wulver_REG_MSE_DATE-01-TO-01_UNDER-00.yaml