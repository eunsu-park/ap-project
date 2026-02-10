#!/bin/bash -l

#SBATCH --job-name=AP_VALIDATION
#SBATCH --output=/home/hl545/ap/renew/train_outs/%x.%j.out
#SBATCH --error=/home/hl545/ap/renew/train_errs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --qos=high_wangj
#SBATCH --account=wangj
#SBATCH --time=09:00:00

module purge > /dev/null 2>&1
module load wulver # Load slurm, easybuild
conda activate ap

# Validation for all models

# ConvLSTM models
python validation.py --config-name CONV_1_1_4
python validation.py --config-name CONV_1_2_6
python validation.py --config-name CONV_1_3_3
python validation.py --config-name CONV_2_1_8
python validation.py --config-name CONV_2_2_6
python validation.py --config-name CONV_2_3_3
python validation.py --config-name CONV_3_1_2
python validation.py --config-name CONV_3_2_1
python validation.py --config-name CONV_3_3_1
python validation.py --config-name CONV_4_1_9
python validation.py --config-name CONV_4_2_2
python validation.py --config-name CONV_4_3_3
python validation.py --config-name CONV_5_1_1
python validation.py --config-name CONV_5_2_6
python validation.py --config-name CONV_5_3_2
python validation.py --config-name CONV_6_1_13
python validation.py --config-name CONV_6_2_7
python validation.py --config-name CONV_6_3_5
python validation.py --config-name CONV_7_1_6
python validation.py --config-name CONV_7_2_0
python validation.py --config-name CONV_7_3_0

# Transformer models
python validation.py --config-name TRANSFORMER_1_1_4
python validation.py --config-name TRANSFORMER_1_2_6
python validation.py --config-name TRANSFORMER_1_3_3
python validation.py --config-name TRANSFORMER_2_1_8
python validation.py --config-name TRANSFORMER_2_2_6
python validation.py --config-name TRANSFORMER_2_3_3
python validation.py --config-name TRANSFORMER_3_1_2
python validation.py --config-name TRANSFORMER_3_2_1
python validation.py --config-name TRANSFORMER_3_3_1
python validation.py --config-name TRANSFORMER_4_1_9
python validation.py --config-name TRANSFORMER_4_2_2
python validation.py --config-name TRANSFORMER_4_3_3
python validation.py --config-name TRANSFORMER_5_1_1
python validation.py --config-name TRANSFORMER_5_2_6
python validation.py --config-name TRANSFORMER_5_3_2
python validation.py --config-name TRANSFORMER_6_1_13
python validation.py --config-name TRANSFORMER_6_2_7
python validation.py --config-name TRANSFORMER_6_3_5
python validation.py --config-name TRANSFORMER_7_1_6
python validation.py --config-name TRANSFORMER_7_2_0
python validation.py --config-name TRANSFORMER_7_3_0
