import os
import time
import yaml
import sys


HOME = os.path.expanduser('~')

fixed = [
    "#!/bin/bash -l",
    "",
    f"#SBATCH --output={HOME}/TEMP_VAL/%x.%j.out # %x.%j expands to slurm JobName.JobID",
    f"#SBATCH --error={HOME}/TEMP_VAL/%x.%j.err # prints the error message",
    "#SBATCH --partition=gpu",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=4",
    # "#SBATCH --gres=gpu:1",
    "#SBATCH --gres=gpu:a100_10g:1",
    "#SBATCH --mem-per-cpu=8000M # Maximum allowable mempry per CPU 4G",
    "#SBATCH --qos=standard",
    "#SBATCH --account=wangj # Replace PI_ucid which the NJIT UCID of PI",
    "#SBATCH --time=00:59:59  # D-HH:MM:SS",
    " ",
    "# Purge any module loaded by default",
    "module purge > /dev/null 2>&1",
    "module load wulver # Load slurm, easybuild",
    "conda activate ap",
]


def original(base_config, input_day, output_day, epoch, overwrite=False):

    experiment_name = f"sdo_omni_over_days{input_day}_to_day{output_day}"
    save_root = base_config["environment"]["save_root"]
    experiment_dir = f"{save_root}/{experiment_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoint"
    output_root = f"{experiment_dir}/output"

    config_name = f"{experiment_name}_epoch_{epoch:04d}"

    config = base_config.copy()
    config["experiment"]["experiment_name"] = experiment_name
    config["data"]["sdo_sequence_length"] = 4 * input_day
    config["data"]["input_sequence_length"] = 8 * input_day
    config["data"]["target_sequence_length"] = 8 * output_day
    config["data"]["target_day"] = output_day
    config["experiment"]["apply_pos_weight"] = False
    config["experiment"]["enable_undersampling"] = False
    config["experiment"]["enable_oversampling"] = True
    config["experiment"]["num_oversample"] = 13

    config["training"]["report_freq"] = 1000
    checkpoint_path = f"{checkpoint_dir}/model_epoch{epoch}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
        return
    else :
        print(f"Found checkpoint: {checkpoint_path}")

    output_dir = f"{output_root}/epoch_{epoch:03d}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else :
        if overwrite is True :
            os.system(f"rm -rf {output_dir}")
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        else :
            print(f"Already created: {output_dir}")
            return


    config["validation"] = {
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir
    }

    config_path = f"./configs/{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Saved config to: {config_path}")

    lines = fixed.copy()
    lines.insert(2, f"#SBATCH --job-name=ap-val-{config_name}")

    command = f"/home/hl545/miniconda3/envs/ap/bin/python validation.py --config-name {config_name}"
    lines.append(command)

    script_path = f"./{config_name}.sh"

    with open(script_path, "w") as f:
        f.write("\n".join(lines))

    os.system(f"sbatch {script_path}")
    del lines, config, config_name, script_path
    time.sleep(5)
    

def subsample(base_config, input_day, output_day, epoch, subsample_index, overwrite=False):

    experiment_name = f"contrastive_days{input_day}_to_day{output_day}_sub_{subsample_index:02d}"
    save_root = base_config["environment"]["save_root"]
    experiment_dir = f"{save_root}/{experiment_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoint"
    output_root = f"{experiment_dir}/output"

    config_name = f"{experiment_name}_epoch_{epoch:04d}"

    config = base_config.copy()
    config["experiment"]["experiment_name"] = experiment_name

    config["experiment"]["experiment_name"] = experiment_name
    config["data"]["sdo_sequence_length"] = 4 * input_day
    config["data"]["input_sequence_length"] = 8 * input_day
    config["data"]["target_sequence_length"] = 8 * output_day
    config["data"]["target_day"] = output_day
    config["experiment"]["apply_pos_weight"] = False
    config["experiment"]["enable_undersampling"] = True
    config["experiment"]["num_subsample"] = 10
    config["experiment"]["subsample_index"] = subsample_index
    config["experiment"]["enable_oversampling"] = False
    config["experiment"]["num_workers"] = 4
    config["training"]["contrastive_type"] = 'mse'
    config["training"]["report_freq"] = 1000

    checkpoint_path = f"{checkpoint_dir}/model_epoch{epoch}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
        return
    else :
        print(f"Found checkpoint: {checkpoint_path}")

    output_dir = f"{output_root}/epoch_{epoch:03d}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else :
        if overwrite is True :
            os.system(f"rm -rf {output_dir}")
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        else :
            print(f"Already created: {output_dir}")
            return

    config["validation"] = {
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir
    }

    config_path = f"./configs/{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Saved config to: {config_path}")

    lines = fixed.copy()
    lines.insert(2, f"#SBATCH --job-name=ap-val-{config_name}")

    command = f"/home/hl545/miniconda3/envs/ap/bin/python validation.py --config-name {config_name}"
    lines.append(command)

    script_path = f"./{config_name}.sh"

    with open(script_path, "w") as f:
        f.write("\n".join(lines))

    os.system(f"sbatch {script_path}")
    del lines, config, config_name, script_path
    time.sleep(5)


if __name__ == "__main__" :

    base_config_path = "./configs/wulver.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f) 

    input_days = (1, 2, 3, 4, 5, 6, 7)
    output_day = 1

    for input_day in input_days :
        for epoch in range(20, 101, 20) :

            # original(base_config, input_day, output_day, epoch)
            
            for subsample_index in range(10):

                subsample(base_config, input_day, output_day, epoch, subsample_index)                

    ### Single run
    # original(base_config, input_day=4, output_day=1, epoch=100, overwrite=True)
    # weighted_subsample(base_config, input_day=5, output_day=1, epoch=100, subsample_index=1, overwrite=True)
