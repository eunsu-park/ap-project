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
    "#SBATCH --ntasks-per-node=8",
    # "#SBATCH --gres=gpu:1",
    "#SBATCH --gres=gpu:a100_10g:1",
    "#SBATCH --mem=8000M # Maximum allowable memory per CPU 4G",
    "#SBATCH --qos=standard",
    "#SBATCH --account=wangj # Replace PI_ucid which the NJIT UCID of PI",
    "#SBATCH --time=71:59:59  # D-HH:MM:SS",
    " ",
    "# Purge any module loaded by default",
    "module purge > /dev/null 2>&1",
    "module load wulver # Load slurm, easybuild",
    "conda activate ap",
]


def run(base_config, run_info):
    prefix = run_info["prefix"]
    input_day = run_info["input_day"]
    output_day = run_info["output_day"]
    enable_undersampling = run_info["enable_undersampling"]
    num_subsample = run_info["num_subsample"]
    subsample_index = run_info["subsample_index"]
    enable_oversampling = run_info["enable_oversampling"]
    num_oversample = run_info["num_oversample"]
    batch_size = run_info["batch_size"]
    num_workers = run_info["num_workers"]
    contrastive_type = run_info["contrastive_type"]
    contrastive_temperature = run_info["contrastive_temperature"]
    lambda_contrastive = run_info["lambda_contrastive"]
    report_freq = run_info["report_freq"]

    experiment_name = f"{prefix}_{input_day}_to_{output_day}"
    config_name = f"auto_{experiment_name}"
    config = base_config.copy()
    config["experiment"]["experiment_name"] = experiment_name
    config["data"]["sdo_sequence_length"] = 4 * input_day
    config["data"]["input_sequence_length"] = 8 * input_day
    config["data"]["target_sequence_length"] = 8 * output_day
    config["data"]["target_day"] = output_day
    config["experiment"]["enable_undersampling"] = enable_undersampling
    config["experiment"]["num_subsample"] = num_subsample
    config["experiment"]["subsample_index"] = subsample_index
    config["experiment"]["enable_oversampling"] = enable_oversampling
    config["experiment"]["num_oversample"] = num_oversample
    config["experiment"]["batch_size"] = batch_size
    config["experiment"]["num_workers"] = num_workers
    config["training"]["contrastive_type"] = contrastive_type
    config["training"]["contrastive_temperature"] = contrastive_temperature
    config["training"]["lambda_contrastive"] = lambda_contrastive
    config["training"]["report_freq"] = report_freq

    config_path = f"./configs/{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        print(f"Saved config to: {config_path}")

    lines = fixed.copy()
    lines.insert(2, f"#SBATCH --job-name=ap-train-{experiment_name}")

    command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    lines.append(command)

    script_path = f"./auto_{experiment_name}.sh"

    with open(script_path, "w") as f:
        f.write("\n".join(lines))

    os.system(f"sbatch {script_path}")
    del lines, config, config_name, script_path
    time.sleep(10)


def run_under(base_config, run_info):

    prefix = run_info["prefix"]
    input_day = run_info["input_day"]
    output_day = run_info["output_day"]
    enable_undersampling = run_info["enable_undersampling"]
    num_subsample = run_info["num_subsample"]
    enable_oversampling = run_info["enable_oversampling"]
    num_oversample = run_info["num_oversample"]
    batch_size = run_info["batch_size"]
    num_workers = run_info["num_workers"]
    contrastive_type = run_info["contrastive_type"]
    contrastive_temperature = run_info["contrastive_temperature"]
    lambda_contrastive = run_info["lambda_contrastive"]
    report_freq = run_info["report_freq"]

    experiment_name = f"{prefix}_{input_day}_to_{output_day}_under"

    lines = fixed.copy()
    lines.insert(2, f"#SBATCH --job-name=ap-train-{experiment_name}")

    for subsample_index in range(run_info["num_subsample"]):
        config_name = f"auto_{experiment_name}_{subsample_index:02d}"
        config = base_config.copy()
        config["experiment"]["experiment_name"] = experiment_name
        config["data"]["sdo_sequence_length"] = 4 * input_day
        config["data"]["input_sequence_length"] = 8 * input_day
        config["data"]["target_sequence_length"] = 8 * output_day
        config["data"]["target_day"] = output_day
        config["experiment"]["enable_undersampling"] = enable_undersampling
        config["experiment"]["num_subsample"] = num_subsample
        config["experiment"]["subsample_index"] = subsample_index
        config["experiment"]["enable_oversampling"] = enable_oversampling
        config["experiment"]["num_oversample"] = num_oversample
        config["experiment"]["batch_size"] = batch_size
        config["experiment"]["num_workers"] = num_workers
        config["training"]["contrastive_type"] = contrastive_type
        config["training"]["contrastive_temperature"] = contrastive_temperature
        config["training"]["lambda_contrastive"] = lambda_contrastive
        config["training"]["report_freq"] = report_freq

        config_path = f"./configs/{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            print(f"Saved config to: {config_path}")

        command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
        lines.append(command)

    script_path = f"./auto_{experiment_name}_under.sh"

    with open(script_path, "w") as f:
        f.write("\n".join(lines))

    os.system(f"sbatch {script_path}")
    del lines, config, config_name, script_path
    time.sleep(10)


def run_over(base_config, run_info):
    prefix = run_info["prefix"]
    input_day = run_info["input_day"]
    output_day = run_info["output_day"]
    enable_undersampling = run_info["enable_undersampling"]
    num_subsample = run_info["num_subsample"]
    subsample_index = run_info["subsample_index"]
    enable_oversampling = run_info["enable_oversampling"]
    num_oversample = run_info["num_oversample"]
    batch_size = run_info["batch_size"]
    num_workers = run_info["num_workers"]
    contrastive_type = run_info["contrastive_type"]
    contrastive_temperature = run_info["contrastive_temperature"]
    lambda_contrastive = run_info["lambda_contrastive"]
    report_freq = run_info["report_freq"]

    experiment_name = f"{prefix}_{input_day}_to_{output_day}_over"
    config_name = f"auto_{experiment_name}"
    config = base_config.copy()
    config["experiment"]["experiment_name"] = experiment_name
    config["data"]["sdo_sequence_length"] = 4 * input_day
    config["data"]["input_sequence_length"] = 8 * input_day
    config["data"]["target_sequence_length"] = 8 * output_day
    config["data"]["target_day"] = output_day
    config["experiment"]["enable_undersampling"] = enable_undersampling
    config["experiment"]["num_subsample"] = num_subsample
    config["experiment"]["subsample_index"] = subsample_index
    config["experiment"]["enable_oversampling"] = enable_oversampling
    config["experiment"]["num_oversample"] = num_oversample
    config["experiment"]["batch_size"] = batch_size
    config["experiment"]["num_workers"] = num_workers
    config["training"]["contrastive_type"] = contrastive_type
    config["training"]["contrastive_temperature"] = contrastive_temperature
    config["training"]["lambda_contrastive"] = lambda_contrastive
    config["training"]["report_freq"] = report_freq

    config_path = f"./configs/{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        print(f"Saved config to: {config_path}")

    lines = fixed.copy()
    lines.insert(2, f"#SBATCH --job-name=ap-train-{experiment_name}")

    command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    lines.append(command)

    script_path = f"./auto_{experiment_name}_over.sh"

    with open(script_path, "w") as f:
        f.write("\n".join(lines))

    os.system(f"sbatch {script_path}")
    del lines, config, config_name, script_path
    time.sleep(10)


if __name__ == "__main__" :

    base_config_path = "./configs/wulver.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    input_days = (1, 2, 3, 4, 5, 6, 7)
    output_day = 1

    for input_day in input_days :

        run_info = {
            "prefix": "reg",
            "input_day": input_day,
            "output_day": output_day,
            "enable_undersampling": False,
            "num_subsample": 10,
            "subsample_index": 0,
            "enable_oversampling": False,
            "num_oversample": 13,
            "batch_size": 4,
            "num_workers": 4,
            "contrastive_type": "mse",
            "contrastive_temperature" : 0.3,
            "lambda_contrastive": 0.0,
            "report_freq": 1000
        }
        # run(base_config, run_info)
        del run_info

        run_info = {
            "prefix": "reg",
            "input_day": input_day,
            "output_day": output_day,
            "enable_undersampling": False,
            "num_subsample": 10,
            "subsample_index": 0,
            "enable_oversampling": True,
            "num_oversample": 5,
            "batch_size": 4,
            "num_workers": 4,
            "contrastive_type": "mse",
            "contrastive_temperature" : 0.3,
            "lambda_contrastive": 0.0,
            "report_freq": 1000
        }
        run_over(base_config, run_info)
        del run_info

        run_info = {
            "prefix": "reg",
            "input_day": input_day,
            "output_day": output_day,
            "enable_undersampling": True,
            "num_subsample": 10,
            "subsample_index": 0,
            "enable_oversampling": False,
            "num_oversample": 13,
            "batch_size": 4,
            "num_workers": 4,
            "contrastive_type": "mse",
            "contrastive_temperature" : 0.3,
            "lambda_contrastive": 0.0,
            "report_freq": 100
        }
        # run_under(base_config, run_info)
        del run_info
