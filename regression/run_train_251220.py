import os
import yaml
from utils.slurm import WulverSubmitter


## System info
HOME = os.path.expanduser('~')
PYTHON_PATH = "/home/hl545/miniconda3/envs/ap/bin/python"

# Default parameters
SYSTEM = "wulver" # "local" or "wulver"
CONTRASTIVE = [
    {"type": "mse", "temperature": 0.3, "lambda": 0.1},
]   
INPUT_DAYS = [1, 2, 3, 4, 5, 6, 7]
OUTPUT_DAYS = [3]
NUM_OVERSAMPLING = 5
NUM_SUBSAMPLING = 14
NUM_OVERSAMPLING_MAX = 5
NUM_SUBSAMPLING_MIX = 3


def generate_config(**info):

    experiment_name = []

    # Order: prefix + contrastive + date + over + under

    if not info["prefix"] == None :
        experiment_name.append(info["prefix"])

    if info["contrastive_type"] == None :
        pass
    elif info["contrastive_type"] == "mse" :
        experiment_name.append("MSE")
    elif info["contrastive_type"] == "infonce" :
        experiment_name.append("INFONCE")

    experiment_name.append(f"DATE-{info["input_day"]:02d}-TO-{info["output_day"]:02d}")

    if info["enable_oversampling"] == True and info["enable_undersampling"] == True :
        experiment_name.append("MIXED")
    elif info["enable_oversampling"] == True :
        experiment_name.append("OVER")
    elif info["enable_undersampling"] == True :
        experiment_name.append(f"UNDER-{info['subsampling_index']:02d}")
    else :
        experiment_name.append("ORIGINAL")

    experiment_name = "_".join(experiment_name)

    default_config_path = f"./configs/{info["system"].lower()}_default.yaml"
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["experiment"]["experiment_name"] = experiment_name
    config["data"]["sdo_sequence_length"] = 4 * info["input_day"]
    config["data"]["input_sequence_length"] = 8 * info["input_day"]
    config["data"]["target_sequence_length"] = 8 * info["output_day"]
    config["data"]["target_day"] = info["output_day"]
    config["experiment"]["enable_undersampling"] = info["enable_undersampling"]
    config["experiment"]["num_subsample"] = info["num_subsampling"]
    config["experiment"]["subsample_index"] = info["subsampling_index"]
    config["experiment"]["enable_oversampling"] = info["enable_oversampling"]
    config["experiment"]["num_oversample"] = info["num_oversampling"]
    config["training"]["contrastive_type"] = info["contrastive_type"]
    config["training"]["contrastive_temperature"] = info["contrastive_temperature"]
    config["training"]["lambda_contrastive"] = info["lambda_contrastive"]

    config_name = f"AUTO-TRAIN_{info['system'].lower()}_{experiment_name}.yaml"
    config_path = f"./configs/{config_name}"
    config_path = os.path.abspath(config_path)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_path}")

    return experiment_name, config_name


def run_single(dry_run, **info):



    experiment_name, config_name = generate_config(**info)
    job_name = "TRAIN_" + experiment_name
    print(f"Generated config: {config_name} for job: {job_name}")

    commands= f"{PYTHON_PATH} train.py --config-name {config_name}"
    script_path = f"AUTO-TRAIN_{experiment_name}.sh"

    SUBMITTER.submit(
        job_name=job_name,
        commands=commands,
        script_path=script_path,
        dry_run=dry_run
    )


if __name__ == "__main__" :

    WULVER_CONFIG = {
        "OUT_DIR": f"{HOME}/ap/renew/train_outs",
        "ERR_DIR": f"{HOME}/ap/renew/train_errs",
        "PARTITION": "gpu",
        "NUM_NODE": 1,
        "NUM_CPU_CORE": 8,
        "NUM_GPU": 1,
        "GPU": "gpu",
        "MEM": 8000,
        "QOS": "standard",
        # "QOS": "low",
        # "QOS": "high_wangj",
        "PI": "wangj",
        "TIME": "3-00:00:00" # D-HH:MM:SS"
    }
    SUBMITTER = WulverSubmitter(WULVER_CONFIG)

    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--dry_run', action='store_true', help='Dry run without submitting jobs')
    args = parse.parse_args()
    dry_run = args.dry_run

    base_config_path = "./configs/wulver_default.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)


    f = open("not_trained_tasks.txt", "r")
    lines = f.readlines()
    f.close()

    COUNT = 0

    for line in lines:
        items = line.strip().split("\t")
        prefix = items[0]
        sampling_type = items[1]
        input_day = int(items[2])
        output_day = int(items[3])
        subsampling_index = int(items[4])

        info = {}
        info["system"] = SYSTEM.upper()
        info["prefix"] = "REG"
        info["contrastive_type"] = "mse"
        info["contrastive_temperature"] = 0.3
        info["lambda_contrastive"] = 0.1
        info["enable_oversampling"] = False
        info["num_oversampling"] = 0
        info["enable_undersampling"] = True
        info["num_subsampling"] = NUM_SUBSAMPLING
        info["subsampling_index"] = subsampling_index
        info["input_day"] = input_day
        info["output_day"] = output_day

        experiment_name, config_name = generate_config(**info)
        job_name = "TRAIN_" + experiment_name
        print(f"Generated config: {config_name} for job: {job_name}")
        commands = f"{PYTHON_PATH} train.py --config-name {config_name}"
        script_path = f"AUTO-TRAIN_{experiment_name}.sh"

        SUBMITTER.submit(
            job_name=job_name,
            commands=commands,
            script_path=script_path,
            dry_run=dry_run
        )

        COUNT += 1

    print(COUNT)
