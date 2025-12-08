import os
import yaml
from utils.slurm import WulverSubmitter


HOME = os.path.expanduser('~')
PYTHON_PATH = "/home/hl545/miniconda3/envs/ap/bin/python"
WULVER_CONFIG = {
    "OUT_DIR": f"{HOME}/ap/renew/train_outs",
    "ERR_DIR": f"{HOME}/ap/renew/train_errs",
    "PARTITION": "gpu",
    "NUM_NODE": 1,
    "NUM_CPU_CORE": 8,
    "NUM_GPU": 1,
    "MIG": False,
    "MEM": 8000,
    "QOS": "standard",
    "PI": "wangj",
    "TIME": "3-00:00:00" # D-HH:MM:SS"
}
SUBMITTER = WulverSubmitter(WULVER_CONFIG)

SYSTEM = "wulver" # "local" or "wulver"

CONTRASTIVE = [
    {"type": None, "temperature": 0.0, "lambda": 0.0},
    {"type": "mse", "temperature": 0.3, "lambda": 0.1},
    {"type": "infonce", "temperature": 0.3, "lambda": 0.1}
]   

INPUT_DAYS = [1, 2, 3, 4, 5, 6, 7]
OUTPUT_DAYS = [1]

SAMPLING = []
NUM_OVERSAMPLING = 5
SAMPLING.append(
    {"oversampling": True, "undersampling": False, "num_oversampling": NUM_OVERSAMPLING, "num_subsampling": 0, "subsampling_index": 0}    
)

NUM_SUBSAMPLING = 10
for n in range(NUM_SUBSAMPLING):
    SAMPLING.append(
        {"oversampling": False, "undersampling": True, "num_oversampling": 0, "num_subsampling": NUM_SUBSAMPLING, "subsampling_index": n}
    )

NUM_OVERSAMPLING_MAX = 5
NUM_SUBSAMPLING_MIX = 3
for n in range(NUM_SUBSAMPLING_MIX):
    SAMPLING.append(
        {"oversampling": True, "undersampling": True, "num_oversampling": NUM_OVERSAMPLING_MAX, "num_subsampling": NUM_SUBSAMPLING_MIX, "subsampling_index": n}
    )


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

    if info["enable_oversampling"] == True :
        experiment_name.append(f"OVER-{info['num_oversampling']:02d}")

    if info["enable_undersampling"] == True :
        experiment_name.append(f"UNDER-{info['subsampling_index']+1:02d}-OF-{info['num_subsampling']:02d}")

    experiment_name = "_".join(experiment_name)

    default_config_path = f"./configs/{info["system"]}_default.yaml"
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

    config_name = f"AUTO-TRAIN_{info['system']}_{experiment_name}.yaml"
    config_path = f"./configs/{config_name}"
    config_path = os.path.abspath(config_path)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_path}")

    return experiment_name, config_name


if __name__ == "__main__" :



    COUNT = 0

    for contrastive in CONTRASTIVE :
        contrastive_type = contrastive["type"]  # "mse" or "infonce" or None
        contrastive_temperature = contrastive["temperature"]
        lambda_contrastive = contrastive["lambda"]  # Weight for contrastive loss

        for sampling in SAMPLING :
            enable_oversampling = sampling["oversampling"]
            enable_undersampling = sampling["undersampling"]
            num_oversampling = sampling["num_oversampling"]
            num_subsampling = sampling["num_subsampling"]
            subsampling_index = sampling["subsampling_index"]

            for input_day in INPUT_DAYS :
                for output_day in OUTPUT_DAYS :

                    info = {}
                    info["system"] = SYSTEM.upper()
                    info["prefix"] = "REG"
                    info["contrastive_type"] = contrastive_type
                    info["contrastive_temperature"] = contrastive_temperature
                    info["lambda_contrastive"] = lambda_contrastive
                    info["enable_oversampling"] = enable_oversampling
                    info["num_oversampling"] = num_oversampling
                    info["enable_undersampling"] = enable_undersampling
                    info["num_subsampling"] = num_subsampling
                    info["subsampling_index"] = subsampling_index
                    info["input_day"] = input_day
                    info["output_day"] = output_day

                    experiment_name, config_name = generate_config(**info)
                    job_name = "TRAIN_" + experiment_name
                    print(f"Generated config: {config_name} for job: {job_name}")

                    commands= f"{PYTHON_PATH} train.py --config-name {config_name}"
                    script_path = f"AUTO-TRAIN_{experiment_name}.sh"

                    SUBMITTER.submit(
                        job_name=job_name,
                        commands=commands,
                        script_path=script_path,
                        dry_run=True
                    )
                    COUNT += 1

    print(f"\nTotal jobs prepared: {COUNT}")
