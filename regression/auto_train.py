import os
import time
import yaml
from utils import WulverSubmitter


HOME = os.path.expanduser('~')
PYTHON_PATH = "/home/hl545/miniconda3/envs/ap/bin/python"
WULVER_CONFIG = {
    "OUT_DIR": f"{HOME}/TEMP",
    "ERR_DIR": f"{HOME}/TEMP",
    "PARTITION": "gpu",
    "NUM_NODE": 1,
    "NUM_CPU_CORE": 8,
    "NUM_GPU": 1,
    "MIG": False,
    "MEM": 8000,
    "QOS": "standard",
    "PI": "wangj",
    "TIME": "71:59:59" # D-HH:MM:SS"
}
SUBMITTER = WulverSubmitter(WULVER_CONFIG)
RUN_INFO = {
    "PREFIX": "reg",
    "SUFFIX": None,
    "INPUT_DAY": None,
    "OUTPUT_DAY": None,
    "ENABLE_UNDERSAMPLING": False,
    "NUM_SUBSAMPLE": 10,
    "SUBSAMPLE_INDEX": None,
    "ENABLE_OVERSAMPLING": False,
    "NUM_OVERSAMPLE": 13,
    "BATCH_SIZE": 4,
    "NUM_WORKERS": 4,
    "CONTRASTIVE_TYPE": "mse",
    "CONTRASTIVE_TEMPERATURE" : 0.3,
    "LAMBDA_CONTRASTIVE": 0.0,
    "REPORT_FREQ": 1000
}


def generate_config(base_config, info):
    experiment_name = f"{info["INPUT_DAY"]}_to_{info["OUTPUT_DAY"]}"
    if info["PREFIX"] :
        experiment_name = f"{info["PREFIX"]}_{experiment_name}"
    if info["SUFFIX"] :
        experiment_name = f"{experiment_name}_{info["SUFFIX"]}"
    config = base_config.copy()
    config["experiment"]["experiment_name"] = experiment_name
    config["data"]["sdo_sequence_length"] = 4 * info["INPUT_DAY"]
    config["data"]["input_sequence_length"] = 8 * info["INPUT_DAY"]
    config["data"]["target_sequence_length"] = 8 * info["OUTPUT_DAY"]
    config["data"]["target_day"] = info["OUTPUT_DAY"]
    config["experiment"]["enable_undersampling"] = info["ENABLE_UNDERSAMPLING"]
    config["experiment"]["num_subsample"] = info["NUM_SUBSAMPLE"]
    config["experiment"]["subsample_index"] = info["SUBSAMPLE_INDEX"]
    config["experiment"]["enable_oversampling"] = info["ENABLE_OVERSAMPLING"]
    config["experiment"]["num_oversample"] = info["NUM_OVERSAMPLE"]
    config["experiment"]["batch_size"] = info["BATCH_SIZE"]
    config["experiment"]["num_workers"] = info["NUM_WORKERS"]
    config["training"]["contrastive_type"] = info["CONTRASTIVE_TYPE"]
    config["training"]["contrastive_temperature"] = info["CONTRASTIVE_TEMPERATURE"]
    config["training"]["lambda_contrastive"] = info["LAMBDA_CONTRASTIVE"]
    config["training"]["report_freq"] = info["REPORT_FREQ"]
    return config


def run(base_config, info, dry_run=True):
    config = generate_config(base_config, info)
    experiment_name = config["experiment"]["experiment_name"]
    config_name = f"auto_{experiment_name}"
    job_name = f"ap-train-{experiment_name}"

    config_path = f"./configs/{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        print(f"Saved config to: {config_path}")

    commands = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    script_path = f"./auto_{experiment_name}.sh"
    SUBMITTER.submit(job_name=job_name, commands=commands, script_path=script_path, dry_run=dry_run)


def run_under(base_config, info, dry_run=True):
    commands = []
    main_experiment_name = f"{info["INPUT_DAY"]}_to_{info["OUTPUT_DAY"]}_under"
    if info["PREFIX"] :
        main_experiment_name = f"{info["PREFIX"]}_{main_experiment_name}"
    main_job_name = f"ap-train-{main_experiment_name}"

    for subsample_index in range(info["NUM_SUBSAMPLE"]):
        sub_info = info.copy()
        sub_info["ENABLE_UNDERSAMPLING"] = True
        sub_info["SUFFIX"] = f"sub_{subsample_index:02d}"
        sub_info["SUBSAMPLE_INDEX"] = subsample_index
        config = generate_config(base_config, sub_info)
        experiment_name = config["experiment"]["experiment_name"]
        config_name = f"auto_{experiment_name}"
        
        config_path = f"./configs/{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            print(f"Saved config to: {config_path}")
        command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
        commands.append(command)

    script_path = f"./auto_{main_experiment_name}.sh"
    SUBMITTER.submit(job_name=main_job_name, commands=commands, script_path=script_path, dry_run=dry_run)


def run_over(base_config, info, dry_run=True):

    info["ENABLE_OVERSAMPLING"] = True
    info["SUFFIX"] = "over"

    config = generate_config(base_config, info)
    experiment_name = config["experiment"]["experiment_name"]
    config_name = f"auto_{experiment_name}"
    job_name = f"ap-train-{experiment_name}"

    config_path = f"./configs/{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        print(f"Saved config to: {config_path}")

    commands = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    script_path = f"./auto_{experiment_name}.sh"
    SUBMITTER.submit(job_name=job_name, commands=commands, script_path=script_path, dry_run=dry_run)



if __name__ == "__main__" :

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", default=False)
    args = parser.parse_args()

    base_config_path = "./configs/wulver.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    input_days = (1, 2, 3, 4, 5, 6, 7)
    output_day = 1

    for input_day in input_days :
        RUN_INFO["INPUT_DAY"] = input_day
        RUN_INFO["OUTPUT_DAY"] = output_day
        WULVER_CONFIG["MIG"] = False
        run(base_config, RUN_INFO, dry_run=not args.run)
        WULVER_CONFIG["MIG"] = True
        run_under(base_config, RUN_INFO, dry_run=not args.run)
        WULVER_CONFIG["MIG"] = False
        run_over(base_config, RUN_INFO, dry_run=not args.run)
