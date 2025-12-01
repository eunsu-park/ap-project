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
    "TIME": "23:59:59" # D-HH:MM:SS"
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
    """
    기본 실험의 validation을 실행
    여러 epoch의 체크포인트를 하나의 job으로 묶어서 제출
    """
    config = generate_config(base_config, info)
    experiment_name = config["experiment"]["experiment_name"]
    save_root = base_config["environment"]["save_root"]
    experiment_dir = f"{save_root}/{experiment_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoint"
    output_root = f"{experiment_dir}/validation"
    job_name = f"ap-val-{experiment_name}"

    commands = []
    
    print(f"\n=== Running validation for base experiment: {experiment_name} ===")

    for epoch in range(20, 101, 20):
        config_name = f"auto_val_{experiment_name}_epoch_{epoch:04d}"
        checkpoint_path = f"{checkpoint_dir}/model_epoch{epoch}.pth"

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
            continue
        else:
            print(f"Found checkpoint: {checkpoint_path}")

        output_dir = f"{output_root}/epoch_{epoch:04d}"        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        else:
            print(f"Already created: {output_dir}")
            continue

        config["validation"] = {
            "checkpoint_path": checkpoint_path,
            "output_dir": output_dir
        }

        config_path = f"./configs/{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            print(f"Saved config to: {config_path}")

        command = f"/home/hl545/miniconda3/envs/ap/bin/python validation.py --config-name {config_name}"
        commands.append(command)

    if len(commands) > 0:
        script_path = f"./auto_val_{experiment_name}.sh"
        SUBMITTER.submit(job_name=job_name, commands=commands, script_path=script_path, dry_run=dry_run)
    else:
        print(f"No valid checkpoints found for {experiment_name}, skipping job submission.")



def run_over(base_config, info, dry_run=True):
    """
    오버샘플링 실험의 validation을 실행
    여러 epoch의 체크포인트를 하나의 job으로 묶어서 제출
    """
    info["ENABLE_OVERSAMPLING"] = True
    info["SUFFIX"] = "over"
    info["NUM_OVERSAMPLE"] = 5

    config = generate_config(base_config, info)
    experiment_name = config["experiment"]["experiment_name"]
    save_root = base_config["environment"]["save_root"]
    experiment_dir = f"{save_root}/{experiment_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoint"
    output_root = f"{experiment_dir}/validation"
    job_name = f"ap-val-{experiment_name}"

    commands = []
    
    print(f"\n=== Running validation for oversampling experiment: {experiment_name} ===")

    for epoch in range(20, 101, 20):
        config_name = f"auto_val_{experiment_name}_epoch_{epoch:04d}"
        checkpoint_path = f"{checkpoint_dir}/model_epoch{epoch}.pth"

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
            continue
        else:
            print(f"Found checkpoint: {checkpoint_path}")

        output_dir = f"{output_root}/epoch_{epoch:04d}"        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        else:
            print(f"Already created: {output_dir}")
            continue

        config["validation"] = {
            "checkpoint_path": checkpoint_path,
            "output_dir": output_dir
        }

        config_path = f"./configs/{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            print(f"Saved config to: {config_path}")

        command = f"/home/hl545/miniconda3/envs/ap/bin/python validation.py --config-name {config_name}"
        commands.append(command)

    if len(commands) > 0:
        script_path = f"./auto_val_{experiment_name}.sh"
        SUBMITTER.submit(job_name=job_name, commands=commands, script_path=script_path, dry_run=dry_run)
    else:
        print(f"No valid checkpoints found for {experiment_name}, skipping job submission.")


def run_under(base_config, info, dry_run=True):
    """
    언더샘플링으로 학습된 여러 서브샘플 모델들의 validation을 실행
    각 subsample_index마다 별도의 job을 생성하여 제출
    """
    main_experiment_name = f"{info["INPUT_DAY"]}_to_{info["OUTPUT_DAY"]}_under"
    if info["PREFIX"]:
        main_experiment_name = f"{info["PREFIX"]}_{main_experiment_name}"
    
    print(f"\n=== Running validation for undersampling experiments: {main_experiment_name} ===")
    
    for subsample_index in range(info["NUM_SUBSAMPLE"]):
        sub_info = info.copy()
        sub_info["ENABLE_UNDERSAMPLING"] = True
        sub_info["SUFFIX"] = f"sub_{subsample_index:02d}"
        sub_info["SUBSAMPLE_INDEX"] = subsample_index
        
        config = generate_config(base_config, sub_info)
        experiment_name = config["experiment"]["experiment_name"]
        save_root = base_config["environment"]["save_root"]
        experiment_dir = f"{save_root}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoint"
        output_root = f"{experiment_dir}/validation"
        job_name = f"ap-val-{experiment_name}"
        
        commands = []
        
        print(f"\n--- Processing subsample {subsample_index:02d}: {experiment_name} ---")
        
        for epoch in range(20, 101, 20):
            config_name = f"auto_val_{experiment_name}_epoch_{epoch:04d}"
            checkpoint_path = f"{checkpoint_dir}/model_epoch{epoch}.pth"
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
                continue
            else:
                print(f"Found checkpoint: {checkpoint_path}")
            
            output_dir = f"{output_root}/epoch_{epoch:04d}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            else:
                print(f"Already created: {output_dir}")
                continue
            
            config["validation"] = {
                "checkpoint_path": checkpoint_path,
                "output_dir": output_dir
            }
            
            config_path = f"./configs/{config_name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                print(f"Saved config to: {config_path}")
            
            command = f"/home/hl545/miniconda3/envs/ap/bin/python validation.py --config-name {config_name}"
            commands.append(command)
        
        if len(commands) > 0:
            script_path = f"./auto_val_{experiment_name}.sh"
            SUBMITTER.submit(job_name=job_name, commands=commands, script_path=script_path, dry_run=dry_run)
        else:
            print(f"No valid checkpoints found for {experiment_name}, skipping job submission.")


def run_mix(base_config, info, dry_run=True):
    """
    오버샘플링 + 언더샘플링 혼합 실험의 validation을 실행
    각 subsample_index마다 별도의 job을 생성하여 제출
    """
    main_experiment_name = f"{info["INPUT_DAY"]}_to_{info["OUTPUT_DAY"]}_mix"
    if info["PREFIX"]:
        main_experiment_name = f"{info["PREFIX"]}_{main_experiment_name}"
    
    print(f"\n=== Running validation for mixed (over+under) sampling experiments: {main_experiment_name} ===")
    
    # 혼합 샘플링: 언더샘플링 3개 + 오버샘플링 5개
    num_subsample_mix = 3
    num_oversample_mix = 5
    
    for subsample_index in range(num_subsample_mix):
        sub_info = info.copy()
        sub_info["ENABLE_UNDERSAMPLING"] = True
        sub_info["ENABLE_OVERSAMPLING"] = True
        sub_info["SUFFIX"] = f"mix_sub_{subsample_index:02d}"
        sub_info["SUBSAMPLE_INDEX"] = subsample_index
        sub_info["NUM_SUBSAMPLE"] = num_subsample_mix
        sub_info["NUM_OVERSAMPLE"] = num_oversample_mix
        
        config = generate_config(base_config, sub_info)
        experiment_name = config["experiment"]["experiment_name"]
        save_root = base_config["environment"]["save_root"]
        experiment_dir = f"{save_root}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoint"
        output_root = f"{experiment_dir}/validation"
        job_name = f"ap-val-{experiment_name}"
        
        commands = []
        
        print(f"\n--- Processing mixed subsample {subsample_index:02d}: {experiment_name} ---")
        print(f"    Undersampling: enabled (subsample_index={subsample_index})")
        print(f"    Oversampling: enabled (num_oversample={num_oversample_mix})")
        
        for epoch in range(20, 101, 20):
            config_name = f"auto_val_{experiment_name}_epoch_{epoch:04d}"
            checkpoint_path = f"{checkpoint_dir}/model_epoch{epoch}.pth"
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
                continue
            else:
                print(f"Found checkpoint: {checkpoint_path}")
            
            output_dir = f"{output_root}/epoch_{epoch:04d}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            else:
                print(f"Already created: {output_dir}")
                continue
            
            config["validation"] = {
                "checkpoint_path": checkpoint_path,
                "output_dir": output_dir
            }
            
            config_path = f"./configs/{config_name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                print(f"Saved config to: {config_path}")
            
            command = f"/home/hl545/miniconda3/envs/ap/bin/python validation.py --config-name {config_name}"
            commands.append(command)
        
        if len(commands) > 0:
            script_path = f"./auto_val_{experiment_name}.sh"
            SUBMITTER.submit(job_name=job_name, commands=commands, script_path=script_path, dry_run=dry_run)
        else:
            print(f"No valid checkpoints found for {experiment_name}, skipping job submission.")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", default=False)
    args = parser.parse_args()

    base_config_path = "./configs/wulver.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    input_days = (1, 2, 3, 4, 5, 6, 7)
    output_day = 1
    
    print("="*80)
    print("AUTO VALIDATION - SLURM Job Generator")
    print("="*80)
    print(f"Mode: {'SUBMIT' if args.run else 'DRY RUN'}")
    print(f"Input days: {input_days}")
    print(f"Output day: {output_day}")
    print(f"Epochs to validate: 20, 40, 60, 80, 100")
    print(f"Total experiments per input_day: 1 (base) + 10 (under) + 1 (over) + 3 (mix) = 15")
    print(f"Total jobs to be created: {len(input_days)} × 4 = {len(input_days) * 4}")
    print("="*80)

    for idx, input_day in enumerate(input_days, 1):
        print(f"\n{'='*80}")
        print(f"Processing input_day={input_day} ({idx}/{len(input_days)})")
        print(f"{'='*80}")
        
        RUN_INFO["INPUT_DAY"] = input_day
        RUN_INFO["OUTPUT_DAY"] = output_day
        
        run(base_config, RUN_INFO.copy(), dry_run=not args.run)
        run_under(base_config, RUN_INFO.copy(), dry_run=not args.run)
        run_over(base_config, RUN_INFO.copy(), dry_run=not args.run)
        run_mix(base_config, RUN_INFO.copy(), dry_run=not args.run)
    
    print("\n" + "="*80)
    print("AUTO VALIDATION - COMPLETED")
    print("="*80)
