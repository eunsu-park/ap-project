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
    "#SBATCH --ntasks-per-node=4",
    "#SBATCH --gres=gpu:1",
    # "#SBATCH --gres=gpu:a100_10g:1",
    "#SBATCH --mem=8000M # Maximum allowable mempry per CPU 4G",
    "#SBATCH --qos=standard",
    "#SBATCH --account=wangj # Replace PI_ucid which the NJIT UCID of PI",
    "#SBATCH --time=71:59:59  # D-HH:MM:SS",
    " ",
    "# Purge any module loaded by default",
    "module purge > /dev/null 2>&1",
    "module load wulver # Load slurm, easybuild",
    "conda activate ap",
]


if __name__ == "__main__" :

    base_config_path = "./configs/wulver.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    
    input_days = (1, 2, 3, 4, 5, 6, 7)
    output_day = 1
    for subsample_index in range(10):

        lines = fixed.copy()
        lines.insert(2, f"#SBATCH --job-name=ap-train-sub:{subsample_index:02d}")

        for input_day in input_days :
            # experiment_name = f"mig_balanced_days{input_day}_to_day{output_day}_sub_{subsample_index}"
            experiment_name = f"contrastive_days{input_day}_to_day{output_day}_sub_{subsample_index:02d}"
            config_name = experiment_name
            config = base_config.copy()
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
            config["experiment"]["batch_size"] = 4
            config["experiment"]["num_workers"] = 4
            config["training"]["report_freq"] = 1000

            config_path = f"./configs/{config_name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                print(f"Saved config to: {config_path}")

            command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
            lines.append(command)

        script_path = f"./{config_name}.sh"

        with open(script_path, "w") as f:
            f.write("\n".join(lines))

        os.system(f"sbatch {script_path}")
        del lines, config, config_name, script_path
        # time.sleep(5)



    # base_config_path = "./configs/wulver.yaml"
    # with open(base_config_path, 'r') as f:
    #     base_config = yaml.safe_load(f)

    # scripts = []

    
    # input_days = (1, 2, 3, 4, 5, 6, 7)
    # output_day = 1
    # for input_day in input_days :
    #     for subsample_index in range(10):

    #         # experiment_name = f"mig_balanced_days{input_day}_to_day{output_day}_sub_{subsample_index}"
    #         experiment_name = f"contrastive_days{input_day}_to_day{output_day}_sub_{subsample_index}"
    #         config_name = experiment_name
    #         config = base_config.copy()
    #         config["experiment"]["experiment_name"] = experiment_name
    #         config["data"]["sdo_sequence_length"] = 4 * input_day
    #         config["data"]["input_sequence_length"] = 8 * input_day
    #         config["data"]["target_sequence_length"] = 8 * output_day
    #         config["data"]["target_day"] = output_day
    #         config["experiment"]["apply_pos_weight"] = False
    #         config["experiment"]["enable_undersampling"] = True
    #         config["experiment"]["num_subsample"] = 10
    #         config["experiment"]["subsample_index"] = subsample_index
    #         config["experiment"]["enable_oversampling"] = False
    #         config["experiment"]["batch_size"] = 4
    #         config["experiment"]["num_workers"] = 4
    #         config["training"]["report_freq"] = 1000

    #         config_path = f"./configs/{config_name}.yaml"
    #         with open(config_path, 'w') as f:
    #             yaml.dump(config, f)
    #             print(f"Saved config to: {config_path}")

    #         lines = fixed.copy()
    #         lines.insert(2, f"#SBATCH --job-name=ap-train-{config_name}")

    #         command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    #         lines.append(command)

    #         script_path = f"./{config_name}.sh"

    #         with open(script_path, "w") as f:
    #             f.write("\n".join(lines))

    #         os.system(f"sbatch {script_path}")
    #         del lines, config, config_name, script_path
    #         # time.sleep(5)

        # experiment_name = f"sdo_omni_over_days{input_day}_to_day{output_day}"
        # config_name = experiment_name
        # config = base_config.copy()
        # config["experiment"]["experiment_name"] = experiment_name
        # config["data"]["sdo_sequence_length"] = 4 * input_day
        # config["data"]["input_sequence_length"] = 8 * input_day
        # config["data"]["target_sequence_length"] = 8 * output_day
        # config["data"]["target_day"] = output_day
        # config["experiment"]["apply_pos_weight"] = False
        # config["experiment"]["enable_undersampling"] = True
        # config["experiment"]["num_subsample"] = 10
        # config["experiment"]["enable_oversampling"] = False
        # config["experiment"]["batch_size"] = 4
        # config["experiment"]["num_workers"] = 4
        # config["training"]["report_freq"] = 1000
        # config_path = f"./configs/{config_name}.yaml"

        # with open(config_path, 'w') as f:
        #     yaml.dump(config, f)
        #     print(f"Saved config to: {config_path}")

        # lines = fixed.copy()
        # lines.insert(2, f"#SBATCH --job-name=ap-train-{config_name}")

        # command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
        # lines.append(command)

        # script_path = f"./{config_name}.sh"

        # with open(script_path, "w") as f:
        #     f.write("\n".join(lines))

        # os.system(f"sbatch {script_path}")
        # del lines, config, config_name, script_path
        # # time.sleep(5)
