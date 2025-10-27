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
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --gres=gpu:1",
    "#SBATCH --mem-per-cpu=4000M # Maximum allowable mempry per CPU 4G",
    "#SBATCH --qos=standard",
    "#SBATCH --account=wangj # Replace PI_ucid which the NJIT UCID of PI",
    "#SBATCH --time=23:59:59  # D-HH:MM:SS",
    " ",
    "# Purge any module loaded by default",
    "module purge > /dev/null 2>&1",
    "module load wulver # Load slurm, easybuild",
    "conda activate ap",
]


if __name__ == "__main__" :

    input_days = (7, 6, 5, 4, 3, 2, 1)
    output_days = (4, 3, 2, 1)

    experiment_names =[]

    base_config_path = "./configs/wulver.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f) 

    for input_day in input_days :
        for output_day in output_days :
            experiment_name = f"days{input_day}_to_day{output_day}"
            config_name = experiment_name
            config = base_config.copy()
            config["experiment"]["experiment_name"] = experiment_name
            config["data"]["sdo_sequence_length"] = 4 * input_day
            config["data"]["input_sequence_length"] = 8 * input_day
            config["data"]["target_sequence_length"] = 8 * output_day
            config["data"]["target_day"] = output_day
            config_path = f"./configs/{config_name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                print(f"Saved config to: {config_path}")

            lines = fixed.copy()
            lines.insert(2, f"#SBATCH --job-name=ap-train-{config_name}")

            command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
            lines.append(command)

            script_path = f"./{config_name}.sh"

            with open(script_path, "w") as f:
                f.write("\n".join(lines))

            time.sleep(5)
            os.system(f"sbatch {script_path}")
            time.sleep(5)

            del lines, config_name, script_path



    # IOs = ((5, 1), (5, 2), (7, 2), (7, 3))

    # experiment_names =[]

    # base_config_path = "./configs/wulver.yaml"
    # with open(base_config_path, 'r') as f:
    #     base_config = yaml.safe_load(f) 

    # for input_day, output_day in IOs :

    #     experiment_name = f"days{input_day}_to_day{output_day}"
    #     config_name = experiment_name
    #     config = base_config.copy()
    #     config["experiment"]["experiment_name"] = experiment_name
    #     config["data"]["sdo_sequence_length"] = 4 * input_day
    #     config["data"]["input_sequence_length"] = 8 * input_day
    #     config["data"]["target_sequence_length"] = 8 * output_day
    #     config["data"]["target_day"] = output_day
    #     config_path = f"./configs/{config_name}.yaml"
    #     with open(config_path, 'w') as f:
    #         yaml.dump(config, f)
    #         print(f"Saved config to: {config_path}")

    #     lines = fixed.copy()
    #     lines.insert(2, f"#SBATCH --job-name=ap-train-{config_name}")

    #     command = f"/home/hl545/miniconda3/envs/ap/bin/python train.py --config-name {config_name}"
    #     lines.append(command)

    #     script_path = f"./{config_name}.sh"

    #     with open(script_path, "w") as f:
    #         f.write("\n".join(lines))

    #     time.sleep(5)
    #     os.system(f"sbatch {script_path}")
    #     time.sleep(5)

    #     del lines, config_name, script_path