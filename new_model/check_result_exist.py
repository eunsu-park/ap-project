import os
from glob import glob
import hydra


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    save_root = config.environment.save_root

    result_list = sorted(glob(f"{save_root}/days*"))
    num_result = len(result_list)
    print(num_result)
    
    for result_dir in result_list :
        output_dir = f"{result_dir}/output"
        for epoch in range(100, 1001, 100):
            epoch_dir = f"{output_dir}/epoch_{epoch}"
            if not os.path.exists(epoch_dir):
                print(epoch_dir)
            

if __name__ == "__main__" :
    main()
