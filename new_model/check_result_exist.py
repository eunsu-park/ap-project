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
        no_dir_list = []
        for n in range(10):
            epoch = (n+1) * 100
            epoch_dir = f"{output_dir}/epoch_{epoch}"
            if not os.path.exists(epoch_dir):
                no_dir_list.append(epoch)
        if len(no_dir_list) != 0 :
            print(os.path.basename(result_dir), no_dir_list)


    result_list = sorted(glob(f"{save_root}/weighted_days*"))
    num_result = len(result_list)
    print(num_result)
    
    for result_dir in result_list :
        output_dir = f"{result_dir}/output"
        no_dir_list = []
        for n in range(10):
            epoch = (n+1) * 100
            epoch_dir = f"{output_dir}/epoch_{epoch}"
            if not os.path.exists(epoch_dir):
                no_dir_list.append(epoch)
        if len(no_dir_list) != 0 :
            print(os.path.basename(result_dir), no_dir_list)



if __name__ == "__main__" :
    main()
