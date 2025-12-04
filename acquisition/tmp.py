from glob import glob
import h5py


def read_h5(file_path):
    with h5py.File(file_path, 'r') as f :
        for key in f.keys():
            for sub_key in f[key].keys():
                data = f[key][sub_key]
                print(f"{key}/{sub_key}: {data.shape}")


if __name__ == "__main__" :

    data_dir = "/Users/eunsupark/projects/ap/datasets/original"

    data_list = glob(f"{data_dir}/*.h5")
    print(len(data_list))

    file_path = data_list[0]
    read_h5(file_path)