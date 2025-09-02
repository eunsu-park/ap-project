import os
import sys
import datetime
from glob import glob
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool


def jp2_to_array(jp2_path):
    with Image.open(jp2_path) as img:
        array = np.array(img)
    array = np.expand_dims(array, 0)
    array = np.expand_dims(array, 0)
    return array

def main(load_dir, save_root):
    if not os.path.exists(load_dir):
        print(f"경고: 디렉토리가 없습니다: {load_dir}")
        return
    
    if not os.path.isdir(load_dir):
        print(f"경고: 디렉토리가 아닙니다: {load_dir}")
        return

    dir_name = os.path.basename(load_dir)
    if not dir_name.startswith("19") and not dir_name.startswith("20") :
        print(f"경고: 잘못된 디렉토리 이름: {dir_name}")
        return

    start_date = datetime.datetime.strptime(dir_name.split('-')[0], "%Y%m%d%H")
    hour = start_date.hour
    if hour % 3 != 0 :
        print(f"3시간 간격이 아닙니다: {dir_name}")
        return
    
    print(f"Processing dataset: {start_date}")
    
    list_193 = sorted(glob(f"{load_dir}/193/*.jp2"))
    list_211 = sorted(glob(f"{load_dir}/211/*.jp2"))

    if len(list_193) == 0 or len(list_211) == 0 :
        print(f"경고: 193 또는 211 파장 이미지가 없습니다: {load_dir}")
        return
        
    if not len(list_193) == 20 :
        print(f"경고: 193 파장 이미지 개수가 20개가 아닙니다: {len(list_193)}")
        return

    if not len(list_211) == 20 :
        print(f"경고: 211 파장 이미지 개수가 20개가 아닙니다: {len(list_211)}")
        return

    sequence_193 = [jp2_to_array(p) for p in list_193]
    sequence_211 = [jp2_to_array(p) for p in list_211]

    sequence_193 = np.concatenate(sequence_193, axis=0)
    sequence_211 = np.concatenate(sequence_211, axis=0)

    sw_data = pd.read_csv(f"{load_dir}/hourly_data.csv")#, parse_dates=['base_time'], index_col='base_time')

    save_path = f"{save_root}/{start_date:%Y%m%d%H}.h5"
    print(save_path)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('sdo_193', data=sequence_193)#, compression="gzip")
        f.create_dataset('sdo_211', data=sequence_211)#, compression="gzip")
        for col in sw_data.columns:
            f.create_dataset(f"omni_{col}", data=sw_data[col].values)#, compression="gzip")
        


if __name__ == "__main__" :

    list_dataset = glob("/Users/eunsupark/ap_project/data/processed/*")
    nb_dataset = len(list_dataset)
    print(f"총 {nb_dataset}개의 데이터셋이 있습니다.")

    save_root = "/Users/eunsupark/ap_project/dataset"

    with Pool(processes=8) as pool:
        pool.starmap(main, [(d, save_root) for d in list_dataset])

    # f = "/projects/ap/dataset/2011010109.h5"

    # with h5py.File(f, 'r') as f:
    #     print(f.keys())
    #     sdo_193 = f['sdo_193'][:]
    #     sdo_211 = f['sdo_211'][:]
    #     bz_gsm = f['omni_Bz_GSM'][:]
    #     print(bz_gsm)
    #     base_time = f['omni_base_time'][:]
    #     print(base_time)

    #     print(f['sdo_193'].shape)
    #     print(f['sdo_211'].shape)
    #     for key in f.keys():
    #         if key.startswith('omni_'):
    #             print(key, f[key].shape)


    # sdo_193 = np.squeeze(sdo_193)
    # sdo_211 = np.squeeze(sdo_211)

    # sdo_193 = np.hstack([sdo_193[n] for n in range(sdo_193.shape[0])])
    # sdo_211 = np.hstack([sdo_211[n] for n in range(sdo_211.shape[0])])

    # print(sdo_193.shape, sdo_211.shape)

    # from imageio import imsave

    # imsave("/work/sdo_193.png", sdo_193)
    # imsave("/work/sdo_211.png", sdo_211)

    # plt.plot(base_time, bz_gsm)
    # plt.savefig("/work/bz_gsm.png")
    # plt.close()