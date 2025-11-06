import os
import time
import random
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import cv2


LOAD_ROOT = "ap-data/sdo_jp2"
MAX_WORKERS = 8
IMAGE_SIZES = (64, 128, 256, 512, 1024, 2048)


def resize_image_optimized(img_array, target_size=(64, 64)):
    """이미지 리사이징 최적화 - OpenCV 사용"""
    if img_array is None:
        return None
    
    # OpenCV resize가 bin_ndarray보다 빠름 (네이티브 최적화)
    # INTER_AREA는 다운샘플링에 최적
    img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
    return np.clip(img_resized, 0, 255).astype(np.uint8)


def main(file_path):

    img_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img_array is None:
        return 

    for image_size in IMAGE_SIZES :
        save_root = f"{LOAD_ROOT}_{image_size}"
        save_path = file_path.replace(LOAD_ROOT, save_root)

        if os.path.exists(save_path):
            pass

        img_resized = resize_image_optimized(img_array, (image_size, image_size))
        cv2.imwrite(save_path, img_resized, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000])


if __name__ == "__main__" :

    file_list = glob(f"{LOAD_ROOT}/*/*/*.jp2")
    file_num = len(file_list)
    print(file_num)

    for image_size in IMAGE_SIZES :
        tmp_path = f"ap-data/sdo_jp2_{image_size}"
        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(f"{tmp_path}/aia", exist_ok=True)
        os.makedirs(f"{tmp_path}/aia/193", exist_ok=True)
        os.makedirs(f"{tmp_path}/aia/211", exist_ok=True)
        os.makedirs(f"{tmp_path}/hmi", exist_ok=True)
        os.makedirs(f"{tmp_path}/hmi/magnetogram", exist_ok=True)

    random.shuffle(file_list)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_date = {executor.submit(main, file_path): file_path for file_path in file_list}



    # N = 0
    # for file_path in file_list :
    #     main(file_path)
    #     N += 1
    #     if N == 10 :
    #         break



