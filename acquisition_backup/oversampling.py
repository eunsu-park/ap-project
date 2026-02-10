import os
import shutil

import h5py
import numpy as np
import pandas as pd
from PIL import Image


ROOT = '/opt/projects/ap/datasets'

CSV_PATH = f"{ROOT}/three_twelve_to_one_train.csv"
ORIGINAL_DIR = f"{ROOT}/original"
OVERSAMPLING_DIR = f"{ROOT}/oversampling"

# 회전할 이미지 dataset 이름들
IMAGE_KEYS = ("sdo/aia_193", "sdo/aia_211", "sdo/hmi_magnetogram")

# class_day1 == 1 인 경우 생성할 suffix와 각도 (bilinear)
# 0.5도 간격, 총 13개(_0~_12)
ANGLE_MAP = {
    "_0":  0.0,
    "_1":  0.5,
    "_2": -0.5,
    "_3":  1.0,
    "_4": -1.0,
    "_5":  1.5,
    "_6": -1.5,
    "_7":  2.0,
    "_8": -2.0,
    "_9":  2.5,
    "_10":-2.5,
    "_11": 3.0,
    "_12":-3.0,
}


def rotate_slice_bilinear(slice_2d: np.ndarray, angle_deg: float) -> np.ndarray:
    """2D 이미지 하나를 중심 기준으로 bilinear 회전."""
    if angle_deg == 0:
        return slice_2d.copy()

    arr = slice_2d.astype(np.float32)
    pil_img = Image.fromarray(arr, mode="F")

    rotated = pil_img.rotate(angle_deg, resample=Image.BILINEAR)
    rotated_arr = np.array(rotated, dtype=np.float32)

    return rotated_arr.astype(slice_2d.dtype)


def rotate_stack(data: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    sdo_* 데이터셋 전체(40프레임)를 angle_deg만큼 회전.
    shape: (T, 1, H, W) 또는 (T, H, W)
    """
    if angle_deg == 0:
        return data.copy()

    data = np.asarray(data)
    rotated = np.empty_like(data)

    if data.ndim == 4:
        # (T, C, H, W)
        T, C, H, W = data.shape
        for t in range(T):
            for c in range(C):
                rotated[t, c] = rotate_slice_bilinear(data[t, c], angle_deg)

    elif data.ndim == 3:
        # (T, H, W)
        T, H, W = data.shape
        for t in range(T):
            rotated[t] = rotate_slice_bilinear(data[t], angle_deg)

    else:
        raise ValueError(f"Unexpected image data shape: {data.shape}")

    return rotated


def copy_h5_with_rotation(src_path: str, dst_path: str, angle_deg: float):
    """
    h5 파일을 복사하며, 이미지 3개(sdo/* 데이터셋)에만 회전 적용.
    나머지 omni/* 데이터는 그대로 복사.
    그룹 구조를 유지하며 복사.
    """
    def copy_group_recursive(src_group, dst_group, path_prefix=""):
        """재귀적으로 그룹과 데이터셋을 복사"""
        # 그룹 attributes 복사
        for k, v in src_group.attrs.items():
            dst_group.attrs[k] = v
        
        for name, item in src_group.items():
            current_path = f"{path_prefix}/{name}" if path_prefix else name
            
            if isinstance(item, h5py.Group):
                # 하위 그룹 생성 및 재귀 호출
                sub_group = dst_group.create_group(name)
                copy_group_recursive(item, sub_group, current_path)
            
            elif isinstance(item, h5py.Dataset):
                # 데이터셋 처리
                data = item[()]
                
                # 이미지 데이터셋인 경우 회전 적용
                if current_path in IMAGE_KEYS and angle_deg != 0.0:
                    data_to_save = rotate_stack(data, angle_deg)
                else:
                    data_to_save = data
                
                # 데이터셋 생성
                d_out = dst_group.create_dataset(
                    name,
                    data=data_to_save,
                    dtype=data_to_save.dtype,
                )
                
                # 데이터셋 attributes 복사
                for ak, av in item.attrs.items():
                    d_out.attrs[ak] = av
    
    with h5py.File(src_path, "r") as f_in:
        with h5py.File(dst_path, "w") as f_out:
            # 파일-level attributes 복사
            for k, v in f_in.attrs.items():
                f_out.attrs[k] = v
            
            # 루트부터 재귀적으로 복사
            copy_group_recursive(f_in, f_out)


def process():
    os.makedirs(OVERSAMPLING_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    err_list = []

    for idx, row in df.iterrows():
        file_name = str(row["file_name"])
        class_day1 = int(row["class_day1"])

        src_path = os.path.join(ORIGINAL_DIR, file_name)

        if not os.path.exists(src_path):
            print(f"[WARN] Missing file: {src_path}")
            err_list.append(src_path)
            continue

        # Negative class → 1개만 복사
        if class_day1 == 0:
            # dst_path = os.path.join(OVERSAMPLING_DIR, file_name)
            # shutil.copy2(src_path, dst_path)
            # print(f"[NEG] copy   {file_name}")
            continue

        # Positive class → 13개 생성
        base, ext = os.path.splitext(file_name)

        for suffix, angle in ANGLE_MAP.items():
            new_name = f"{base}{suffix}{ext}"
            dst_path = os.path.join(OVERSAMPLING_DIR, new_name)

            if angle == 0.0:
                shutil.copy2(src_path, dst_path)
                print(f"[POS] copy   {file_name} → {new_name}")
            else:
                copy_h5_with_rotation(src_path, dst_path, angle)
                print(f"[POS] rotate {file_name} → {new_name} ({angle}°)")


if __name__ == "__main__":
    process()

# import os
# import shutil

# import h5py
# import numpy as np
# import pandas as pd
# from PIL import Image


# ROOT = '/opt/projects/ap/datasets'

# CSV_PATH = f"{ROOT}/three_twelve_to_one_train.csv"
# ORIGINAL_DIR = f"{ROOT}/original"
# OVERSAMPLING_DIR = f"{ROOT}/oversampling"

# # 회전할 이미지 dataset 이름들
# IMAGE_KEYS = ("sdo/aia_193", "sdo/aia_211", "sdo/hmi_magnetogram")

# # class_day1 == 1 인 경우 생성할 suffix와 각도 (bilinear)
# # 0.5도 간격, 총 13개(_0~_12)
# ANGLE_MAP = {
#     "_0":  0.0,
#     "_1":  0.5,
#     "_2": -0.5,
#     "_3":  1.0,
#     "_4": -1.0,
#     "_5":  1.5,
#     "_6": -1.5,
#     "_7":  2.0,
#     "_8": -2.0,
#     "_9":  2.5,
#     "_10":-2.5,
#     "_11": 3.0,
#     "_12":-3.0,
# }


# def rotate_slice_bilinear(slice_2d: np.ndarray, angle_deg: float) -> np.ndarray:
#     """2D 이미지 하나를 중심 기준으로 bilinear 회전."""
#     if angle_deg == 0:
#         return slice_2d.copy()

#     arr = slice_2d.astype(np.float32)
#     pil_img = Image.fromarray(arr, mode="F")

#     rotated = pil_img.rotate(angle_deg, resample=Image.BILINEAR)
#     rotated_arr = np.array(rotated, dtype=np.float32)

#     return rotated_arr.astype(slice_2d.dtype)


# def rotate_stack(data: np.ndarray, angle_deg: float) -> np.ndarray:
#     """
#     sdo_* 데이터셋 전체(40프레임)를 angle_deg만큼 회전.
#     shape: (T, 1, H, W) 또는 (T, H, W)
#     """
#     if angle_deg == 0:
#         return data.copy()

#     data = np.asarray(data)
#     rotated = np.empty_like(data)

#     if data.ndim == 4:
#         # (T, C, H, W)
#         T, C, H, W = data.shape
#         for t in range(T):
#             for c in range(C):
#                 rotated[t, c] = rotate_slice_bilinear(data[t, c], angle_deg)

#     elif data.ndim == 3:
#         # (T, H, W)
#         T, H, W = data.shape
#         for t in range(T):
#             rotated[t] = rotate_slice_bilinear(data[t], angle_deg)

#     else:
#         raise ValueError(f"Unexpected image data shape: {data.shape}")

#     return rotated


# def copy_h5_with_rotation(src_path: str, dst_path: str, angle_deg: float):
#     """
#     h5 파일을 복사하며, 이미지 3개(sdo_* 데이터셋)에만 회전 적용.
#     나머지 omni_* 데이터는 그대로 복사.
#     """
#     with h5py.File(src_path, "r") as f_in:

#         # 데이터 로딩 및 attr 저장
#         data_dict = {}
#         dset_attrs = {}
#         for name, dset in f_in.items():
#             breakpoint()
#             data_dict[name] = dset[()]
#             dset_attrs[name] = dict(dset.attrs)

#         file_attrs = dict(f_in.attrs)

#         # 새 h5 생성
#         with h5py.File(dst_path, "w") as f_out:
#             # 파일-level attrs 복사
#             for k, v in file_attrs.items():
#                 f_out.attrs[k] = v

#             # dataset 생성
#             for name, data in data_dict.items():

#                 if name in IMAGE_KEYS and angle_deg != 0.0:
#                     data_to_save = rotate_stack(data, angle_deg)
#                 else:
#                     data_to_save = data

#                 d_out = f_out.create_dataset(
#                     name,
#                     data=data_to_save,
#                     dtype=data_to_save.dtype,
#                 )

#                 # 데이터셋 attr 복사
#                 for ak, av in dset_attrs[name].items():
#                     d_out.attrs[ak] = av


# def process():
#     os.makedirs(OVERSAMPLING_DIR, exist_ok=True)

#     df = pd.read_csv(CSV_PATH)

#     err_list = []

#     for idx, row in df.iterrows():
#         file_name = str(row["file_name"])
#         class_day1 = int(row["class_day1"])

#         src_path = os.path.join(ORIGINAL_DIR, file_name)

#         if not os.path.exists(src_path):
#             print(f"[WARN] Missing file: {src_path}")
#             err_list.append(src_path)
#             continue

#         # Negative class → 1개만 복사
#         if class_day1 == 0:
#             # dst_path = os.path.join(OVERSAMPLING_DIR, file_name)
#             # shutil.copy2(src_path, dst_path)
#             # print(f"[NEG] copy   {file_name}")
#             continue

#         # Positive class → 13개 생성
#         base, ext = os.path.splitext(file_name)

#         for suffix, angle in ANGLE_MAP.items():
#             new_name = f"{base}{suffix}{ext}"
#             dst_path = os.path.join(OVERSAMPLING_DIR, new_name)

#             if angle == 0.0:
#                 shutil.copy2(src_path, dst_path)
#                 print(f"[POS] copy   {file_name} → {new_name}")
#             else:
#                 copy_h5_with_rotation(src_path, dst_path, angle)
#                 print(f"[POS] rotate {file_name} → {new_name} ({angle}°)")


# if __name__ == "__main__":
#     process()