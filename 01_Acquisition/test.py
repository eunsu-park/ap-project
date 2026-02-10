import h5py
import numpy as np
import matplotlib.pyplot as plt

filepath = '/Volumes/AP-PROJECT/datasets/original/2011010100.h5'

with h5py.File(filepath, 'r') as f:
    # 1. 파일 속성 확인
    print("=== File Attributes ===")
    print(f"Timestamp: {f.attrs['timestamp']}")
    
    # 2. OMNI 그룹 확인
    print("\n=== OMNI Data ===")
    print(f"Keys: {list(f['omni'].keys())}")
    for key in f['omni'].keys():
        print(f"  {key}: {f['omni'][key][()]}")
    
    # 3. SDO 그룹 확인
    print("\n=== SDO Data ===")
    for channel in f['sdo'].keys():
        ds = f['sdo'][channel]
        print(f"{channel}:")
        print(f"  Shape: {ds.shape}")
        print(f"  Dtype: {ds.dtype}")
        print(f"  Compression: {ds.compression}")
        print(f"  Min/Max: {ds[:].min()} / {ds[:].max()}")

        plt.imshow(ds, cmap="gray")
        plt.show()