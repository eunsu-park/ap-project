import os
import time
from glob import glob


DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
CSV_PATH = f"{DATA_ROOT}/sdo_fits.csv"


while True :
    os.system("clear")
    print(f"AIA TMP         : {len(glob(f"{DATA_ROOT}/aia_tmp/*.fits")) + len(glob(f"{DATA_ROOT}/aia_tmp2/*.fits")):6d}")
    print(f"AIA             : {len(glob(f"{DATA_ROOT}/aia/*.fits")):6d}")
    print(f"HMI TMP         : {len(glob(f"{DATA_ROOT}/hmi_tmp/*.fits")) + len(glob(f"{DATA_ROOT}/hmi_tmp2/*.fits")):6d}")
    print(f"HMI             : {len(glob(f"{DATA_ROOT}/hmi/*.fits")):6d}")
    print(f"INVALID FILE    : {len(glob(f"{DATA_ROOT}/invalid_file/*.fits")):6d}")
    print(f"INVALID HEADER. : {len(glob(f"{DATA_ROOT}/invalid_header/*.fits")):6d}")
    print(f"NON ZERO QUALITY: {len(glob(f"{DATA_ROOT}/non_zero_quality/*.fits")):6d}")
    time.sleep(60)
    