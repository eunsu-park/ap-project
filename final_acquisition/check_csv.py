import os
import datetime
import pandas as pd

DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
CSV_PATH = f"{DATA_ROOT}/sdo_fits.csv"



df = pd.read_csv(CSV_PATH, parse_dates=['datetime'])

date = datetime.datetime(
    year = 2025,
    month = 1,
    day = 1,
    hour = 0,
)

date_end = datetime.datetime(
    year = 2025,
    month = 10,
    day = 1,
    hour = 0,
)

num_total = 0
num_aia_193 = 0
num_aia_211 = 0
num_hmi_magnetogram = 0


while date < date_end :


    num_total += 1
    mask = df['datetime'] == date

    if mask.any() :
        row = df.loc[mask].iloc[0]
        aia_193 = row['aia_193']
        aia_211 = row['aia_211']
        hmi_magnetogram = row['hmi_magnetogram']

        if not pd.isna(aia_193) :
            file_path = f"{DATA_ROOT}/aia_tmp/{aia_193}"
            if os.path.exists(file_path) is True :
                num_aia_193 += 1
                save_path = f"{DATA_ROOT}/aia/"
                os.system(f"mv {file_path} {save_path}")

        if not pd.isna(aia_211) :
            file_path = f"{DATA_ROOT}/aia_tmp/{aia_211}"
            if os.path.exists(file_path) is True :
                num_aia_211 += 1
                save_path = f"{DATA_ROOT}/aia/"
                os.system(f"mv {file_path} {save_path}")

        if not pd.isna(hmi_magnetogram) :
            file_path = f"{DATA_ROOT}/hmi_tmp/{hmi_magnetogram}"
            if os.path.exists(file_path) is True :
                num_hmi_magnetogram += 1
                save_path = f"{DATA_ROOT}/hmi/"
                os.system(f"mv {file_path} {save_path}")

    date += datetime.timedelta(hours=1)
    
ratio_aia_193 = 100 * float(num_aia_193)/float(num_total)
ratio_aia_211 = 100 * float(num_aia_211)/float(num_total)
ratio_hmi_magnetogram = 100 * float(num_hmi_magnetogram)/float(num_total)

print(f"{num_aia_193} ({ratio_aia_193:.2f} %) {num_aia_211} ({ratio_aia_211:.2f} %) {num_hmi_magnetogram} ({ratio_hmi_magnetogram:.2f} %)")