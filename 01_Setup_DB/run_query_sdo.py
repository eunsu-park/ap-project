import os
import datetime

PYTHON_PATH = "/Users/eunsupark/Softwares/miniconda3/envs/ap-backend/bin/python"
SCRIPT_PATH = "scripts/query_sdo.py"
JSON_DIR = "./json"



start_date = datetime.datetime(2024, 1, 1, 0, 0, 0)
while start_date < datetime.datetime(2025, 1, 1, 0, 0, 0):
    command = []
    command.append(f"{PYTHON_PATH} {SCRIPT_PATH}")
    command.append(f"--instruments aia_193 aia_211 --email eunsupark@kasi.re.kr")
    # command.append(f"--instruments hmi_magnetogram --email harim.lee@njit.edu")
    command.append(f'--target-time "{start_date.strftime("%Y-%m-%d %H:%M:%S")}"')
    command.append(f'--output {JSON_DIR}/{start_date:%Y%m%d%H%M%S}.json')
    command.append(f'--time-range 12')
    command = ' '.join(command)
    print(command)
    os.system(command)
    start_date += datetime.timedelta(hours=1)
