import os
import datetime

PYTHON_PATH = "/Users/eunsupark/Softwares/miniconda3/envs/ap-backend/bin/python"
SCRIPT_PATH = "scripts/download_lasco.py"

step = 27
n = 0

start_date = datetime.datetime(2010, 1, 1)
# start_date = datetime.datetime(1996, 1, 1)
while start_date < datetime.datetime(2026, 1, 1) :

    end_date = start_date + datetime.timedelta(days = step)
    
    command = []
    command.append(f"{PYTHON_PATH} {SCRIPT_PATH}")
    command.append(f"--vso")
    command.append(f"--cameras c2 c3")
    command.append(f"--start-date {start_date.strftime("%Y-%m-%d")} --end-date {end_date.strftime("%Y-%m-%d")}")

    command = ' '.join(command)
    print(command)
    os.system(command)

    start_date += datetime.timedelta(days=step+1)
    n += 1

