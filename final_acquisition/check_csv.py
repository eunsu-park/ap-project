import os
import datetime


CSV_DIR = "/Users/eunsupark/JSOC"



date = datetime.datetime(
    year=2010,
    month=9,
    day=1    
)

date_end = datetime.datetime(
    year=2025,
    month=1,
    day=1
)


while date < date_end :

    aia_csv = f"{CSV_DIR}/{date:%Y%m%d}_aia.csv"
    hmi_csv = f"{CSV_DIR}/{date:%Y%m%d}_hmi.csv"

    if os.path.exists(aia_csv) is False :
        print(f"{date} : missing aia")
    
    if os.path.exists(hmi_csv) is False :
        print(f"{date} : missing hmi")

    date += datetime.timedelta(days=7)