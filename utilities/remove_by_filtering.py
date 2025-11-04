import os
from glob import glob
import datetime

LOAD_ROOT = "./sdo_jp2"
HOUR_FILTER = (2,3, 5,6, 8,9, 11,12, 14,15, 17,18, 20,21, 23,0)


def hour_filter(file_path, instrument, wavelength):
    instrument_upper = instrument.upper()
    file_format = f"%Y_%m_%d__%H_%M_%S_%f__SDO_{instrument_upper}_{instrument_upper}_{wavelength}.jp2"
    file_name = os.path.basename(file_path)
    file_date = datetime.datetime.strptime(file_name, file_format)
    file_hour = file_date.hour
    if file_hour in HOUR_FILTER :
        return False
    else :
        return True

def delete_with_filter(file_path, instrument, wavelength):
    if hour_filter(file_path, instrument, wavelength) is True :
        os.system(f"rm -rf {file_path}")
        return True
    return False
        

def main(date, instrument, wavelength):
    instrument_upper = instrument.upper()
    date_dir = f"{LOAD_ROOT}/{instrument}/{wavelength}/{date:%Y}/{date:%Y%m%d}"
    pattern = f"*__SDO_{instrument_upper}_{instrument_upper}_{wavelength}.jp2"
    file_list = glob(f"{date_dir}/{pattern}")
    if len(file_list) == 0:
        return
    
    N = 0 # For test only
    for file_path in file_list :
        is_deleted = delete_with_filter(file_path, instrument, wavelength)
        if is_deleted is True :
            print(f"{file_path} is deleted")
        N += 1 # For test only
        # if N == 100 : # For test only
        #     break # For test only

if __name__ == "__main__" :


    date = datetime.datetime(year=2010, month=9, day=1)
    date_end = datetime.datetime(year=2025, month=1, day=1)
    while date < date_end :

        print(date)

        main(date, 'aia', 193)
        main(date, 'aia', 211)
        main(date, 'hmi', 'magnetogram')

        date += datetime.timedelta(days=1)


