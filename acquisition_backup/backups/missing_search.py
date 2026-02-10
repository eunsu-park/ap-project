import datetime

import pandas as pd

from utils_database import query_by_exact_time


DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"
COLUMNS = ["instrument", "wavelength", "date"]

table_names = ["aia_193", "aia_211", "hmi_magnetogram"]
missing_data = []

date = datetime.datetime(year=2010, month=9, day=1, hour=0)
date_end = datetime.datetime(year=2025, month=1, day=1)

while date < date_end:
    for table_name in table_names:
        df = query_by_exact_time(table_name, date, return_type='dataframe')
        
        if df is None or len(df) == 0:
            instrument, wavelength = table_name.split('_', 1)  # 'hmi_magnetogram' 처리
            
            missing_dict = {
                "instrument": instrument,
                "wavelength": wavelength,
                "date": date
            }
            
            missing_data.append(missing_dict)  # 딕셔너리만 추가
            print(missing_dict)
        
    date += datetime.timedelta(hours=1)


if len(missing_data) > 0:
    missing_df = pd.DataFrame(missing_data)
    missing_df = missing_df.sort_values(by='date', ascending=True)  # 'datetime'이 아닌 'date'
    missing_df.to_csv(f"{DATA_ROOT}/missing.csv", index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
    print("missing.csv 저장 완료!")
    print(f"\n처음 10개:")
    print(missing_df.head(10))
else:
    print("누락된 데이터가 없습니다!")