import datetime

import pandas as pd


DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"


df = pd.read_csv(f"{DATA_ROOT}/missing.csv")

print(len(df))

