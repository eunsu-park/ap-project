import datetime
import pandas as pd


SDO = "./sdo_list.csv"
OMNI = "./omni/lowres/omni_lowres_total.csv"

DAYS_BEFORE = 10
DAYS_AFTER = 5

SDO_CADENCE = 6
SDO_HOURS = [0, 6, 12, 18]

OMNI_CADENCE = 3
OMNI_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

IMAGE_SIZE = 64

LEN_SDO_IMAGES = DAYS_BEFORE * (24 // SDO_CADENCE)
LEN_OMNI_VARIABLES = (DAYS_BEFORE + DAYS_AFTER) * (24 // OMNI_CADENCE)


def main(date_target, sdo_df, omni_df):

    sdo_start_time = date_target - datetime.timedelta(days=DAYS_BEFORE)
    sdo_end_time = date_target
    print(f"SDO: {sdo_start_time} - {sdo_end_time}, {LEN_SDO_IMAGES}")

    filtered_sdo_df = sdo_df[(sdo_df['datetime'] >= sdo_start_time) & 
                            (sdo_df['datetime'] < sdo_end_time)]
    filtered_sdo_df = filtered_sdo_df[filtered_sdo_df['hour'].isin(SDO_HOURS)]
    filtered_sdo_df = filtered_sdo_df.sort_values(by='datetime', ascending=True)
    # print(filtered_sdo_df)

    aia_193_list = filtered_sdo_df['aia_193'].tolist()
    if (None in aia_193_list) or (any(pd.isna(val) for val in aia_193_list)) :
        return False
    aia_211_list = filtered_sdo_df['aia_211'].tolist()
    if (None in aia_211_list) or (any(pd.isna(val) for val in aia_211_list)) :
        return False
    hmi_magnetogram_list = filtered_sdo_df['hmi_magnetogram'].tolist()
    if (None in hmi_magnetogram_list) or (any(pd.isna(val) for val in hmi_magnetogram_list)) :
        return False


    omni_start_time = date_target - datetime.timedelta(days=DAYS_BEFORE)
    omni_end_time = date_target + datetime.timedelta(days=DAYS_AFTER)
    print(f"OMNI: {omni_start_time} - {omni_end_time}, {LEN_OMNI_VARIABLES}")

    filtered_omni_df = omni_df[(omni_df['datetime'] >= omni_start_time) & 
                               (omni_df['datetime'] < omni_end_time)]
    filtered_omni_df = filtered_omni_df[filtered_omni_df['Hour'].isin(OMNI_HOURS)]
    filtered_omni_df = filtered_omni_df.sort_values(by='datetime', ascending=True)
    # print(filtered_omni_df)

    column_list = filtered_omni_df.columns.tolist()

    print(column_list)

    column_list.pop(0)
    column_list.pop(0)
    column_list.pop(0)
    column_list.pop(0)

    print(column_list)




    # print(f"원본 SDO 데이터: {len(sdo_df):,}개")
    # print(f"추출된 SDO 데이터: {len(filtered_sdo_df):,}개")

    # print(f"원본 OMNI 데이터: {len(omni_df):,}개")
    # print(f"추출된 OMNI 데이터: {len(filtered_omni_df):,}개")



if __name__ == "__main__" :

    sdo_df = pd.read_csv(SDO)
    sdo_df['datetime'] = pd.to_datetime(sdo_df['datetime'])

    omni_df = pd.read_csv(OMNI)
    omni_df['datetime'] = pd.to_datetime(omni_df['datetime'])

    date_target = datetime.datetime(year=2011, month=10, day=1, hour=0)

    main(date_target, sdo_df, omni_df)

