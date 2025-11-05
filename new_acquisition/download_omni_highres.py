import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

def parse_fortran_format_highres(format_string):
    """
    Fortran 포맷 문자열을 파싱하여 각 필드의 위치와 길이를 반환
    """
    format_parts = []
    
    # 간단한 파싱: nXw.d 형식 (n=반복횟수, X=타입, w=너비, d=소수점)
    pattern = r'(\d*)([IFA])(\d+)(?:\.(\d+))?'
    
    position = 0
    for match in re.finditer(pattern, format_string):
        repeat = int(match.group(1)) if match.group(1) else 1
        field_type = match.group(2)
        width = int(match.group(3))
        
        for _ in range(repeat):
            format_parts.append({
                'start': position,
                'end': position + width,
                'width': width,
                'type': field_type
            })
            position += width
    
    return format_parts

def download_from_url(url):
    """
    URL에서 OMNI .asc 파일 다운로드
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"다운로드 에러: {e}")
        return None

def parse_omni_line_highres(line, format_parts):
    """
    한 줄의 데이터를 파싱하여 리스트로 반환
    """
    values = []
    for fmt in format_parts:
        try:
            field_str = line[fmt['start']:fmt['end']].strip()
            if field_str:
                if fmt['type'] == 'I':  # Integer
                    values.append(int(field_str))
                else:  # Float
                    values.append(float(field_str))
            else:
                values.append(np.nan)
        except (ValueError, IndexError):
            values.append(np.nan)
    
    return values

def apply_fill_values_highres(df):
    """
    Fill values를 NaN으로 변환 (HRO 공식 문서 기준)
    
    공식 문서에는 명시적인 fill value가 없으나, 일반적으로 사용되는 값들:
    - Integer fields: 999, 9999, 99999, 999999, 9999999
    - Float fields: 99.99, 999.9, 9999.99, etc.
    """
    # 각 컬럼별 fill value 정의 (0-based index)
    fill_values = {
        4: [99],  # IMF Spacecraft ID
        5: [99],  # SW Plasma Spacecraft ID
        6: [999],  # Points in IMF averages
        7: [999],  # Points in Plasma averages
        8: [999],  # Percent interp
        9: [999999, 9999999],  # Timeshift, sec
        10: [999999, 9999999],  # RMS Timeshift
        11: [99.99],  # RMS Phase front normal
        12: [999999, 9999999, 9999],  # Time between observations
        13: [9999.99],  # B magnitude
        14: [9999.99],  # Bx GSE
        15: [9999.99],  # By GSE
        16: [9999.99],  # Bz GSE
        17: [9999.99],  # By GSM
        18: [9999.99],  # Bz GSM
        19: [9999.99],  # RMS SD B scalar
        20: [9999.99],  # RMS SD B vector
        21: [99999.9, 99999.99],  # Flow speed
        22: [99999.9, 99999.99],  # Vx GSE
        23: [99999.9, 99999.99],  # Vy GSE
        24: [99999.9, 99999.99],  # Vz GSE
        25: [999.99],  # Proton density
        26: [9999999., 9999999.0],  # Temperature
        27: [99.99],  # Flow pressure
        28: [999.99],  # Electric field
        29: [999.99],  # Plasma beta
        30: [999.9],  # Alfven mach number
        31: [9999.99],  # S/C X GSE
        32: [9999.99],  # S/C Y GSE
        33: [9999.99],  # S/C Z GSE
        34: [9999.99],  # BSN X GSE
        35: [9999.99],  # BSN Y GSE
        36: [9999.99],  # BSN Z GSE
        37: [99999, 999999],  # AE index
        38: [99999, 999999],  # AL index
        39: [99999, 999999],  # AU index
        40: [99999, 999999],  # SYM/D index
        41: [99999, 999999],  # SYM/H index
        42: [99999, 999999],  # ASY/D index
        43: [99999, 999999],  # ASY/H index
        44: [999.99],  # PC(N) index
        45: [99.9],  # Magnetosonic mach number
        # 5-min 데이터의 경우 추가 필드
        46: [99999.99],  # Proton flux >10 MeV (optional for 5-min)
        47: [99999.99],  # Proton flux >30 MeV (optional for 5-min)
        48: [99999.99],  # Proton flux >60 MeV (optional for 5-min)
    }
    
    # Fill values를 NaN으로 교체
    for col_idx, fill_val_list in fill_values.items():
        if col_idx < len(df.columns):
            # 각 fill value를 NaN으로 교체
            for fill_val in fill_val_list:
                df.iloc[:, col_idx] = df.iloc[:, col_idx].replace(fill_val, np.nan)
    
    return df

def create_datetime_highres(row):
    """
    Year, Day, Hour, Minute으로부터 datetime 생성
    """
    try:
        year = int(row['Year'])
        day_of_year = int(row['Day'])
        hour = int(row['Hour'])
        minute = int(row['Minute'])
        
        # Day of year를 datetime으로 변환
        base_date = datetime(year, 1, 1)
        delta = timedelta(days=day_of_year-1, hours=hour, minutes=minute)
        return base_date + delta
    except (ValueError, TypeError):
        # 시간 데이터가 유효하지 않으면 NaT 반환
        return pd.NaT

def download_omni_highres(url, output_csv, save_asc=True, is_5min=False):
    """
    OMNI .asc 파일을 다운로드하여 CSV로 변환 (HRO 공식 문서 기준)
    
    Parameters:
    -----------
    url : str
        OMNI .asc 파일 URL
    output_csv : str
        출력 CSV 파일 경로
    save_asc : bool
        원본 .asc 파일을 저장할지 여부 (기본값: True)
    is_5min : bool
        5분 데이터 여부 (5분 데이터는 GOES flux 3개 추가)
    """
    # 컬럼명 정의 (HRO format 기준 - 48개 기본 + 5분 데이터의 경우 3개 추가)
    column_names = [
        'Year', 'Day', 'Hour', 'Minute',
        'IMF_SC_ID', 'SW_Plasma_SC_ID',
        'IMF_Avg_Points', 'Plasma_Avg_Points', 'Percent_Interp',
        'Timeshift_sec', 'RMS_Timeshift', 'RMS_Phase_Front_Normal',
        'Time_Between_Obs_sec',
        'B_Magnitude_nT', 'Bx_GSE_nT', 'By_GSE_nT', 'Bz_GSE_nT',
        'By_GSM_nT', 'Bz_GSM_nT',
        'RMS_SD_B_Scalar_nT', 'RMS_SD_B_Vector_nT',
        'Flow_Speed_km_s',
        'Vx_GSE_km_s', 'Vy_GSE_km_s', 'Vz_GSE_km_s',
        'Proton_Density_n_cc', 'Temperature_K',
        'Flow_Pressure_nPa', 'Electric_Field_mV_m',
        'Plasma_Beta', 'Alfven_Mach_Number',
        'SC_X_GSE_Re', 'SC_Y_GSE_Re', 'SC_Z_GSE_Re',
        'BSN_X_GSE_Re', 'BSN_Y_GSE_Re', 'BSN_Z_GSE_Re',
        'AE_Index_nT', 'AL_Index_nT', 'AU_Index_nT',
        'SYM_D_nT', 'SYM_H_nT', 'ASY_D_nT', 'ASY_H_nT',
        'PC_N_Index', 'Magnetosonic_Mach_Number'
    ]
    
    # 5분 데이터의 경우 GOES flux 추가
    if is_5min:
        column_names.extend([
            'Proton_Flux_gt10MeV',
            'Proton_Flux_gt30MeV',
            'Proton_Flux_gt60MeV'
        ])
    
    print(f"다운로드 중: {url}")
    data_text = download_from_url(url)
    
    if data_text is None:
        print("다운로드 실패")
        return False
    
    # .asc 파일 저장
    if save_asc:
        asc_filename = output_csv.replace('.csv', '.asc')
        print(f".asc 파일 저장 중: {asc_filename}")
        try:
            with open(asc_filename, 'w') as f:
                f.write(data_text)
            print(f".asc 파일 저장 완료: {asc_filename}")
        except Exception as e:
            print(f".asc 파일 저장 에러: {e}")
    
    print("데이터 파싱 중...")
    
    # Fortran 포맷 파싱 (HRO 공식 문서)
    # 1-min: (2I4,4I3,3I4,2I7,F6.2,I7, 8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.2, F5.1)
    # 5-min: 위 포맷 + 3F9.2 (GOES flux)
    if is_5min:
        format_string = "(2I4,4I3,3I4,2I7,F6.2,I7, 8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.2, F5.1,3F9.2)"
    else:
        format_string = "(2I4,4I3,3I4,2I7,F6.2,I7, 8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.2, F5.1)"
    
    format_parts = parse_fortran_format_highres(format_string)
    
    # 각 라인 파싱
    lines = data_text.strip().split('\n')
    data_rows = []
    
    for i, line in enumerate(lines):
        if line.strip():  # 빈 줄 제외
            try:
                values = parse_omni_line_highres(line, format_parts)
                data_rows.append(values)
            except Exception as e:
                print(f"라인 {i+1} 파싱 에러: {e}")
                continue
    
    print(f"총 {len(data_rows)}개 레코드 파싱 완료")
    
    # DataFrame 생성
    df = pd.DataFrame(data_rows, columns=column_names)
    
    # 시간 관련 컬럼은 Int64 (Nullable Integer)로 변환
    time_columns = ['Year', 'Day', 'Hour', 'Minute']
    for col in time_columns:
        if col in df.columns:
            df[col] = df[col].astype('Int64')
    
    # 다른 정수형 컬럼은 float으로 변환하여 NaN을 안전하게 처리
    other_integer_columns = [
        'IMF_SC_ID', 'SW_Plasma_SC_ID',
        'IMF_Avg_Points', 'Plasma_Avg_Points', 'Percent_Interp',
        'Timeshift_sec', 'RMS_Timeshift', 'Time_Between_Obs_sec',
        'AE_Index_nT', 'AL_Index_nT', 'AU_Index_nT',
        'SYM_D_nT', 'SYM_H_nT', 'ASY_D_nT', 'ASY_H_nT'
    ]
    
    for col in other_integer_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # Fill values 처리 (NaN으로 변환)
    print("Fill values를 NaN으로 처리 중...")
    df = apply_fill_values_highres(df)
    
    # Datetime 컬럼 생성 및 맨 앞에 추가
    print("Datetime 컬럼 생성 중...")
    df['datetime'] = df.apply(create_datetime_highres, axis=1)
    
    # datetime 컬럼을 맨 앞으로 이동
    cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
    df = df[cols]
    
    # CSV 저장
    print(f"CSV 파일 저장 중: {output_csv}")
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
        print(f"성공적으로 저장됨: {output_csv}")
        print(f"데이터 shape: {df.shape}")
        return True
    except Exception as e:
        print(f"CSV 저장 에러: {e}")
        return False


if __name__ == "__main__":
    file_list = []
    total_file = f"/Users/eunsupark/Data/omni/highres/omni_highres_total.csv"

    for year in range(2010, 2025):
        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_min{year}.asc"
        output_file = f"/Users/eunsupark/Data/omni/highres/omni_{year}.csv"
        file_list.append(output_file)
        download_omni_highres(url, output_file, is_5min=False)
        print(f"{year}년 완료\n")
    
    df_list = []
    for file_path in file_list :
        df = pd.read_csv(file_path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(total_file, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')

    # 5분 데이터 다운로드 예시 (GOES flux 포함)
    # for year in range(2010, 2025):
    #     url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min{year}.asc"
    #     output_file = f"omni_5min_{year}.csv"
    #     download_omni_highres(url, output_file, is_5min=True)
    #     print(f"{year}년 완료\n")
