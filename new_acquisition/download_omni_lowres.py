import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

def parse_fortran_format_lowres(format_string):
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
    URL에서 OMNI .dat 파일 다운로드
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"다운로드 에러: {e}")
        return None

def parse_omni_line_lowres(line, format_parts):
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

def apply_fill_values_lowres(df):
    """
    Fill values를 NaN으로 변환 (OMNI2.text 공식 문서 기준)
    """
    # 각 컬럼별 fill value 정의 (0-based index)
    fill_values = {
        3: [9999],  # Bartels rotation number
        4: [99],  # IMF spacecraft ID
        5: [99],  # SW Plasma spacecraft ID
        6: [999],  # Points in IMF averages
        7: [999],  # Points in Plasma averages
        8: [999.9],  # Field Magnitude Average
        9: [999.9],  # Magnitude of Average Field Vector
        10: [999.9],  # Lat.Angle of Aver. Field Vector
        11: [999.9],  # Long.Angle of Aver.Field Vector
        12: [999.9],  # Bx GSE, GSM
        13: [999.9],  # By GSE
        14: [999.9],  # Bz GSE
        15: [999.9],  # By GSM
        16: [999.9],  # Bz GSM
        17: [999.9],  # sigma|B|
        18: [999.9],  # sigma B
        19: [999.9],  # sigma Bx
        20: [999.9],  # sigma By
        21: [999.9],  # sigma Bz
        22: [9999999., 9999999.0],  # Proton temperature
        23: [999.9],  # Proton Density
        24: [9999., 9999.0],  # Plasma (Flow) speed
        25: [999.9],  # Plasma Flow Long. Angle
        26: [999.9],  # Plasma Flow Lat. Angle
        27: [9.999],  # Na/Np
        28: [99.99],  # Flow Pressure
        29: [9999999., 9999999.0],  # sigma T
        30: [999.9],  # sigma N
        31: [9999., 9999.0],  # sigma V
        32: [999.9],  # sigma phi V
        33: [999.9],  # sigma theta V
        34: [9.999],  # sigma-Na/Np
        35: [999.99],  # Electric field
        36: [999.99],  # Plasma beta
        37: [999.9],  # Alfven mach number
        38: [99],  # Kp
        39: [999],  # R (Sunspot number)
        40: [99999],  # DST Index
        41: [9999],  # AE-index
        42: [999999.99],  # Proton flux >1 Mev
        43: [99999.99],  # Proton flux >2 Mev
        44: [99999.99],  # Proton flux >4 Mev
        45: [99999.99],  # Proton flux >10 Mev
        46: [99999.99],  # Proton flux >30 Mev
        47: [99999.99],  # Proton flux >60 Mev
        48: [],  # Flag (0은 fill이 아니라 실제 값)
        49: [999],  # ap-index
        50: [999.9],  # f10.7_index
        51: [999.9],  # PC(N) index
        52: [99999],  # AL-index
        53: [99999],  # AU-index
        54: [99.9],  # Magnetosonic mach number
    }
    
    # Fill values를 NaN으로 교체
    for col_idx, fill_val_list in fill_values.items():
        if col_idx < len(df.columns) and fill_val_list:
            # 각 fill value를 NaN으로 교체
            for fill_val in fill_val_list:
                df.iloc[:, col_idx] = df.iloc[:, col_idx].replace(fill_val, np.nan)
    
    return df

def create_datetime_lowres(row):
    """
    Year, Decimal_Day, Hour로부터 datetime 생성
    """
    try:
        year = int(row['Year'])
        day_of_year = int(row['Decimal_Day'])
        hour = int(row['Hour'])
        
        # Day of year를 datetime으로 변환
        base_date = datetime(year, 1, 1)
        delta = timedelta(days=day_of_year-1, hours=hour)
        return base_date + delta
    except (ValueError, TypeError):
        # 시간 데이터가 유효하지 않으면 NaT 반환
        return pd.NaT

def download_omni_lowres(url, output_csv, save_dat=True):
    """
    OMNI2 .dat 파일을 다운로드하여 CSV로 변환 (공식 omni2.text 포맷 기준)
    
    Parameters:
    -----------
    url : str
        OMNI2 .dat 파일 URL
    output_csv : str
        출력 CSV 파일 경로
    save_dat : bool
        원본 .dat 파일을 저장할지 여부 (기본값: True)
    """
    # 컬럼명 정의 (55개) - OMNI2.text 공식 문서 기준
    column_names = [
        'Year', 'Decimal_Day', 'Hour',
        'Bartels_Rotation_Number',
        'IMF_SC_ID', 'SW_Plasma_SC_ID',
        'IMF_Avg_Points', 'Plasma_Avg_Points',
        'B_Field_Magnitude_Avg_nT', 'B_Magnitude_of_Avg_Field_Vector_nT',
        'B_Lat_Angle_Avg_Field_Vector_deg', 'B_Long_Angle_Avg_Field_Vector_deg',
        'Bx_GSE_GSM_nT', 'By_GSE_nT', 'Bz_GSE_nT', 'By_GSM_nT', 'Bz_GSM_nT',
        'Sigma_B_Magnitude_nT', 'Sigma_B_Vector_nT',
        'Sigma_Bx_nT', 'Sigma_By_nT', 'Sigma_Bz_nT',
        'Proton_Temperature_K', 'Proton_Density_n_cm3',
        'Plasma_Flow_Speed_km_s',
        'Plasma_Flow_Long_Angle_deg', 'Plasma_Flow_Lat_Angle_deg',
        'Na_Np_Ratio', 'Flow_Pressure_nPa',
        'Sigma_Temperature_K', 'Sigma_Density_n_cm3', 'Sigma_Flow_Speed_km_s',
        'Sigma_Phi_V_deg', 'Sigma_Theta_V_deg', 'Sigma_Na_Np',
        'Electric_Field_mV_m', 'Plasma_Beta', 'Alfven_Mach_Number',
        'Kp_Index', 'Sunspot_Number_R',
        'DST_Index_nT', 'AE_Index_nT',
        'Proton_Flux_gt1MeV', 'Proton_Flux_gt2MeV', 'Proton_Flux_gt4MeV',
        'Proton_Flux_gt10MeV', 'Proton_Flux_gt30MeV', 'Proton_Flux_gt60MeV',
        'Flag',
        'ap_Index_nT', 'f10_7_Index_sfu', 'PC_N_Index',
        'AL_Index_nT', 'AU_Index_nT',
        'Magnetosonic_Mach_Number'
    ]
    
    print(f"다운로드 중: {url}")
    data_text = download_from_url(url)
    
    if data_text is None:
        print("다운로드 실패")
        return False
    
    # .dat 파일 저장
    if save_dat:
        dat_filename = output_csv.replace('.csv', '.dat')
        print(f".dat 파일 저장 중: {dat_filename}")
        try:
            with open(dat_filename, 'w') as f:
                f.write(data_text)
            print(f".dat 파일 저장 완료: {dat_filename}")
        except Exception as e:
            print(f".dat 파일 저장 에러: {e}")
    
    print("데이터 파싱 중...")
    
    # Fortran 포맷 파싱 (OMNI2.text 문서의 공식 포맷)
    # (2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2,F9.0,F6.1,F6.0,2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,F6.1,F6.1,2I6,F5.1)
    format_string = "(2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2,F9.0,F6.1,F6.0,2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,F6.1,F6.1,2I6,F5.1)"
    format_parts = parse_fortran_format_lowres(format_string)
    
    # 각 라인 파싱
    lines = data_text.strip().split('\n')
    data_rows = []
    
    for i, line in enumerate(lines):
        if line.strip():  # 빈 줄 제외
            try:
                values = parse_omni_line_lowres(line, format_parts)
                data_rows.append(values)
            except Exception as e:
                print(f"라인 {i+1} 파싱 에러: {e}")
                continue
    
    print(f"총 {len(data_rows)}개 레코드 파싱 완료")
    
    # DataFrame 생성
    df = pd.DataFrame(data_rows, columns=column_names)
    
    # 시간 관련 컬럼은 Int64 (Nullable Integer)로 변환
    time_columns = ['Year', 'Decimal_Day', 'Hour']
    for col in time_columns:
        if col in df.columns:
            df[col] = df[col].astype('Int64')
    
    # 다른 정수형 컬럼은 float으로 변환하여 NaN을 안전하게 처리
    other_integer_columns = [
        'Bartels_Rotation_Number',
        'IMF_SC_ID', 'SW_Plasma_SC_ID',
        'IMF_Avg_Points', 'Plasma_Avg_Points',
        'Kp_Index', 'Sunspot_Number_R',
        'DST_Index_nT', 'AE_Index_nT',
        'Flag', 'ap_Index_nT',
        'AL_Index_nT', 'AU_Index_nT'
    ]
    
    for col in other_integer_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # Fill values 처리 (NaN으로 변환)
    print("Fill values를 NaN으로 처리 중...")
    df = apply_fill_values_lowres(df)
    
    # Datetime 컬럼 생성 및 맨 앞에 추가
    print("Datetime 컬럼 생성 중...")
    df['datetime'] = df.apply(create_datetime_lowres, axis=1)
    
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
    total_file = f"/Users/eunsupark/Data/omni/lowres/omni_lowres_total.csv"

    for year in range(2010, 2025):
        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
        output_file = f"/Users/eunsupark/Data/omni/lowres/omni_lowres_{year}.csv"
        file_list.append(output_file)
        download_omni_lowres(url, output_file)
        print(f"{year}년 완료\n")

    df_list = []
    for file_path in file_list :
        df = pd.read_csv(file_path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(total_file, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
