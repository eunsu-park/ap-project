import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

def extract_time_range_highres(center_datetime, hours_before, hours_after, 
                                data_dir, output_file):
    """
    고해상도 OMNI 데이터에서 특정 datetime 기준으로 시간 범위 데이터 추출
    1분 데이터를 시간 단위로 평균하여 반환
    
    Parameters:
    -----------
    center_datetime : str or datetime
        기준 datetime (예: '2024-01-01 12:00:00')
    hours_before : int
        기준 시간 이전 몇 시간
    hours_after : int
        기준 시간 이후 몇 시간
    data_dir : str
        CSV 파일들이 있는 디렉토리
    output_file : str
        출력 CSV 파일명
    
    Returns:
    --------
    pd.DataFrame : 추출된 데이터프레임
    """
    # datetime 변환
    if isinstance(center_datetime, str):
        center_dt = pd.to_datetime(center_datetime)
    else:
        center_dt = center_datetime
    
    # 시간 범위 계산
    start_dt = center_dt - timedelta(hours=hours_before)
    end_dt = center_dt + timedelta(hours=hours_after)
    
    print(f"기준 시간: {center_dt}")
    print(f"추출 범위: {start_dt} ~ {end_dt}")
    print(f"총 시간: {hours_before + hours_after + 1}시간")
    
    # 필요한 연도 파일 목록 생성
    years = set()
    current_dt = start_dt
    while current_dt <= end_dt:
        years.add(current_dt.year)
        current_dt += timedelta(days=1)
    
    print(f"필요한 연도: {sorted(years)}")
    
    # 데이터 로드
    all_data = []
    for year in sorted(years):
        # 파일 패턴 검색 (omni_YYYY.csv 또는 omni_min_YYYY.csv 등)
        file_patterns = [
            os.path.join(data_dir, f'omni_{year}.csv'),
            os.path.join(data_dir, f'omni_min{year}.csv'),
            os.path.join(data_dir, f'omni_min_{year}.csv'),
            os.path.join(data_dir, f'omni_highres_{year}.csv')
        ]
        
        file_found = False
        for pattern in file_patterns:
            if os.path.exists(pattern):
                print(f"로딩 중: {pattern}")
                df_year = pd.read_csv(pattern, parse_dates=['datetime'])
                all_data.append(df_year)
                file_found = True
                break
        
        if not file_found:
            print(f"경고: {year}년 파일을 찾을 수 없습니다.")
    
    if not all_data:
        raise FileNotFoundError("데이터 파일을 찾을 수 없습니다.")
    
    # 모든 데이터 병합
    df = pd.concat(all_data, ignore_index=True)
    print(f"전체 데이터 로드 완료: {len(df)} 레코드")
    
    # 시간 범위 필터링
    df_filtered = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
    print(f"필터링 후: {len(df_filtered)} 레코드")
    
    if len(df_filtered) == 0:
        print("경고: 해당 시간 범위에 데이터가 없습니다.")
        return None
    
    # 시간별로 평균 계산 (1분 데이터 → 1시간 데이터)
    print("시간별 평균 계산 중...")
    
    # datetime을 시간 단위로 변환 (분/초는 0으로)
    df_filtered['datetime_hour'] = df_filtered['datetime'].dt.floor('H')
    
    # 평균 계산할 컬럼 (시간 관련 컬럼 제외)
    exclude_cols = ['datetime', 'datetime_hour', 'Year', 'Day', 'Hour', 'Minute']
    numeric_cols = [col for col in df_filtered.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_filtered[col])]
    
    # 시간별 그룹화 및 평균 (NaN 제외)
    df_hourly = df_filtered.groupby('datetime_hour')[numeric_cols].mean()
    df_hourly = df_hourly.reset_index()
    df_hourly.rename(columns={'datetime_hour': 'datetime'}, inplace=True)
    
    # Year, Day, Hour 재생성
    df_hourly['Year'] = df_hourly['datetime'].dt.year
    df_hourly['Day'] = df_hourly['datetime'].dt.dayofyear
    df_hourly['Hour'] = df_hourly['datetime'].dt.hour
    
    # 컬럼 순서 재정렬 (datetime을 맨 앞으로)
    cols = ['datetime', 'Year', 'Day', 'Hour'] + [col for col in df_hourly.columns 
                                                    if col not in ['datetime', 'Year', 'Day', 'Hour']]
    df_hourly = df_hourly[cols]
    
    print(f"시간별 평균 완료: {len(df_hourly)} 시간")
    
    # CSV 저장
    df_hourly.to_csv(output_file, index=False)
    print(f"저장 완료: {output_file}")
    
    return df_hourly

def extract_time_range_lowres(center_datetime, hours_before, hours_after, 
                               data_dir, output_file):
    """
    저해상도 OMNI2 Extended 데이터에서 특정 datetime 기준으로 시간 범위 데이터 추출
    
    Parameters:
    -----------
    center_datetime : str or datetime
        기준 datetime (예: '2024-01-01 12:00:00')
    hours_before : int
        기준 시간 이전 몇 시간
    hours_after : int
        기준 시간 이후 몇 시간
    data_dir : str
        CSV 파일들이 있는 디렉토리
    output_file : str
        출력 CSV 파일명
    
    Returns:
    --------
    pd.DataFrame : 추출된 데이터프레임
    """
    # datetime 변환
    if isinstance(center_datetime, str):
        center_dt = pd.to_datetime(center_datetime)
    else:
        center_dt = center_datetime
    
    # 시간 범위 계산
    start_dt = center_dt - timedelta(hours=hours_before)
    end_dt = center_dt + timedelta(hours=hours_after)
    
    print(f"기준 시간: {center_dt}")
    print(f"추출 범위: {start_dt} ~ {end_dt}")
    print(f"총 시간: {hours_before + hours_after + 1}시간")
    
    # 필요한 연도 파일 목록 생성
    years = set()
    current_dt = start_dt
    while current_dt <= end_dt:
        years.add(current_dt.year)
        current_dt += timedelta(days=1)
    
    print(f"필요한 연도: {sorted(years)}")
    
    # 데이터 로드
    all_data = []
    for year in sorted(years):
        # 파일 패턴 검색 (omni2_YYYY.csv, omni2_extended_YYYY.csv 등)
        file_patterns = [
            os.path.join(data_dir, f'omni2_{year}.csv'),
            os.path.join(data_dir, f'omni2_extended_{year}.csv'),
            os.path.join(data_dir, f'omni_2_{year}.csv'),
            os.path.join(data_dir, f'omni_lowres_{year}.csv')
        ]
        
        file_found = False
        for pattern in file_patterns:
            if os.path.exists(pattern):
                print(f"로딩 중: {pattern}")
                df_year = pd.read_csv(pattern, parse_dates=['datetime'])
                all_data.append(df_year)
                file_found = True
                break
        
        if not file_found:
            print(f"경고: {year}년 파일을 찾을 수 없습니다.")
    
    if not all_data:
        raise FileNotFoundError("데이터 파일을 찾을 수 없습니다.")
    
    # 모든 데이터 병합
    df = pd.concat(all_data, ignore_index=True)
    print(f"전체 데이터 로드 완료: {len(df)} 레코드")
    
    # 시간 범위 필터링
    df_filtered = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
    print(f"필터링 후: {len(df_filtered)} 레코드")
    
    if len(df_filtered) == 0:
        print("경고: 해당 시간 범위에 데이터가 없습니다.")
        return None
    
    # CSV 저장
    df_filtered.to_csv(output_file, index=False)
    print(f"저장 완료: {output_file}")
    
    return df_filtered

def extract_omni_data(center_datetime, hours_before, hours_after, 
                      data_dir, output_file, data_type='auto'):
    """
    OMNI 데이터에서 특정 datetime 기준으로 시간 범위 데이터 추출
    
    Parameters:
    -----------
    center_datetime : str or datetime
        기준 datetime (예: '2024-01-01 12:00:00')
    hours_before : int
        기준 시간 이전 몇 시간
    hours_after : int
        기준 시간 이후 몇 시간
    data_dir : str
        CSV 파일들이 있는 디렉토리
    output_file : str
        출력 CSV 파일명
    data_type : str
        데이터 타입 ('highres', 'lowres', 'auto')
        'auto'인 경우 파일을 검색하여 자동 판단
    
    Returns:
    --------
    pd.DataFrame : 추출된 데이터프레임
    """
    if data_type == 'auto':
        # 디렉토리에서 파일 검색하여 타입 자동 판단
        highres_files = glob.glob(os.path.join(data_dir, 'omni_*.csv'))
        lowres_files = glob.glob(os.path.join(data_dir, 'omni2_*.csv'))
        
        if highres_files and not lowres_files:
            data_type = 'highres'
            print("고해상도 OMNI 데이터로 감지됨")
        elif lowres_files and not highres_files:
            data_type = 'lowres'
            print("저해상도 OMNI2 데이터로 감지됨")
        elif highres_files and lowres_files:
            print("경고: 두 종류의 데이터가 모두 있습니다. 고해상도 데이터를 사용합니다.")
            data_type = 'highres'
        else:
            raise FileNotFoundError(f"'{data_dir}' 디렉토리에서 OMNI 데이터 파일을 찾을 수 없습니다.")
    
    if data_type == 'highres':
        return extract_time_range_highres(center_datetime, hours_before, hours_after, 
                                          data_dir, output_file)
    elif data_type == 'lowres':
        return extract_time_range_lowres(center_datetime, hours_before, hours_after, 
                                         data_dir, output_file)
    else:
        raise ValueError("data_type은 'highres', 'lowres', 또는 'auto' 중 하나여야 합니다.")

# 사용 예시
if __name__ == "__main__":

    extract_omni_data(
        center_datetime='2020-01-01 00:00:00',
        hours_before=10,
        hours_after=10,
        data_dir='/Users/eunsupark/ap_project/data/omni/highres',
        output_file='./omni_extract_highres.csv',
        data_type='highres'
    )

    extract_omni_data(
        center_datetime='2020-01-01 00:00:00',
        hours_before=10,
        hours_after=10,
        data_dir='/Users/eunsupark/ap_project/data/omni/lowres',
        output_file='./omni_extract_lowres.csv',
        data_type='lowres'
    )




    # 예시 1: 고해상도 데이터 (1분 → 1시간 평균)
    # extract_omni_data(
    #     center_datetime='2020-01-01 00:00:00',
    #     hours_before=24,
    #     hours_after=48,
    #     data_dir='./omni_highres_data',
    #     output_file='omni_extract_highres.csv',
    #     data_type='highres'
    # )
    
    # 예시 2: 저해상도 데이터 (1시간 데이터)
    # extract_omni_data(
    #     center_datetime='2024-01-15 12:00:00',
    #     hours_before=24,
    #     hours_after=48,
    #     data_dir='./omni_lowres_data',
    #     output_file='omni_extract_lowres.csv',
    #     data_type='lowres'
    # )
    
    # 예시 3: 자동 감지
    # extract_omni_data(
    #     center_datetime='2024-01-15 12:00:00',
    #     hours_before=24,
    #     hours_after=48,
    #     data_dir='./omni_data',
    #     output_file='omni_extract.csv',
    #     data_type='auto'
    # )
    
    # print("extract_omni_data 함수를 사용하세요.")
    # print("예시:")
    # print("extract_omni_data('2024-01-15 12:00:00', 24, 48, './data', 'output.csv')")