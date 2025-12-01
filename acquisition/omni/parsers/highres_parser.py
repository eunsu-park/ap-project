"""
High Resolution 파서

OMNI High Resolution (1-minute or 5-minute) data parser
"""
from datetime import datetime, timedelta
import pandas as pd
from .base_parser import BaseOMNIParser


class HighResParser(BaseOMNIParser):
    """
    High Resolution 파서
    
    - 46개 컬럼 (1-min) 또는 49개 (5-min)
    - Minutely 데이터
    - .asc 파일 포맷
    """
    
    def __init__(self, is_5min=False):
        """
        초기화 - High Res 특화 설정
        
        Args:
            is_5min: True면 5분 데이터 (GOES flux 3개 추가)
        """
        super().__init__()
        
        self.is_5min = is_5min
        
        # Fortran 포맷 (HRO 공식 문서 기준)
        if is_5min:
            # 5-min: 기본 + GOES flux 3개
            self.fortran_format = "(2I4,4I3,3I4,2I7,F6.2,I7, 8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.2, F5.1,3F9.2)"
        else:
            # 1-min: 기본 46개
            self.fortran_format = "(2I4,4I3,3I4,2I7,F6.2,I7, 8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.2, F5.1)"
        
        # 46개 컬럼명 (기본)
        self.column_names = [
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
        
        # 5분 데이터는 GOES flux 3개 추가
        if is_5min:
            self.column_names.extend([
                'Proton_Flux_gt10MeV',
                'Proton_Flux_gt30MeV',
                'Proton_Flux_gt60MeV'
            ])
        
        # Fill values (0-based index)
        self.fill_values = {
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
        }
        
        # 5-min 데이터의 경우 GOES flux fill values 추가
        if is_5min:
            self.fill_values.update({
                46: [99999.99],  # Proton flux >10 MeV
                47: [99999.99],  # Proton flux >30 MeV
                48: [99999.99],  # Proton flux >60 MeV
            })
        
        # 시간 관련 컬럼 (Int64로 변환)
        self.time_columns = ['Year', 'Day', 'Hour', 'Minute']
        
        # 정수형 컬럼 (float64로 변환 - NaN 처리)
        self.integer_columns = [
            'IMF_SC_ID', 'SW_Plasma_SC_ID',
            'IMF_Avg_Points', 'Plasma_Avg_Points', 'Percent_Interp',
            'Timeshift_sec', 'RMS_Timeshift', 'Time_Between_Obs_sec',
            'AE_Index_nT', 'AL_Index_nT', 'AU_Index_nT',
            'SYM_D_nT', 'SYM_H_nT', 'ASY_D_nT', 'ASY_H_nT'
        ]
    
    def create_datetime(self, row):
        """
        Year, Day, Hour, Minute으로부터 datetime 생성
        
        Args:
            row: DataFrame row
        
        Returns:
            datetime or pd.NaT
        """
        try:
            year = int(row['Year'])
            day_of_year = int(row['Day'])
            hour = int(row['Hour'])
            minute = int(row['Minute'])
            
            # Day of year를 datetime으로 변환
            base_date = datetime(year, 1, 1)
            delta = timedelta(days=day_of_year - 1, hours=hour, minutes=minute)
            return base_date + delta
            
        except (ValueError, TypeError):
            # 시간 데이터가 유효하지 않으면 NaT 반환
            return pd.NaT


if __name__ == '__main__':
    # 파서 테스트
    print("=" * 60)
    print("1-minute parser:")
    parser_1min = HighResParser(is_5min=False)
    print(f"  Columns: {len(parser_1min.column_names)}")
    print(f"  Fill values: {len(parser_1min.fill_values)} columns")
    
    print("\n" + "=" * 60)
    print("5-minute parser:")
    parser_5min = HighResParser(is_5min=True)
    print(f"  Columns: {len(parser_5min.column_names)}")
    print(f"  Fill values: {len(parser_5min.fill_values)} columns")