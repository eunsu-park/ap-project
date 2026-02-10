"""
Low Resolution 파서

OMNI2 Low Resolution (Hourly) data parser
"""
from datetime import datetime, timedelta
import pandas as pd
from .base_parser import BaseOMNIParser


class LowResParser(BaseOMNIParser):
    """
    Low Resolution 파서
    
    - 55개 컬럼
    - Hourly 데이터
    - .dat 파일 포맷
    """
    
    def __init__(self):
        """초기화 - Low Res 특화 설정"""
        super().__init__()
        
        # Fortran 포맷 (OMNI2.text 공식 문서 기준)
        self.fortran_format = "(2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2,F9.0,F6.1,F6.0,2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,F6.1,F6.1,2I6,F5.1)"
        
        # 55개 컬럼명
        self.column_names = [
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
        
        # Fill values (0-based index)
        self.fill_values = {
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
        
        # 시간 관련 컬럼 (Int64로 변환)
        self.time_columns = ['Year', 'Decimal_Day', 'Hour']
        
        # 정수형 컬럼 (float64로 변환 - NaN 처리)
        self.integer_columns = [
            'Bartels_Rotation_Number',
            'IMF_SC_ID', 'SW_Plasma_SC_ID',
            'IMF_Avg_Points', 'Plasma_Avg_Points',
            'Kp_Index', 'Sunspot_Number_R',
            'DST_Index_nT', 'AE_Index_nT',
            'Flag', 'ap_Index_nT',
            'AL_Index_nT', 'AU_Index_nT'
        ]
    
    def create_datetime(self, row):
        """
        Year, Decimal_Day, Hour로부터 datetime 생성
        
        Args:
            row: DataFrame row
        
        Returns:
            datetime or pd.NaT
        """
        try:
            year = int(row['Year'])
            day_of_year = int(row['Decimal_Day'])
            hour = int(row['Hour'])
            
            # Day of year를 datetime으로 변환
            base_date = datetime(year, 1, 1)
            delta = timedelta(days=day_of_year - 1, hours=hour)
            return base_date + delta
            
        except (ValueError, TypeError):
            # 시간 데이터가 유효하지 않으면 NaT 반환
            return pd.NaT


if __name__ == '__main__':
    # 파서 테스트
    parser = LowResParser()
    print(f"Low Resolution Parser")
    print(f"  Columns: {len(parser.column_names)}")
    print(f"  Fill values: {len(parser.fill_values)} columns")
    print(f"  Time columns: {parser.time_columns}")
    print(f"  Integer columns: {len(parser.integer_columns)}")