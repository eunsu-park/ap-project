"""
Low Resolution 테이블 스키마

OMNI2 Low Resolution (Hourly) table schema with 55 columns
Based on OMNI2.text format specification
"""

def get_lowres_schema():
    """
    Low Resolution 테이블 스키마 반환
    
    Returns:
        dict: 55개 컬럼의 PostgreSQL 스키마
    """
    return {
        # ====================================================================
        # 시간 정보 (4개) - Primary Key
        # ====================================================================
        'datetime': 'TIMESTAMP PRIMARY KEY',
        'year': 'INTEGER NOT NULL',
        'decimal_day': 'INTEGER NOT NULL',  # Day of year (1-366)
        'hour': 'INTEGER NOT NULL',         # Hour (0-23)
        
        # ====================================================================
        # 메타데이터 (5개)
        # ====================================================================
        'bartels_rotation_number': 'INTEGER',
        'imf_sc_id': 'SMALLINT',             # IMF Spacecraft ID
        'sw_plasma_sc_id': 'SMALLINT',       # SW Plasma Spacecraft ID
        'imf_avg_points': 'INTEGER',         # Points in IMF averages
        'plasma_avg_points': 'INTEGER',      # Points in Plasma averages
        
        # ====================================================================
        # 자기장 데이터 (14개) - Magnetic Field
        # ====================================================================
        'b_field_magnitude_avg_nt': 'REAL',                 # |B| average
        'b_magnitude_of_avg_field_vector_nt': 'REAL',       # |<B>|
        'b_lat_angle_avg_field_vector_deg': 'REAL',         # Latitude angle
        'b_long_angle_avg_field_vector_deg': 'REAL',        # Longitude angle
        'bx_gse_gsm_nt': 'REAL',                            # Bx GSE, GSM
        'by_gse_nt': 'REAL',                                # By GSE
        'bz_gse_nt': 'REAL',                                # Bz GSE
        'by_gsm_nt': 'REAL',                                # By GSM
        'bz_gsm_nt': 'REAL',                                # Bz GSM
        'sigma_b_magnitude_nt': 'REAL',                     # RMS σ|B|
        'sigma_b_vector_nt': 'REAL',                        # RMS σB
        'sigma_bx_nt': 'REAL',                              # RMS σBx
        'sigma_by_nt': 'REAL',                              # RMS σBy
        'sigma_bz_nt': 'REAL',                              # RMS σBz
        
        # ====================================================================
        # 플라즈마 데이터 (14개) - Solar Wind Plasma
        # ====================================================================
        'proton_temperature_k': 'REAL',                     # Temperature (K)
        'proton_density_n_cm3': 'REAL',                     # Density (n/cc)
        'plasma_flow_speed_km_s': 'REAL',                   # Flow speed (km/s)
        'plasma_flow_long_angle_deg': 'REAL',               # Flow longitude angle
        'plasma_flow_lat_angle_deg': 'REAL',                # Flow latitude angle
        'na_np_ratio': 'REAL',                              # Alpha/Proton ratio
        'flow_pressure_npa': 'REAL',                        # Flow pressure (nPa)
        'sigma_temperature_k': 'REAL',                      # RMS σT
        'sigma_density_n_cm3': 'REAL',                      # RMS σN
        'sigma_flow_speed_km_s': 'REAL',                    # RMS σV
        'sigma_phi_v_deg': 'REAL',                          # RMS σφV
        'sigma_theta_v_deg': 'REAL',                        # RMS σθV
        'sigma_na_np': 'REAL',                              # RMS σ(Na/Np)
        'electric_field_mv_m': 'REAL',                      # Electric field (mV/m)
        'plasma_beta': 'REAL',                              # Plasma beta
        'alfven_mach_number': 'REAL',                       # Alfven Mach number
        
        # ====================================================================
        # 지자기 인덱스 (10개) - Geomagnetic Indices
        # ====================================================================
        'kp_index': 'SMALLINT',                             # Kp index
        'sunspot_number_r': 'INTEGER',                      # Sunspot number R
        'dst_index_nt': 'INTEGER',                          # DST index (nT)
        'ae_index_nt': 'INTEGER',                           # AE index (nT)
        'proton_flux_gt1mev': 'REAL',                       # Proton flux >1 MeV
        'proton_flux_gt2mev': 'REAL',                       # Proton flux >2 MeV
        'proton_flux_gt4mev': 'REAL',                       # Proton flux >4 MeV
        'proton_flux_gt10mev': 'REAL',                      # Proton flux >10 MeV
        'proton_flux_gt30mev': 'REAL',                      # Proton flux >30 MeV
        'proton_flux_gt60mev': 'REAL',                      # Proton flux >60 MeV
        
        # ====================================================================
        # 기타 인덱스 및 데이터 (8개)
        # ====================================================================
        'flag': 'SMALLINT',                                 # Flag
        'ap_index_nt': 'INTEGER',                           # ap index (nT)
        'f10_7_index_sfu': 'REAL',                          # F10.7 index (sfu)
        'pc_n_index': 'REAL',                               # PC(N) index
        'al_index_nt': 'INTEGER',                           # AL index (nT)
        'au_index_nt': 'INTEGER',                           # AU index (nT)
        'magnetosonic_mach_number': 'REAL',                 # Magnetosonic Mach number
    }


def get_lowres_indices():
    """
    Low Resolution 테이블의 추가 인덱스 정의
    
    Returns:
        list: 인덱스 생성 쿼리 리스트
    """
    table_name = 'low_resolution'
    
    return [
        f"CREATE INDEX idx_lowres_year ON {table_name}(year)",
        f"CREATE INDEX idx_lowres_date ON {table_name}(year, decimal_day)",
        f"CREATE INDEX idx_lowres_datetime ON {table_name}(datetime)",
    ]


if __name__ == '__main__':
    # 스키마 확인용
    schema = get_lowres_schema()
    print(f"Low Resolution Schema: {len(schema)} columns")
    print("\nColumns:")
    for col_name, col_type in schema.items():
        print(f"  {col_name}: {col_type}")