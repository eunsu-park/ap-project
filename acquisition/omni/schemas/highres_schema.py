"""
High Resolution 테이블 스키마

OMNI High Resolution (1-minute) table schema with 46 columns
Optional: 5-minute data includes 3 additional GOES flux columns (49 total)
Based on HRO format specification
"""

def get_highres_schema(include_5min_fields=False):
    """
    High Resolution 테이블 스키마 반환
    
    Args:
        include_5min_fields: True면 5분 데이터용 GOES flux 3개 컬럼 포함
    
    Returns:
        dict: 46개 (또는 49개) 컬럼의 PostgreSQL 스키마
    """
    schema = {
        # ====================================================================
        # 시간 정보 (5개) - Primary Key
        # ====================================================================
        'datetime': 'TIMESTAMP PRIMARY KEY',
        'year': 'INTEGER NOT NULL',
        'day': 'INTEGER NOT NULL',      # Day of year (1-366)
        'hour': 'INTEGER NOT NULL',     # Hour (0-23)
        'minute': 'INTEGER NOT NULL',   # Minute (0-59 for 1-min, 0,5,10,...55 for 5-min)
        
        # ====================================================================
        # 메타데이터 (8개)
        # ====================================================================
        'imf_sc_id': 'SMALLINT',                    # IMF Spacecraft ID
        'sw_plasma_sc_id': 'SMALLINT',              # SW Plasma Spacecraft ID
        'imf_avg_points': 'INTEGER',                # Points in IMF averages
        'plasma_avg_points': 'INTEGER',             # Points in Plasma averages
        'percent_interp': 'INTEGER',                # Percent interpolated
        'timeshift_sec': 'INTEGER',                 # Timeshift (sec)
        'rms_timeshift': 'INTEGER',                 # RMS Timeshift
        'rms_phase_front_normal': 'REAL',           # RMS Phase front normal
        'time_between_obs_sec': 'INTEGER',          # Time between observations (sec)
        
        # ====================================================================
        # 자기장 데이터 (8개) - Magnetic Field
        # ====================================================================
        'b_magnitude_nt': 'REAL',                   # |B| magnitude
        'bx_gse_nt': 'REAL',                        # Bx GSE
        'by_gse_nt': 'REAL',                        # By GSE
        'bz_gse_nt': 'REAL',                        # Bz GSE
        'by_gsm_nt': 'REAL',                        # By GSM
        'bz_gsm_nt': 'REAL',                        # Bz GSM
        'rms_sd_b_scalar_nt': 'REAL',               # RMS SD B scalar
        'rms_sd_b_vector_nt': 'REAL',               # RMS SD B vector
        
        # ====================================================================
        # 플라즈마 데이터 (10개) - Solar Wind Plasma
        # ====================================================================
        'flow_speed_km_s': 'REAL',                  # Flow speed (km/s)
        'vx_gse_km_s': 'REAL',                      # Vx GSE (km/s)
        'vy_gse_km_s': 'REAL',                      # Vy GSE (km/s)
        'vz_gse_km_s': 'REAL',                      # Vz GSE (km/s)
        'proton_density_n_cc': 'REAL',              # Proton density (n/cc)
        'temperature_k': 'REAL',                    # Temperature (K)
        'flow_pressure_npa': 'REAL',                # Flow pressure (nPa)
        'electric_field_mv_m': 'REAL',              # Electric field (mV/m)
        'plasma_beta': 'REAL',                      # Plasma beta
        'alfven_mach_number': 'REAL',               # Alfven Mach number
        
        # ====================================================================
        # 위성 위치 (6개) - Spacecraft Position
        # ====================================================================
        'sc_x_gse_re': 'REAL',                      # S/C X GSE (Re)
        'sc_y_gse_re': 'REAL',                      # S/C Y GSE (Re)
        'sc_z_gse_re': 'REAL',                      # S/C Z GSE (Re)
        'bsn_x_gse_re': 'REAL',                     # BSN X GSE (Re)
        'bsn_y_gse_re': 'REAL',                     # BSN Y GSE (Re)
        'bsn_z_gse_re': 'REAL',                     # BSN Z GSE (Re)
        
        # ====================================================================
        # 지자기 인덱스 (9개) - Geomagnetic Indices
        # ====================================================================
        'ae_index_nt': 'INTEGER',                   # AE index (nT)
        'al_index_nt': 'INTEGER',                   # AL index (nT)
        'au_index_nt': 'INTEGER',                   # AU index (nT)
        'sym_d_nt': 'INTEGER',                      # SYM/D index (nT)
        'sym_h_nt': 'INTEGER',                      # SYM/H index (nT)
        'asy_d_nt': 'INTEGER',                      # ASY/D index (nT)
        'asy_h_nt': 'INTEGER',                      # ASY/H index (nT)
        'pc_n_index': 'REAL',                       # PC(N) index
        'magnetosonic_mach_number': 'REAL',         # Magnetosonic Mach number
    }
    
    # ====================================================================
    # 5분 데이터용 GOES Proton Flux (3개) - Optional
    # ====================================================================
    if include_5min_fields:
        schema.update({
            'proton_flux_gt10mev': 'REAL',          # Proton flux >10 MeV
            'proton_flux_gt30mev': 'REAL',          # Proton flux >30 MeV
            'proton_flux_gt60mev': 'REAL',          # Proton flux >60 MeV
        })
    
    return schema


def get_highres_indices():
    """
    High Resolution 테이블의 추가 인덱스 정의
    
    Returns:
        list: 인덱스 생성 쿼리 리스트
    """
    table_name = 'high_resolution'
    
    return [
        f"CREATE INDEX idx_highres_year ON {table_name}(year)",
        f"CREATE INDEX idx_highres_date ON {table_name}(year, day)",
        f"CREATE INDEX idx_highres_datetime ON {table_name}(year, day, hour)",
    ]


if __name__ == '__main__':
    # 스키마 확인용
    print("=" * 60)
    print("1-minute data schema:")
    schema_1min = get_highres_schema(include_5min_fields=False)
    print(f"Columns: {len(schema_1min)}")
    
    print("\n" + "=" * 60)
    print("5-minute data schema:")
    schema_5min = get_highres_schema(include_5min_fields=True)
    print(f"Columns: {len(schema_5min)}")
    
    print("\n" + "=" * 60)
    print("Column details:")
    for col_name, col_type in schema_1min.items():
        print(f"  {col_name}: {col_type}")