"""
공통 파싱 로직

Base parser class with common parsing methods for OMNI data
"""
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class BaseOMNIParser:
    """
    OMNI 데이터 파싱 베이스 클래스
    
    공통 파싱 로직을 제공하며, 각 해상도별 파서는 이를 상속받아 구현
    """
    
    def __init__(self):
        """초기화 - 서브클래스에서 오버라이드"""
        self.fortran_format = None
        self.column_names = []
        self.fill_values = {}
        self.time_columns = []
        self.integer_columns = []
    
    def parse_fortran_format(self, format_string):
        """
        Fortran 포맷 문자열 파싱 (공통)
        
        Args:
            format_string: Fortran 포맷 문자열 (예: "(2I4,I3,I5,...)")
        
        Returns:
            list: 각 필드의 위치와 타입 정보
                [{'start': 0, 'end': 4, 'width': 4, 'type': 'I'}, ...]
        """
        format_parts = []
        
        # 정규식: nXw.d 형식 파싱
        # n: 반복 횟수 (선택), X: 타입 (I/F/A), w: 너비, d: 소수점 자리수 (선택)
        pattern = r'(\d*)([IFA])(\d+)(?:\.(\d+))?'
        
        position = 0
        for match in re.finditer(pattern, format_string):
            repeat = int(match.group(1)) if match.group(1) else 1
            field_type = match.group(2)
            width = int(match.group(3))
            
            # 반복 횟수만큼 필드 추가
            for _ in range(repeat):
                format_parts.append({
                    'start': position,
                    'end': position + width,
                    'width': width,
                    'type': field_type
                })
                position += width
        
        return format_parts
    
    def parse_line(self, line, format_parts):
        """
        한 줄의 고정폭 데이터 파싱 (공통)
        
        Args:
            line: 파싱할 텍스트 라인
            format_parts: parse_fortran_format() 결과
        
        Returns:
            list: 파싱된 값 리스트
        """
        values = []
        
        for fmt in format_parts:
            try:
                # 위치에 따라 문자열 추출
                field_str = line[fmt['start']:fmt['end']].strip()
                
                if field_str:
                    if fmt['type'] == 'I':  # Integer
                        values.append(int(field_str))
                    else:  # Float (F or A)
                        values.append(float(field_str))
                else:
                    values.append(np.nan)
                    
            except (ValueError, IndexError):
                # 파싱 실패 시 NaN
                values.append(np.nan)
        
        return values
    
    def apply_fill_values(self, df):
        """
        Fill values를 NaN으로 변환 (공통)
        
        Args:
            df: pandas DataFrame
        
        Returns:
            DataFrame: Fill values가 NaN으로 변환된 DataFrame
        """
        for col_idx, fill_val_list in self.fill_values.items():
            if col_idx < len(df.columns) and fill_val_list:
                # 각 fill value를 NaN으로 교체
                for fill_val in fill_val_list:
                    df.iloc[:, col_idx] = df.iloc[:, col_idx].replace(fill_val, np.nan)
        
        return df
    
    def create_datetime(self, row):
        """
        시간 컬럼으로부터 datetime 생성 (서브클래스에서 오버라이드)
        
        Args:
            row: DataFrame row
        
        Returns:
            datetime or pd.NaT
        """
        raise NotImplementedError("Subclass must implement create_datetime()")
    
    def convert_column_types(self, df):
        """
        컬럼 타입 변환 (공통)
        
        시간 컬럼: Int64 (Nullable Integer)
        정수형 컬럼: float64 (NaN 처리 위해)
        
        Args:
            df: pandas DataFrame
        
        Returns:
            DataFrame: 타입이 변환된 DataFrame
        """
        # 시간 관련 컬럼은 Int64 (Nullable Integer)
        for col in self.time_columns:
            if col in df.columns:
                df[col] = df[col].astype('Int64')
        
        # 다른 정수형 컬럼은 float64 (NaN 안전 처리)
        for col in self.integer_columns:
            if col in df.columns:
                df[col] = df[col].astype('float64')
        
        return df
    
    def convert_to_records(self, df):
        """
        DataFrame → DB 삽입용 레코드 리스트 변환 (공통)
        
        - 컬럼명 소문자 변환
        - NaN → None (PostgreSQL NULL)
        
        Args:
            df: pandas DataFrame
        
        Returns:
            list: DB 삽입용 딕셔너리 리스트
        """
        # 컬럼명 소문자 변환
        df.columns = df.columns.str.lower()
        
        # Dict 리스트로 변환
        records = df.to_dict('records')
        
        # NaN → None
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        return records
    
    def parse_data(self, data_text):
        """
        전체 데이터 파싱 (공통 워크플로우)
        
        Args:
            data_text: 다운로드한 텍스트 데이터
        
        Returns:
            DataFrame: 파싱 및 처리된 DataFrame
        """
        # 1. Fortran 포맷 파싱
        format_parts = self.parse_fortran_format(self.fortran_format)
        
        # 2. 각 라인 파싱
        lines = data_text.strip().split('\n')
        data_rows = []
        
        for i, line in enumerate(lines):
            if line.strip():  # 빈 줄 제외
                try:
                    values = self.parse_line(line, format_parts)
                    data_rows.append(values)
                except Exception as e:
                    print(f"  Warning: Line {i+1} parsing error: {e}")
                    continue
        
        print(f"  Parsed {len(data_rows)} records")
        
        # 3. DataFrame 생성
        df = pd.DataFrame(data_rows, columns=self.column_names)
        
        # 4. 타입 변환
        df = self.convert_column_types(df)
        
        # 5. Fill values 처리
        df = self.apply_fill_values(df)
        
        # 6. Datetime 컬럼 생성
        df['datetime'] = df.apply(self.create_datetime, axis=1)
        
        # 7. datetime 컬럼을 맨 앞으로
        cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
        df = df[cols]
        
        return df