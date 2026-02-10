"""
스키마 검증 스크립트

데이터베이스 연결 없이 스키마 정의를 검증합니다.
"""
from schemas.lowres_schema import get_lowres_schema, get_lowres_indices
from schemas.highres_schema import get_highres_schema, get_highres_indices


def validate_lowres_schema():
    """Low Resolution 스키마 검증"""
    print("=" * 60)
    print("Low Resolution 스키마 검증")
    print("=" * 60)
    
    schema = get_lowres_schema()
    indices = get_lowres_indices()
    
    print(f"\n총 컬럼 수: {len(schema)}")
    print(f"인덱스 수: {len(indices)}")
    
    # Primary Key 확인
    has_pk = False
    for col_name, col_type in schema.items():
        if 'PRIMARY KEY' in col_type:
            print(f"Primary Key: {col_name}")
            has_pk = True
            break
    
    if not has_pk:
        print("⚠ Warning: No PRIMARY KEY found!")
    
    # 시간 컬럼 확인
    time_cols = ['datetime', 'year', 'decimal_day', 'hour']
    print(f"\n시간 컬럼 ({len(time_cols)}):")
    for col in time_cols:
        if col in schema:
            print(f"  ✓ {col}: {schema[col]}")
        else:
            print(f"  ✗ {col}: MISSING")
    
    # NOT NULL 컬럼 확인
    not_null_cols = [col for col, dtype in schema.items() if 'NOT NULL' in dtype]
    print(f"\nNOT NULL 컬럼: {len(not_null_cols)}")
    for col in not_null_cols:
        print(f"  - {col}")
    
    # 인덱스 확인
    print(f"\n인덱스:")
    for idx in indices:
        print(f"  - {idx}")
    
    print("\n✓ Low Resolution 스키마 검증 완료")


def validate_highres_schema():
    """High Resolution 스키마 검증"""
    print("\n" + "=" * 60)
    print("High Resolution 스키마 검증")
    print("=" * 60)
    
    # 1-min 스키마
    schema_1min = get_highres_schema(include_5min_fields=False)
    print(f"\n1-min 스키마:")
    print(f"  총 컬럼 수: {len(schema_1min)}")
    
    # 5-min 스키마
    schema_5min = get_highres_schema(include_5min_fields=True)
    print(f"\n5-min 스키마:")
    print(f"  총 컬럼 수: {len(schema_5min)}")
    print(f"  추가 컬럼: {len(schema_5min) - len(schema_1min)}")
    
    # 차이점 확인
    diff_cols = set(schema_5min.keys()) - set(schema_1min.keys())
    if diff_cols:
        print(f"\n5-min 추가 컬럼:")
        for col in diff_cols:
            print(f"  + {col}: {schema_5min[col]}")
    
    # Primary Key 확인
    has_pk = False
    for col_name, col_type in schema_1min.items():
        if 'PRIMARY KEY' in col_type:
            print(f"\nPrimary Key: {col_name}")
            has_pk = True
            break
    
    if not has_pk:
        print("\n⚠ Warning: No PRIMARY KEY found!")
    
    # 시간 컬럼 확인
    time_cols = ['datetime', 'year', 'day', 'hour', 'minute']
    print(f"\n시간 컬럼 ({len(time_cols)}):")
    for col in time_cols:
        if col in schema_1min:
            print(f"  ✓ {col}: {schema_1min[col]}")
        else:
            print(f"  ✗ {col}: MISSING")
    
    # NOT NULL 컬럼 확인
    not_null_cols = [col for col, dtype in schema_1min.items() if 'NOT NULL' in dtype]
    print(f"\nNOT NULL 컬럼: {len(not_null_cols)}")
    for col in not_null_cols:
        print(f"  - {col}")
    
    # 인덱스 확인
    indices = get_highres_indices()
    print(f"\n인덱스: {len(indices)}")
    for idx in indices:
        print(f"  - {idx}")
    
    print("\n✓ High Resolution 스키마 검증 완료")


def compare_schemas():
    """두 스키마 비교"""
    print("\n" + "=" * 60)
    print("스키마 비교")
    print("=" * 60)
    
    lowres = get_lowres_schema()
    highres = get_highres_schema()
    
    print(f"\nLow Resolution:  {len(lowres)} 컬럼")
    print(f"High Resolution: {len(highres)} 컬럼")
    print(f"차이: {abs(len(lowres) - len(highres))} 컬럼")
    
    # 공통 컬럼 찾기
    lowres_cols = set(lowres.keys())
    highres_cols = set(highres.keys())
    
    common = lowres_cols & highres_cols
    lowres_only = lowres_cols - highres_cols
    highres_only = highres_cols - lowres_cols
    
    print(f"\n공통 컬럼: {len(common)}")
    print(f"Low Res 전용: {len(lowres_only)}")
    print(f"High Res 전용: {len(highres_only)}")
    
    if lowres_only:
        print(f"\nLow Res 전용 컬럼:")
        for col in sorted(lowres_only):
            print(f"  - {col}")
    
    if highres_only:
        print(f"\nHigh Res 전용 컬럼:")
        for col in sorted(highres_only):
            print(f"  - {col}")


def main():
    """메인 실행"""
    print("=" * 60)
    print("OMNI 스키마 검증")
    print("=" * 60)
    
    # Low Res 검증
    validate_lowres_schema()
    
    # High Res 검증
    validate_highres_schema()
    
    # 비교
    compare_schemas()
    
    print("\n" + "=" * 60)
    print("✓ 모든 스키마 검증 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()