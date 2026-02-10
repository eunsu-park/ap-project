# OMNI Database Downloader - 사용 가이드

## 📁 프로젝트 구조

```
omni_database/
├── config.py                      # 설정 파일
├── create_tables.py               # 테이블 생성 스크립트
├── download_all.py                # 메인 다운로드 스크립트 ⭐
├── validate_schemas.py            # 스키마 검증
│
├── schemas/                       # 테이블 스키마 정의
│   ├── base_schema.py
│   ├── lowres_schema.py          # Low Res (55 컬럼)
│   └── highres_schema.py         # High Res (46/49 컬럼)
│
├── parsers/                       # 데이터 파싱
│   ├── base_parser.py            # 공통 파싱 로직
│   ├── lowres_parser.py          # Low Res 파서
│   └── highres_parser.py         # High Res 파서
│
└── downloaders/                   # 다운로드 및 DB 삽입
    └── base_downloader.py        # 공통 다운로더
```

---

## 🚀 빠른 시작

### 1. 테이블 생성

```bash
cd /Users/eunsupark/GitHub/ap-project/acquisition/omni
python create_tables.py
```

**출력:**
```
============================================================
OMNI 데이터베이스 생성
============================================================
✓ omni 데이터베이스 이미 존재

============================================================
Low Resolution 테이블 생성
============================================================
테이블 생성 중: low_resolution (56 컬럼)
✓ 테이블 생성 완료
✓ 3개 인덱스 생성 완료

============================================================
High Resolution 테이블 생성
============================================================
테이블 생성 중: high_resolution (47 컬럼, 1-min)
✓ 테이블 생성 완료
✓ 3개 인덱스 생성 완료
```

### 2. 데이터 다운로드

**Low Resolution (2010-2024):**
```bash
python download_all.py --lowres --start 2010 --end 2024
```

**High Resolution (2023):**
```bash
python download_all.py --highres --start 2023 --end 2023
```

**둘 다 (2020-2024):**
```bash
python download_all.py --lowres --highres --start 2020 --end 2024
```

### 3. 데이터 확인

```bash
python download_all.py --status
```

---

## 📖 상세 사용법

### create_tables.py - 테이블 생성

**기본 사용:**
```bash
python create_tables.py
```

**테이블 재생성:**
```python
# create_tables.py 수정
create_low_resolution_table(drop_if_exists=True)   # 삭제 후 재생성
create_high_resolution_table(drop_if_exists=True)
```

**5분 데이터용 테이블:**
```python
# create_tables.py 수정
create_high_resolution_table(
    drop_if_exists=False, 
    include_5min=True  # GOES flux 3개 컬럼 추가
)
```

---

### download_all.py - 데이터 다운로드

#### 기본 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--lowres` | Low Res 다운로드 | `--lowres` |
| `--highres` | High Res 다운로드 | `--highres` |
| `--5min` | 5분 High Res (GOES flux) | `--5min` |
| `--start YEAR` | 시작 연도 | `--start 2010` |
| `--end YEAR` | 종료 연도 | `--end 2024` |
| `--save-raw` | 원본 파일 저장 | `--save-raw` |
| `--status` | DB 상태 확인 | `--status` |
| `--verify TABLE YEAR` | 데이터 검증 | `--verify low_resolution 2023` |

#### 사용 예시

**1. Low Res 한 해만:**
```bash
python download_all.py --lowres --start 2023 --end 2023
```

**2. High Res 여러 해:**
```bash
python download_all.py --highres --start 2020 --end 2024
```

**3. 둘 다 + 원본 파일 저장:**
```bash
python download_all.py --lowres --highres --start 2023 --end 2023 --save-raw
```

**4. 5분 High Res:**
```bash
python download_all.py --highres --5min --start 2023 --end 2023
```

**5. 데이터베이스 상태 확인:**
```bash
python download_all.py --status
```

출력:
```
============================================================
DATABASE STATUS
============================================================

low_resolution:
  Total records: 131,400
  Years: 15
    2010: 8,760 records
    2011: 8,760 records
    2012: 8,784 records
    ...

high_resolution:
  Total records: 525,600
  Years: 1
    2023: 525,600 records
```

**6. 특정 연도 검증:**
```bash
python download_all.py --verify low_resolution 2023
```

출력:
```
============================================================
Verifying low_resolution - Year 2023
============================================================
Total records: 8,760
Date range: 2023-01-01 00:00:00 to 2023-12-31 23:00:00

Sample data (first 3 records):
  Record 1:
    datetime: 2023-01-01 00:00:00
    year: 2023
    decimal_day: 1
    hour: 0
    bz_gsm_nt: -2.3
    plasma_flow_speed_km_s: 450.5
```

---

## 🔧 고급 설정

### config.py 수정

**데이터베이스 설정:**
```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'omni',
    'user': 'eunsupark',
    'password': 'eunsupark',
    'log_queries': False  # True로 바꾸면 SQL 쿼리 로깅
}
```

**배치 크기 조정:**
```python
BATCH_SIZE = 1000  # 기본값, 메모리에 따라 조정 가능 (100-5000)
```

**타임아웃 설정:**
```python
DOWNLOAD_TIMEOUT = 30  # 초 (느린 네트워크는 60으로 증가)
```

---

## 📊 데이터 구조

### Low Resolution (Hourly)

**테이블:** `low_resolution`
**컬럼:** 56개 (datetime + 55개)
**시간 해상도:** 1시간
**예상 레코드/년:** ~8,760개

**주요 컬럼:**
- `datetime` (PRIMARY KEY)
- `year`, `decimal_day`, `hour`
- 자기장: `bx_gse_gsm_nt`, `by_gse_nt`, `bz_gse_nt`, `by_gsm_nt`, `bz_gsm_nt`
- 플라즈마: `proton_temperature_k`, `proton_density_n_cm3`, `plasma_flow_speed_km_s`
- 인덱스: `kp_index`, `dst_index_nt`, `ae_index_nt`

### High Resolution (Minutely)

**테이블:** `high_resolution`
**컬럼:** 47개 (1-min) / 50개 (5-min)
**시간 해상도:** 1분
**예상 레코드/년:** ~525,600개

**주요 컬럼:**
- `datetime` (PRIMARY KEY)
- `year`, `day`, `hour`, `minute`
- 자기장: `bx_gse_nt`, `by_gse_nt`, `bz_gse_nt`, `by_gsm_nt`, `bz_gsm_nt`
- 플라즈마: `temperature_k`, `proton_density_n_cc`, `flow_speed_km_s`
- 위성 위치: `sc_x_gse_re`, `sc_y_gse_re`, `sc_z_gse_re`
- 인덱스: `ae_index_nt`, `sym_d_nt`, `sym_h_nt`

---

## 💾 디스크 용량

### 예상 용량 (압축 전)

**Low Resolution:**
- 1년: ~2 MB
- 15년 (2010-2024): ~30 MB

**High Resolution:**
- 1년: ~100 MB
- 15년 (2010-2024): ~1.5 GB

**여유 공간:** 최소 2-3 GB 권장

---

## ⏱️ 예상 소요 시간

### 다운로드 + 파싱 + DB 삽입

**Low Resolution:**
- 1년: ~30초
- 15년: ~7-10분

**High Resolution:**
- 1년: ~5-10분
- 15년: ~2-3시간

**네트워크 속도에 따라 변동**

---

## 🐛 트러블슈팅

### 1. 다운로드 실패

**증상:**
```
✗ HTTP error: 404 Client Error
```

**원인:** URL 패턴 변경 또는 해당 연도 데이터 없음

**해결:**
- NASA SPDF 웹사이트에서 URL 확인
- `config.py`의 `url_pattern` 수정

### 2. 파싱 에러

**증상:**
```
✗ Parsing error: ...
```

**원인:** Fortran 포맷 불일치

**해결:**
- 원본 파일 저장: `--save-raw`
- `/tmp/omni_raw/` 에서 파일 확인
- 포맷 문자열 수정: `parsers/lowres_parser.py` 또는 `highres_parser.py`

### 3. DB 삽입 실패

**증상:**
```
✗ Database insertion error: ...
```

**원인:** 데이터 타입 불일치, 중복 키

**해결:**
```bash
# 테이블 재생성
python create_tables.py  # drop_if_exists=True로 수정

# 특정 연도 삭제 후 재시도
psql -U eunsupark -d omni -c "DELETE FROM low_resolution WHERE year = 2023;"
```

### 4. 메모리 부족

**증상:** 프로세스 종료 또는 매우 느림

**해결:**
```python
# config.py 수정
BATCH_SIZE = 100  # 1000에서 100으로 감소
```

### 5. 타임아웃

**증상:**
```
✗ Timeout error (>30s)
```

**해결:**
```python
# config.py 수정
DOWNLOAD_TIMEOUT = 60  # 30에서 60으로 증가
```

---

## 📝 Python 스크립트에서 사용

### 다운로드

```python
from config import DB_CONFIG, TABLE_CONFIGS
from parsers.lowres_parser import LowResParser
from downloaders.base_downloader import BaseDownloader

# Low Res 다운로더 생성
parser = LowResParser()
downloader = BaseDownloader(
    db_config=DB_CONFIG,
    table_config=TABLE_CONFIGS['low_resolution'],
    parser=parser
)

# 단일 연도 다운로드
success = downloader.download_year(2023, save_raw_file=False)

# 여러 연도 다운로드
results = downloader.download_multiple_years(2020, 2024)
print(f"Success: {results['success']}")
print(f"Failed: {results['failed']}")
```

### 데이터 조회

```python
from egghouse.database import PostgresManager
from config import DB_CONFIG

with PostgresManager(**DB_CONFIG) as db:
    # 특정 날짜 범위 조회
    data = db.select_date_range(
        'low_resolution',
        'datetime',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    # DataFrame으로 변환
    df = db.to_dataframe('low_resolution', limit=1000)
    
    # 조건 조회
    geomagnetic_storms = db.select(
        'low_resolution',
        where={'year': 2023},
        conditions="dst_index_nt < -100"
    )
```

---

## ✅ 체크리스트

**초기 설정:**
- [ ] PostgreSQL 실행 중
- [ ] `omni` 데이터베이스 존재
- [ ] egghouse 최신 버전 설치
- [ ] 네트워크 연결 확인

**테이블 생성:**
- [ ] `python create_tables.py` 실행
- [ ] `low_resolution` 테이블 생성 확인
- [ ] `high_resolution` 테이블 생성 확인

**데이터 다운로드:**
- [ ] Low Res 다운로드 완료
- [ ] High Res 다운로드 완료 (선택)
- [ ] `--status`로 확인
- [ ] `--verify`로 검증

---

## 🎯 다음 단계

데이터를 성공적으로 다운로드했다면:

1. **데이터 분석:**
   - Jupyter Notebook에서 데이터 탐색
   - 통계 분석, 시각화

2. **머신러닝:**
   - 기존 모델과 통합
   - 새로운 특성 추출

3. **자동화:**
   - Cron job으로 자동 업데이트
   - 매일/매주 최신 데이터 다운로드

---

## 📚 참고 자료

- **OMNI2 공식 문서:** https://spdf.gsfc.nasa.gov/pub/data/omni/
- **egghouse 문서:** /Users/eunsupark/GitHub/egghouse/
- **PostgreSQL 문서:** https://www.postgresql.org/docs/

---

**문제가 있으면 언제든 알려주세요!** 😊