# Data Acquisition - 태양 물리학 데이터 수집

PostgreSQL 데이터베이스에서 태양 및 태양풍 관측 데이터를 추출하는 모듈입니다.

## 설치

### 필수 의존성

```bash
pip install pandas h5py
pip install git+https://github.com/eunsu-park/egghouse.git
```

## 사용법

### OMNI 태양풍 데이터 조회

```python
from datetime import datetime
from acquisition import get_omni_data

# 날짜 범위 지정
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 31)

# 데이터 조회
df = get_omni_data(start, end)
print(df.head())

# 특정 컬럼만 조회
df = get_omni_data(start, end, columns=['timestamp', 'v_sw', 'bz_gsm'])
```

### SDO AIA 관측 데이터 조회

```python
from datetime import datetime
from acquisition import get_sdo_aia_193, get_sdo_aia_211

start = datetime(2024, 1, 1)
end = datetime(2024, 1, 31)

# AIA 193 파장 데이터
df_193 = get_sdo_aia_193(start, end)

# AIA 211 파장 데이터
df_211 = get_sdo_aia_211(start, end)
```

### SDO HMI 자기장 데이터 조회

```python
from datetime import datetime
from acquisition import get_sdo_hmi_magnetogram

start = datetime(2024, 1, 1)
end = datetime(2024, 1, 31)

df = get_sdo_hmi_magnetogram(start, end)
print(df.head())
```

### 특정 시간의 모든 데이터 조회

```python
from datetime import datetime
from acquisition import get_all_data_at_time

# 특정 시간 T
t = datetime(2011, 1, 1, 0, 0, 0)

# 모든 데이터 한번에 추출
data = get_all_data_at_time(t)

# 각 데이터 접근
print(data['omni'])           # OMNI 태양풍 데이터
print(data['aia_193'])        # SDO AIA 193Å
print(data['aia_211'])        # SDO AIA 211Å
print(data['hmi_magnetogram']) # SDO HMI 자기장
```

## 데이터베이스 구성

| 데이터베이스 | 테이블 | 설명 |
|-------------|--------|------|
| `omni` | `low_resolution` | OMNI 태양풍 파라미터 |
| `sdo` | `aia_193` | SDO AIA 193Å 관측 |
| `sdo` | `aia_211` | SDO AIA 211Å 관측 |
| `sdo` | `hmi_magnetogram` | SDO HMI 자기장 |

## 함수 목록

| 함수 | 설명 | 반환값 |
|------|------|--------|
| `get_omni_data(start, end, columns)` | OMNI 태양풍 데이터 (날짜 범위) | DataFrame |
| `get_sdo_aia_193(start, end, columns)` | AIA 193Å 데이터 (날짜 범위) | DataFrame |
| `get_sdo_aia_211(start, end, columns)` | AIA 211Å 데이터 (날짜 범위) | DataFrame |
| `get_sdo_hmi_magnetogram(start, end, columns)` | HMI 자기장 데이터 (날짜 범위) | DataFrame |
| `get_all_data_at_time(t)` | 특정 시간의 모든 데이터 | Dict[str, DataFrame] |

## SDO 영상 전처리 스크립트

`01_get_data_from_database.py`는 SDO 영상을 전처리하고 배치 데이터셋을 생성하는 스크립트입니다.

### 배치 처리 워크플로우

1. 2011-01-01 ~ 2025-01-01까지 **3시간 단위**로 순회
2. `aia_193`, `aia_211`, `hmi_magnetogram` **모두 존재**해야 진행
3. OMNI 데이터 → dict 형태로 변환
4. SDO 영상 → Level 1.5 변환 → AIA degradation 보정 → uint8 이미지
5. `/Volumes/AP-PROJECT/datasets/original/%Y%m%d%H.h5`로 저장 (HDF5)

### 이미지 전처리

| 채널 | 마스킹 | 스케일링 |
|------|--------|----------|
| AIA (193, 211) | 1.2 × r_sun 외부 | log2 스케일링 |
| HMI | 0.99 × r_sun 외부 | 선형 ([-100, 100] → [0, 255]) |

### 실행

```bash
python 01_get_data_from_database.py
```

### HDF5 파일 구조

```
%Y%m%d%H.h5
├── omni/           # OMNI 태양풍 파라미터 (각각 데이터셋)
├── sdo/            # SDO 이미지 (gzip 압축)
│   ├── aia_193
│   ├── aia_211
│   └── hmi_magnetogram
└── attrs: timestamp
```

### 출력 파일 읽기

```python
import h5py

with h5py.File('/Volumes/AP-PROJECT/datasets/original/2011010100.h5', 'r') as f:
    print("Timestamp:", f.attrs['timestamp'])
    print("OMNI keys:", list(f['omni'].keys()))
    print("SDO keys:", list(f['sdo'].keys()))
    print("AIA 193 shape:", f['sdo/aia_193'].shape)

    # 이미지 로드
    aia_193 = f['sdo/aia_193'][:]
```

### 코드에서 사용

```python
from 01_get_data_from_database import process_sdo_channel, to_image

# 단일 채널 처리
m_processed = process_sdo_channel("/path/to/fits", "aia_193")

# uint8 이미지로 변환
image = to_image(m_processed, "aia_193")
```

## 데이터 병합 스크립트

`02_merge_and_resize.py`는 시계열 데이터셋을 병합하고 SDO 이미지를 64×64로 리사이즈합니다.

### 데이터 범위

| 데이터 | 기간 | 간격 | 포인트 수 |
|--------|------|------|-----------|
| OMNI | T-7일 ~ T+3일 | 3시간 | 80개 |
| SDO | T-7일 ~ T | 6시간 | 28개 |

### 실행

```bash
# 데이터셋 생성 (멀티프로세싱)
python 02_merge_and_resize.py

# 파일 검증 (랜덤 10개)
python 02_merge_and_resize.py --verify

# 파일 검증 (랜덤 50개)
python 02_merge_and_resize.py --verify 50

# 전체 파일 검증
python 02_merge_and_resize.py --verify 0
```

### 출력 파일 구조

```
/Volumes/AP-PROJECT/datasets/merged/%Y%m%d%H.h5
├── omni/
│   ├── data              # (80, num_features)
│   └── columns           # feature names
├── sdo/
│   ├── aia_193           # (28, 64, 64)
│   ├── aia_211           # (28, 64, 64)
│   └── hmi_magnetogram   # (28, 64, 64)
└── attrs: t_target, t_start, t_end
```

### 출력 파일 읽기

```python
import h5py

with h5py.File('/Volumes/AP-PROJECT/datasets/merged/2011010100.h5', 'r') as f:
    print("t_target:", f.attrs['t_target'])
    print("OMNI shape:", f['omni/data'].shape)  # (80, num_features)
    print("OMNI columns:", [c.decode() for c in f['omni/columns'][:]])
    print("AIA 193 shape:", f['sdo/aia_193'].shape)  # (28, 64, 64)
```

### 파일 검증 (코드에서 사용)

```python
from 02_merge_and_resize import verify_merged_file

# 단일 파일 검증
result = verify_merged_file('/Volumes/AP-PROJECT/datasets/merged/2011010100.h5')
print("Valid:", result['valid'])
print("Info:", result['info'])
if not result['valid']:
    print("Errors:", result['errors'])
```

## 참고

- 외부 라이브러리: [egghouse](https://github.com/eunsu-park/egghouse)
- egghouse.database 사용법: [EGGHOUSE_README.md](EGGHOUSE_README.md)
- [aiapy 문서](https://aiapy.readthedocs.io/)
