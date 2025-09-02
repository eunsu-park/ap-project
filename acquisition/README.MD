# 태양 데이터 처리 파이프라인

태양 관측 데이터(SDO, LASCO, SECCHI)와 태양풍 데이터(OMNI)를 다운로드하고 처리하여 머신러닝용 데이터셋을 생성하는 통합 파이프라인입니다.

## 주요 기능

- **다중 임무 지원**: SDO AIA, SOHO LASCO, STEREO SECCHI, NASA OMNI
- **병렬 처리**: ProcessPoolExecutor를 사용한 고속 다운로드 및 처리
- **에러값 처리**: OMNI 데이터의 50개 컬럼 에러값 자동 처리
- **시간 매칭**: 이미지 시퀀스와 태양풍 데이터 시간 동기화
- **HDF5 출력**: 압축된 HDF5 형식으로 최종 데이터셋 생성

## 시스템 요구사항

- **Python**: 3.6 이상
- **OS**: Linux, macOS 지원 (Windows는 일부 제한)
- **메모리**: 최소 8GB RAM (대용량 이미지 처리시 16GB 권장)
- **저장공간**: 처리하는 기간에 따라 수GB~TB

## 설치 및 설정

### 1. 저장소 클론
```bash
git clone <repository-url>
cd solar_data_pipeline
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 설정
```bash
source setup.sh
```

## 디렉토리 구조

```
solar_data_pipeline/
├── core/                    # 핵심 모듈
│   ├── base_downloader.py   # 공통 다운로더 기반 클래스
│   └── utils.py             # 유틸리티 함수
├── downloaders/             # 데이터 다운로더
│   ├── get_sdo.py           # SDO AIA JP2 이미지
│   ├── get_lasco.py         # SOHO LASCO 코로나그래프
│   ├── get_secchi.py        # STEREO SECCHI 데이터
│   └── get_omni.py          # NASA OMNI 태양풍 데이터
├── processors/              # 데이터 처리기
│   ├── process_sdo.py       # SDO 이미지 리사이징/시퀀스 생성
│   ├── process_omni.py      # OMNI 데이터 시간 매칭
│   └── make_dataset.py      # 최종 HDF5 데이터셋 생성
├── config/                  # 설정 파일
│   └── omni_config.yaml     # OMNI 50개 컬럼 정의
├── scripts/                 # 실행 스크립트
└── data/                    # 데이터 저장소
    ├── raw/                 # 원본 다운로드 데이터
    ├── processed/           # 처리된 시퀀스 데이터
    └── final/               # 최종 HDF5 데이터셋
```

## 사용법

### 1. 빠른 시작 (전체 파이프라인)

```bash
cd scripts
./run_all_pipeline.sh
```

### 2. 개별 단계 실행

#### 2.1 데이터 다운로드
```bash
# SDO AIA 이미지 다운로드
./run_sdo_example.sh

# LASCO 코로나그래프 다운로드
./run_lasco_example.sh

# SECCHI 데이터 다운로드
./run_secchi_example.sh

# OMNI 태양풍 데이터 다운로드
./run_omni_example.sh
```

#### 2.2 데이터 처리
```bash
# SDO 이미지 처리 (리사이징, 시퀀스 생성)
./run_process_sdo_example.sh

# OMNI 데이터 매칭 (시간 동기화)
./run_process_omni_example.sh
```

### 3. Python 직접 실행

```bash
# SDO 이미지 다운로드 (2024년 1월 1-7일, 193/211nm)
python downloaders/get_sdo.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-07 \
    --waves 193 211 \
    --destination data/raw/sdo_jp2/aia \
    --parallel 4

# SDO 이미지 처리
python processors/process_sdo.py \
    --input-dir data/raw/sdo_jp2/aia \
    --output-dir data/processed \
    --start-year 2024 --start-month 1 --start-day 1 \
    --end-year 2024 --end-month 1 --end-day 7 \
    --waves 193 211 \
    --parallel --max-workers 6

# OMNI 데이터 다운로드
python downloaders/get_omni.py \
    --dataset omni_low_res \
    --year 2024 \
    --output-dir omni_data

# OMNI 데이터 매칭
python processors/process_omni.py \
    --dataset-dir data/processed \
    --omni-dir omni_data \
    --parallel --max-workers 6
```

## 설정 옵션

### SDO 다운로더
- `--waves`: 파장 선택 (94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500)
- `--parallel`: 병렬 다운로드 개수
- `--start-date`, `--end-date`: 날짜 범위

### LASCO 다운로더  
- `--cameras`: 카메라 선택 (c1, c2, c3, c4)
- `--extensions`: 파일 형식 (fts, fits)

### SECCHI 다운로더
- `--spacecrafts`: 위성 선택 (ahead, behind)
- `--instruments`: 기기 선택 (cor1, cor2, euvi, hi_1, hi_2)
- `--data-types`: 데이터 타입 (science, beacon)

### OMNI 처리기
- `--dataset`: 데이터셋 선택 (omni_low_res, omni_high_res)
- `--columns`: 특정 컬럼만 추출 (기본값: 전체 50개 컬럼)

## 데이터 흐름

1. **다운로드**: 원본 데이터를 `data/raw/`에 저장
2. **전처리**: SDO 이미지를 64x64로 리사이징하여 `data/processed/`에 시퀀스 생성
3. **시간 매칭**: 각 이미지 시퀀스에 대응하는 OMNI 데이터를 `hourly_data.csv`로 생성
4. **데이터셋 생성**: 이미지와 태양풍 데이터를 결합하여 `data/final/`에 HDF5 파일 생성

## 출력 데이터 형식

최종 HDF5 파일 구조:
```
YYYYMMDDHH.h5
├── sdo_193          # (20, 1, 64, 64) - 193nm 이미지 시퀀스
├── sdo_211          # (20, 1, 64, 64) - 211nm 이미지 시퀀스  
├── omni_Bz_GSM      # (N,) - GSM 좌표계 Z방향 자기장
├── omni_Temperature # (N,) - 양성자 온도
├── omni_Flow_speed  # (N,) - 태양풍 속도
└── ...              # 기타 OMNI 파라미터들
```

## OMNI 데이터 컬럼

50개의 태양풍/자기장/지자기 활동 파라미터를 포함:

**자기장**: Bx_GSE, By_GSE, Bz_GSE, By_GSM, Bz_GSM, B_magnitude  
**태양풍**: Flow_speed, Proton_density, Temperature, Flow_pressure  
**전기장**: Electric_field, Plasma_beta, Alfven_mach  
**지자기**: DST_index, AE_index, Kp_index, ap_index  
**태양활동**: R_sunspot, f107_index  
**에너지입자**: Proton_flux_1MeV, Proton_flux_10MeV 등

## 로깅

모든 실행 로그는 `logs/` 디렉토리에 저장됩니다:
- `sdo_download.log`: SDO 다운로드 로그
- `lasco_download.log`: LASCO 다운로드 로그  
- `omni_processing.log`: OMNI 처리 로그

## 문제 해결

### 일반적인 문제

**1. Import 에러**
```bash
# PYTHONPATH 설정 확인
source setup.sh
```

**2. 메모리 부족**
```bash
# 병렬 워커 수 줄이기
python processors/process_sdo.py --max-workers 2
```

**3. 네트워크 타임아웃**
```bash
# 재시도 횟수 증가
python downloaders/get_sdo.py --max-retries 10
```

**4. 디스크 공간 부족**
```bash
# 처리 완료된 원본 데이터 삭제
rm -rf data/raw/*
```

### 성능 최적화

- **병렬 처리**: CPU 코어 수에 맞게 `--max-workers` 조정
- **메모리 사용**: 대용량 처리시 배치 크기 조정  
- **네트워크**: 안정적인 인터넷 연결 사용

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트나 기능 개선 제안은 이슈로 등록해 주시기 바랍니다.

## 데이터 출처

- **SDO**: NASA Solar Dynamics Observatory
- **LASCO**: SOHO Large Angle and Spectrometric Coronagraph
- **SECCHI**: STEREO Sun Earth Connection Coronal and Heliospheric Investigation  
- **OMNI**: NASA GSFC OMNI database