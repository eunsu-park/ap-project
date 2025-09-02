#!/bin/bash

# OMNI 데이터 처리기 실행 예시

echo "=== OMNI Data Processor Examples ==="

# 1. 기본 실행 (저해상도 데이터, 2024년)
echo "1. 기본 실행 - 저해상도 데이터 2024년"
python ../downloaders/get_omni.py \
    --dataset omni_low_res \
    --year 2024 \
    --output-dir ./omni_data \
    --config omni_config.yaml

echo -e "\n"

# 2. 고해상도 데이터 (1분 해상도)
echo "2. 고해상도 1분 데이터 다운로드"
python ../downloaders/get_omni.py \
    --dataset omni_high_res \
    --start-year 2023 \
    --end-year 2024 \
    --output-dir ./omni_data/high_res \
    --overwrite

echo -e "\n"

# 3. 특정 연도 범위
echo "3. 특정 연도 범위 (2020-2024)"
python ../downloaders/get_omni.py \
    --dataset omni_low_res \
    --start-year 2020 \
    --end-year 2024 \
    --output-dir ./omni_data/multi_year

echo -e "\n"

# 4. 로그 파일과 함께
echo "4. 디버그 모드로 실행"
python ../downloaders/get_omni.py \
    --dataset omni_low_res \
    --year 2024 \
    --output-dir ./omni_data \
    --log-level DEBUG \
    --log-file omni_download.log

echo -e "\n"

# 5. 여러 연도 배치 처리
echo "5. 여러 연도 배치 처리 (2010-2024)"
python ../downloaders/get_omni.py \
    --dataset omni_low_res \
    --start-year 2010 \
    --end-year 2024 \
    --output-dir ./omni_data/historical \
    --log-level INFO

echo "OMNI 데이터 처리기 실행 예시 완료"