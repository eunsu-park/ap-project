#!/bin/bash

# LASCO 코로나그래프 데이터 다운로더 실행 예시

echo "=== LASCO Data Downloader Examples ==="

# 1. 기본 실행 (최근 7일, C2 카메라)
echo "1. 기본 실행 - 최근 7일, C2 카메라"
python ../downloaders/get_lasco.py \
    --cameras c2 \
    --destination ./data/lasco \
    --parallel 4 \
    --log-level INFO

echo -e "\n"

# 2. 특정 날짜 범위, 여러 카메라
echo "2. 특정 날짜 범위 - C1, C2, C3 카메라"
python ../downloaders/get_lasco.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-05 \
    --cameras c1 c2 c3 \
    --destination ./data/lasco \
    --parallel 6 \
    --overwrite

echo -e "\n"

# 3. 최근 30일 (모든 카메라)
echo "3. 최근 30일 - 모든 카메라"
python ../downloaders/get_lasco.py \
    --days 30 \
    --cameras c1 c2 c3 c4 \
    --extensions fts fits \
    --destination ./data/lasco \
    --parallel 8

echo -e "\n"

# 4. 특정 기간 (태양 활동 극대기)
echo "4. 특정 기간 다운로드 - 2014년 태양 극대기"
python ../downloaders/get_lasco.py \
    --start-date 2014-01-01 \
    --end-date 2014-01-31 \
    --cameras c2 c3 \
    --destination ./data/lasco/solar_max_2014 \
    --parallel 4 \
    --max-retries 5

echo -e "\n"

# 5. 로그 파일 저장 및 상세 로깅
echo "5. 로그 파일과 함께 상세 다운로드"
python ../downloaders/get_lasco.py \
    --start-date 2024-06-01 \
    --end-date 2024-06-07 \
    --cameras c2 \
    --destination ./data/lasco \
    --parallel 2 \
    --log-level DEBUG \
    --log-file lasco_download.log

echo "LASCO 다운로더 실행 예시 완료"