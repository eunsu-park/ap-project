#!/bin/bash

# SDO JP2 데이터 다운로더 실행 예시
# 파일 위치: scripts/run_sdo_example.sh

echo "=== SDO JP2 Data Downloader Examples ==="

# 1. 기본 실행 (최근 7일, 193/211nm)
echo "1. 기본 실행 - 최근 7일"
python ../downloaders/get_sdo.py \
    --waves 193 211 \
    --destination ../data/raw/sdo_jp2/aia \
    --parallel 4 \
    --log-level INFO

echo -e "\n"

# 2. 특정 날짜 범위
echo "2. 특정 날짜 범위 다운로드"
python ../downloaders/get_sdo.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-03 \
    --waves 193 211 304 \
    --destination ./data/sdo_jp2/aia \
    --parallel 8 \
    --overwrite

echo -e "\n"

# 3. 단일 날짜
echo "3. 단일 날짜 다운로드"
python ../downloaders/get_sdo.py \
    --year 2024 \
    --month 1 \
    --day 15 \
    --waves 193 211 \
    --destination ./data/sdo_jp2/aia

echo -e "\n"

# 4. 최근 30일 (모든 파장)
echo "4. 최근 30일 - 모든 파장"
python ../downloaders/get_sdo.py \
    --days 30 \
    --waves 94 131 171 193 211 304 335 1600 1700 4500 \
    --destination ./data/sdo_jp2/aia \
    --parallel 6 \
    --max-retries 5

echo -e "\n"

# 5. 로그 파일 저장
echo "5. 로그 파일과 함께 실행"
python ../downloaders/get_sdo.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-07 \
    --waves 193 211 \
    --destination ./data/sdo_jp2/aia \
    --parallel 4 \
    --log-level DEBUG \
    --log-file sdo_download.log

echo "SDO 다운로더 실행 예시 완료"