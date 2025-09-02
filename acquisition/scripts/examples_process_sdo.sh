#!/bin/bash

# SDO 이미지 처리기 실행 예시

echo "=== SDO Image Processor Examples ==="

# 1. 기본 실행 (순차 처리)
echo "1. 기본 순차 처리 - 2024년 1월"
python ../processors/process_sdo.py \
    --input-dir /path/to/sdo_jp2/aia \
    --output-dir ./data/processed \
    --start-year 2024 --start-month 1 --start-day 1 \
    --end-year 2024 --end-month 1 --end-day 7 \
    --waves 193 211 \
    --skip-existing

echo -e "\n"

# 2. 병렬 처리 (8개 워커)
echo "2. 병렬 처리 - 6개월 데이터"
python ../processors/process_sdo.py \
    --input-dir /path/to/sdo_jp2/aia \
    --output-dir ./data/processed \
    --start-year 2024 --start-month 1 --start-day 1 \
    --end-year 2024 --end-month 6 --end-day 30 \
    --waves 193 211 \
    --parallel \
    --max-workers 8 \
    --skip-existing

echo -e "\n"

# 3. 고밀도 시계열 (1시간 간격, 40개 시퀀스)
echo "3. 고밀도 시계열 처리"
python ../processors/process_sdo.py \
    --input-dir /path/to/sdo_jp2/aia \
    --output-dir ./data/processed_dense \
    --start-year 2024 --start-month 6 --start-day 1 \
    --end-year 2024 --end-month 6 --end-day 7 \
    --waves 193 211 304 \
    --time-step 3 \
    --num-sequence 40 \
    --interval 1 \
    --parallel \
    --max-workers 6

echo -e "\n"

# 4. 전체 재처리 (기존 파일 무시)
echo "4. 전체 재처리 - 기존 파일 덮어쓰기"
python ../processors/process_sdo.py \
    --input-dir /path/to/sdo_jp2/aia \
    --output-dir ./data/processed \
    --start-year 2024 --start-month 1 --start-day 1 \
    --end-year 2024 --end-month 1 --end-day 31 \
    --waves 193 211 \
    --no-skip \
    --parallel \
    --max-workers 4

echo -e "\n"

# 5. 장기간 배치 처리 (1년간)
echo "5. 장기간 배치 처리 - 2024년 전체"
python ../processors/process_sdo.py \
    --input-dir /path/to/sdo_jp2/aia \
    --output-dir ./data/processed_2024 \
    --start-year 2024 --start-month 1 --start-day 1 \
    --end-year 2024 --end-month 12 --end-day 31 \
    --waves 193 211 \
    --time-step 6 \
    --num-sequence 20 \
    --interval 3 \
    --parallel \
    --max-workers 12 \
    --skip-existing

echo "SDO 이미지 처리기 실행 예시 완료"