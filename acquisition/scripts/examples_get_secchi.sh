#!/bin/bash

# SECCHI (STEREO) 데이터 다운로더 실행 예시

echo "=== SECCHI Data Downloader Examples ==="

# 1. 기본 실행 (최근 7일, STEREO-A COR2)
echo "1. 기본 실행 - 최근 7일, STEREO-A COR2"
python ../downloaders/get_secchi.py \
    --data-types science \
    --spacecrafts ahead \
    --instruments cor2 \
    --destination ./data/secchi \
    --parallel 4

echo -e "\n"

# 2. 양쪽 STEREO 위성 (A, B)
echo "2. STEREO-A, STEREO-B COR1/COR2 다운로드"
python ../downloaders/get_secchi.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-03 \
    --data-types science \
    --spacecrafts ahead behind \
    --instruments cor1 cor2 \
    --destination ./data/secchi \
    --parallel 8 \
    --overwrite

echo -e "\n"

# 3. HI (Heliospheric Imager) 데이터
echo "3. HI (태양권 이미저) 데이터 다운로드"
python ../downloaders/get_secchi.py \
    --start-date 2024-01-15 \
    --end-date 2024-01-20 \
    --spacecrafts ahead \
    --instruments hi_1 hi_2 \
    --destination ./data/secchi/hi \
    --parallel 4

echo -e "\n"

# 4. 실시간 비콘 데이터
echo "4. 실시간 비콘 데이터 다운로드"
python ../downloaders/get_secchi.py \
    --days 3 \
    --data-types beacon \
    --spacecrafts ahead \
    --categories img \
    --instruments cor2 euvi \
    --destination ./data/secchi/beacon \
    --parallel 6

echo -e "\n"

# 5. EUVI (극자외선 이미저) 데이터
echo "5. EUVI 극자외선 이미저 데이터"
python ../downloaders/get_secchi.py \
    --start-date 2024-06-01 \
    --end-date 2024-06-07 \
    --spacecrafts ahead behind \
    --instruments euvi \
    --categories img cal \
    --destination ./data/secchi/euvi \
    --parallel 4 \
    --max-retries 5

echo -e "\n"

# 6. 종합 다운로드 (모든 기기)
echo "6. 종합 다운로드 - 모든 SECCHI 기기"
python ../downloaders/get_secchi.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-02 \
    --data-types science \
    --spacecrafts ahead \
    --categories img \
    --instruments cor1 cor2 euvi hi_1 hi_2 \
    --destination ./data/secchi/comprehensive \
    --parallel 8 \
    --log-level DEBUG \
    --log-file secchi_comprehensive.log

echo "SECCHI 다운로더 실행 예시 완료"