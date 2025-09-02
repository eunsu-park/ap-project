#!/bin/bash

# OMNI 데이터 처리기 실행 예시 (SDO 이미지 시퀀스와 매칭)

echo "=== OMNI Data Processor Examples ==="

# 1. 기본 실행 (순차 처리, 모든 컬럼)
echo "1. 기본 순차 처리 - 모든 OMNI 컬럼 추출"
python ../processors/process_omni.py \
    --dataset-dir ./data/processed \
    --omni-dir ./omni_data \
    --interval 3 \
    --sequence-days 8 \
    --time-offset -1

echo -e "\n"

# 2. 병렬 처리 (8개 워커)
echo "2. 병렬 처리 - 8개 워커로 빠른 처리"
python ../processors/process_omni.py \
    --dataset-dir ./data/processed \
    --omni-dir ./omni_data \
    --interval 3 \
    --sequence-days 8 \
    --time-offset -1 \
    --parallel \
    --max-workers 8

echo -e "\n"

# 3. 특정 컬럼만 추출
echo "3. 핵심 태양풍 파라미터만 추출"
python ../processors/process_omni.py \
    --dataset-dir ./data/processed \
    --omni-dir ./omni_data \
    --columns Bz_GSM Temperature Proton_density Flow_speed Electric_field \
    --interval 3 \
    --sequence-days 8 \
    --time-offset -1 \
    --parallel \
    --max-workers 4

echo -e "\n"

# 4. 고밀도 시계열 (1시간 간격)
echo "4. 고밀도 시계열 매칭 - 1시간 간격"
python ../processors/process_omni.py \
    --dataset-dir ./data/processed_dense \
    --omni-dir ./omni_data \
    --interval 1 \
    --sequence-days 5 \
    --time-offset -1 \
    --parallel \
    --max-workers 6

echo -e "\n"

# 5. 자기장 데이터 중심 추출
echo "5. 자기장 중심 파라미터 추출"
python ../processors/process_omni.py \
    --dataset-dir ./data/processed \
    --omni-dir ./omni_data \
    --columns Bx_GSE By_GSE Bz_GSE By_GSM Bz_GSM B_magnitude B_mag_vector \
    --interval 3 \
    --sequence-days 8 \
    --time-offset -2 \
    --parallel \
    --max-workers 4

echo -e "\n"

# 6. 태양풍-자기권 상호작용 파라미터
echo "6. 태양풍-자기권 상호작용 파라미터"
python ../processors/process_omni.py \
    --dataset-dir ./data/processed \
    --omni-dir ./omni_data \
    --columns Bz_GSM Flow_speed Proton_density Temperature Electric_field \
               Flow_pressure Plasma_beta Alfven_mach DST_index AE_index \
    --interval 3 \
    --sequence-days 8 \
    --time-offset -1 \
    --parallel \
    --max-workers 6

echo -e "\n"

# 7. 디버그 모드 (순차 처리, 상세 로그)
echo "7. 디버그 모드 - 상세 로그"
python ../processors/process_omni.py \
    --dataset-dir ./data/processed \
    --omni-dir ./omni_data \
    --interval 3 \
    --sequence-days 8 \
    --time-offset -1 \
    --log-level DEBUG

echo "OMNI 데이터 처리기 실행 예시 완료"