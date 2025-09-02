#!/bin/bash

# 전체 태양 데이터 처리 파이프라인 실행 예시
# 파일 위치: scripts/run_all_pipeline.sh

echo "=== Complete Solar Data Processing Pipeline ==="

# 설정 변수
START_DATE="2024-01-01"
END_DATE="2024-01-07"
DATA_ROOT="../data"
OMNI_ROOT="../omni_data"
CONFIG_DIR="../config"

# 디렉토리 생성
mkdir -p $DATA_ROOT/raw
mkdir -p $DATA_ROOT/processed  
mkdir -p $DATA_ROOT/final
mkdir -p $OMNI_ROOT
mkdir -p ../logs

echo "Processing date range: $START_DATE to $END_DATE"
echo "Data root: $DATA_ROOT"
echo ""

# 1단계: OMNI 데이터 다운로드
echo "=== Step 1: Download OMNI data ==="
python ../downloaders/get_omni.py \
    --dataset omni_low_res \
    --year 2024 \
    --output-dir $OMNI_ROOT \
    --config $CONFIG_DIR/omni_config.yaml \
    --log-level INFO

echo -e "\n"

# 2단계: SDO 이미지 다운로드
echo "=== Step 2: Download SDO images ==="
python ../downloaders/get_sdo.py \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --waves 193 211 \
    --destination $DATA_ROOT/raw/sdo_jp2/aia \
    --parallel 4 \
    --log-level INFO

echo -e "\n"

# 3단계: LASCO 데이터 다운로드
echo "=== Step 3: Download LASCO data ==="
python ../downloaders/get_lasco.py \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --cameras c2 c3 \
    --destination $DATA_ROOT/raw/lasco \
    --parallel 4 \
    --log-level INFO

echo -e "\n"

# 4단계: SECCHI 데이터 다운로드
echo "=== Step 4: Download SECCHI data ==="
python ../downloaders/get_secchi.py \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --spacecrafts ahead \
    --instruments cor2 \
    --destination $DATA_ROOT/raw/secchi \
    --parallel 4 \
    --log-level INFO

echo -e "\n"

# 5단계: SDO 이미지 처리
echo "=== Step 5: Process SDO images ==="
python ../processors/process_sdo.py \
    --input-dir $DATA_ROOT/raw/sdo_jp2/aia \
    --output-dir $DATA_ROOT/processed \
    --start-year 2024 --start-month 1 --start-day 1 \
    --end-year 2024 --end-month 1 --end-day 7 \
    --waves 193 211 \
    --parallel \
    --max-workers 6 \
    --skip-existing

echo -e "\n"

# 6단계: OMNI 데이터 매칭
echo "=== Step 6: Match OMNI data with SDO sequences ==="
python ../processors/process_omni.py \
    --dataset-dir $DATA_ROOT/processed \
    --omni-dir $OMNI_ROOT \
    --interval 3 \
    --sequence-days 8 \
    --time-offset -1 \
    --parallel \
    --max-workers 6

echo -e "\n"

# 7단계: 최종 데이터셋 생성
echo "=== Step 7: Create final dataset ==="
cd ../processors
python make_dataset.py

echo -e "\n"

# 처리 결과 확인
echo "=== Processing Summary ==="
echo "SDO processed datasets:"
ls -la $DATA_ROOT/processed/ | wc -l
echo "LASCO files:"
find $DATA_ROOT/raw/lasco -name "*.fts" | wc -l
echo "SECCHI files:"  
find $DATA_ROOT/raw/secchi -name "*.fts" | wc -l
echo "OMNI matched files:"
find $DATA_ROOT/processed -name "hourly_data.csv" | wc -l

echo "Complete pipeline execution finished!"