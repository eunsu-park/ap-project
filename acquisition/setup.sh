#!/bin/bash
# 파일 위치: setup.sh (프로젝트 루트)

echo "=== 태양 데이터 처리 파이프라인 설정 ==="

# 1. PYTHONPATH 설정
export PYTHONPATH=$PWD:$PYTHONPATH
echo "PYTHONPATH 설정 완료: $PWD"

# 2. 필요한 디렉토리 생성
echo "디렉토리 구조 생성 중..."

mkdir -p core
mkdir -p downloaders  
mkdir -p processors
mkdir -p config
mkdir -p scripts
mkdir -p data/raw/sdo_jp2/aia
mkdir -p data/raw/lasco
mkdir -p data/raw/secchi
mkdir -p data/processed
mkdir -p data/final
mkdir -p omni_data
mkdir -p logs
mkdir -p tests

echo "디렉토리 구조 생성 완료"

# 3. 실행 권한 설정
echo "스크립트 실행 권한 설정 중..."
chmod +x scripts/*.sh

# 4. __init__.py 파일 생성
touch core/__init__.py
touch downloaders/__init__.py  
touch processors/__init__.py
touch tests/__init__.py

echo "초기 설정 완료!"
echo ""
echo "사용 방법:"
echo "1. 개별 다운로더 실행: cd scripts && ./run_sdo_example.sh"
echo "2. 전체 파이프라인: cd scripts && ./run_all_pipeline.sh"
echo "3. Python 직접 실행: python downloaders/get_sdo.py --help"
echo ""
echo "주의: 매번 새 터미널에서는 'source setup.sh'를 실행해주세요."
