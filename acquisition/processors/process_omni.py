#!/usr/bin/env python3
"""
Simplified OMNI Data Processor
Extracts time-matched OMNI data for SDO image sequences
파일 위치: processors/process_omni.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from glob import glob
import argparse
from concurrent.futures import ProcessPoolExecutor

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from core.utils import TimeSeriesProcessor, DirectoryTimeExtractor, setup_logging

def process_single_dataset(dataset_dir, omni_dir, interval_hours, sequence_days, time_offset):
    """단일 데이터셋 처리 (병렬 처리용 워커 함수)"""
    # 이 워커에 대한 프로세서 생성 (특정 컬럼 지정 안함 = 모든 컬럼)
    processor = TimeSeriesProcessor(omni_dir, columns=None)
    
    # 디렉토리 이름에서 시작 시간 추출
    dir_name = os.path.basename(dataset_dir)
    start_time = DirectoryTimeExtractor.extract_from_pattern(dir_name)
    
    if start_time is None:
        print(f"Cannot extract time from directory: {dir_name}")
        return False
    
    # 종료 시간 계산
    end_time = start_time + timedelta(days=sequence_days)
    
    # 출력 파일 경로
    output_path = os.path.join(dataset_dir, "hourly_data.csv")
    
    # 이미 존재하면 스킵
    if os.path.exists(output_path):
        print(f"Already processed: {dir_name}")
        return True
    
    print(f"Processing {dir_name}: {start_time} to {end_time}")
    
    # 시간 범위 처리
    results = processor.process_time_range(start_time, end_time, interval_hours, time_offset)
    
    # 성공적인 추출 개수 계산
    success_count = sum(1 for r in results if len(r) > 2)  # base_time, target_time보다 많은 데이터
    
    # 결과 저장
    if processor.save_results(results, output_path):
        print(f"Saved {len(results)} records ({success_count} with data) to {output_path}")
        return True
    else:
        print(f"Failed to save results for {dir_name}")
        return False

def process_dataset_sequential(dataset_dir, processor, interval_hours, sequence_days, time_offset_hours):
    """단일 데이터셋 처리 (순차 처리용)"""
    # 디렉토리 이름에서 시작 시간 추출
    dir_name = os.path.basename(dataset_dir)
    start_time = DirectoryTimeExtractor.extract_from_pattern(dir_name)
    
    if start_time is None:
        print(f"Cannot extract time from directory: {dir_name}")
        return False
    
    # 종료 시간 계산
    end_time = start_time + timedelta(days=sequence_days)
    
    # 출력 파일 경로
    output_path = os.path.join(dataset_dir, "hourly_data.csv")
    
    # 이미 존재하면 스킵
    if os.path.exists(output_path):
        print(f"Already processed: {dir_name}")
        return True
    
    print(f"Processing {dir_name}: {start_time} to {end_time}")
    
    # 시간 범위 처리
    results = processor.process_time_range(start_time, end_time, interval_hours, time_offset_hours)
    
    # 성공적인 추출 개수 계산 (시간 컬럼보다 많은 데이터가 있는 행)
    success_count = sum(1 for r in results if len(r) > 2)
    
    # 결과 저장
    if processor.save_results(results, output_path):
        print(f"Saved {len(results)} records ({success_count} with data) to {output_path}")
        return True
    else:
        print(f"Failed to save results for {dir_name}")
        return False

def main():
    parser = argparse.ArgumentParser(description="OMNI Data Processor")
    
    # 입출력 경로
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Directory containing processed SDO datasets')
    parser.add_argument('--omni-dir', type=str, required=True,
                       help='Directory containing OMNI CSV files')
    
    # OMNI 컬럼 추출 (선택사항 - 지정하지 않으면 모든 컬럼 추출)
    parser.add_argument('--columns', type=str, nargs='+', default=None,
                       help='Specific OMNI columns to extract (default: all columns)')
    
    # 처리 파라미터
    parser.add_argument('--interval', type=int, default=3,
                       help='Time interval in hours')
    parser.add_argument('--sequence-days', type=int, default=8,
                       help='Sequence length in days')
    parser.add_argument('--time-offset', type=int, default=-1,
                       help='Time offset in hours (negative for past data)')
    
    # 처리 옵션
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum parallel workers')
    
    # 로깅
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # 디렉토리 검증
    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset directory not found: {args.dataset_dir}")
        return 1
    
    if not os.path.isdir(args.omni_dir):
        print(f"OMNI directory not found: {args.omni_dir}")
        return 1
    
    # 데이터셋 디렉토리 찾기
    dataset_dirs = sorted([d for d in glob(f"{args.dataset_dir}/*") 
                          if os.path.isdir(d) and os.path.basename(d)[0].isdigit()])
    
    if not dataset_dirs:
        print(f"No valid dataset directories found in {args.dataset_dir}")
        return 1
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    if args.columns:
        print(f"OMNI columns: {args.columns}")
    else:
        print("Extracting all OMNI columns")
    print(f"Time offset: {args.time_offset} hours")
    print(f"Processing interval: {args.interval} hours")
    print(f"Sequence duration: {args.sequence_days} days")
    
    # 데이터셋 처리
    if args.parallel:
        # 병렬 처리
        from multiprocessing import cpu_count
        
        max_workers = args.max_workers or min(cpu_count(), len(dataset_dirs))
        print(f"Using {max_workers} parallel workers")
        
        # 워커 함수 인자 준비
        worker_args = [(dataset_dir, args.omni_dir, args.interval, 
                       args.sequence_days, args.time_offset) for dataset_dir in dataset_dirs]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_dataset, *zip(*worker_args)))
        
        successful = sum(results)
        
    else:
        # 순차 처리
        print("Processing sequentially")
        processor = TimeSeriesProcessor(args.omni_dir, args.columns)
        successful = 0
        
        for i, dataset_dir in enumerate(dataset_dirs, 1):
            print(f"[{i}/{len(dataset_dirs)}]", end=" ")
            if process_dataset_sequential(dataset_dir, processor, args.interval, 
                                        args.sequence_days, args.time_offset):
                successful += 1
    
    # 최종 결과
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful}/{len(dataset_dirs)} datasets")
    
    if successful < len(dataset_dirs):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())