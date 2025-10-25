# 파일 위치: processors/process_sdo.py

import os
import datetime
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
import argparse

import cv2
import numpy as np

def resize_image_optimized(img_array, target_size=(64, 64)):
    """이미지 리사이징 최적화 - OpenCV 사용"""
    if img_array is None:
        return None
    
    # OpenCV resize가 bin_ndarray보다 빠름 (네이티브 최적화)
    # INTER_AREA는 다운샘플링에 최적
    img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
    return np.clip(img_resized, 0, 255).astype(np.uint8)

def run_optimized(file_path, save_path, skip_existing=True):
    """최적화된 개별 이미지 파일 처리"""
    if not os.path.exists(file_path):
        return False
    
    # 기존 파일 스킵 옵션
    if skip_existing and os.path.exists(save_path):
        # 파일 크기가 0이 아니면 정상 처리된 것으로 간주
        if os.path.getsize(save_path) > 0:
            return True
    
    try:
        # 이미지 읽기
        img_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img_array is None:
            return False
        
        # 최적화된 리사이징
        img_resized = resize_image_optimized(img_array)
        if img_resized is None:
            return False
        
        # 압축 저장
        success = cv2.imwrite(save_path, img_resized, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000])
        return success
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main_optimized(date_target, args):
    """기준 시간으로부터 이전 시퀀스를 구성하는 처리 함수"""
    try:
        # 1. 시퀀스 시작/종료 시간 계산
        # date_target이 기준 시간 (마지막 시퀀스)
        # 시퀀스: date_target - (num_sequence - 1) * time_step ~ date_target
        total_sequence = args.num_sequence
        
        date_start = date_target - datetime.timedelta(hours=args.time_step * (args.num_sequence))
        date_end = date_target - datetime.timedelta(hours=args.time_step)
        
        # 2. 출력 디렉토리 설정 (기준 시간을 폴더명으로 사용)
        date_dir = f"{args.output_dir}/{date_target:%Y%m%d%H}"
        
        # 기존 완료 체크 (디렉토리가 존재할 경우에만)
        if args.skip_existing and os.path.exists(date_dir):
            expected_files = total_sequence * len(args.waves)
            existing_files = 0
            
            # 정확한 파일 수 체크: 예상 파일 수와 실제 파일 수가 정확히 일치해야 함
            for wave in args.waves:
                save_dir = f"{date_dir}/{wave:d}"
                if os.path.exists(save_dir):
                    valid_files = [f for f in os.listdir(save_dir) 
                                 if f.endswith('.jp2') and os.path.getsize(f"{save_dir}/{f}") > 0]
                    existing_files += len(valid_files)
            
            # 정확히 일치해야 완료된 것으로 간주
            if existing_files == expected_files:
                print(f"{date_target} : Already processed ({existing_files}/{expected_files} files). Skipping.")
                return True
            elif existing_files > 0:
                print(f"{date_target} : Incomplete processing found ({existing_files}/{expected_files} files). Will reprocess.")
                # 불완전한 디렉토리 삭제
                import shutil
                shutil.rmtree(date_dir)
                print(f"{date_target} : Removed incomplete directory")

        # 3. 파일 수집 및 유효성 검사
        filelists = [[] for _ in args.waves]
        date_current = date_start
        
        for i in range(total_sequence):
            year = date_current.year
            month = date_current.month
            day = date_current.day
            hour = date_current.hour

            target_datetime = datetime.datetime(year, month, day, hour)
            
            for wave_idx, wave in enumerate(args.waves):
                data_path = f"{args.input_dir}/{wave:d}/{year:04d}/{year:04d}{month:02d}{day:02d}"
                data_name = f"{year:04d}_{month:02d}_{day:02d}__{hour:02d}_00_*_*__SDO_AIA_AIA_{wave:d}.jp2"
                data_list = sorted(glob(f"{data_path}/{data_name}"))
                print(data_path, data_name, len(data_list))
                
                found = False
                for file_path in data_list:
                    file_name = file_path.split("/")[-1]
                    try:
                        file_date = datetime.datetime.strptime(file_name[:23], "%Y_%m_%d__%H_%M_%S_%f")

                        # if file_date - target_datetime < 60 :
                        if file_date.second < 60:
                            filelists[wave_idx].append(file_path)
                            found = True
                            break
                    except ValueError:
                        continue
                
                if not found:
                    filelists[wave_idx].append("")
                    
            date_current += datetime.timedelta(hours=args.time_step)

        # 4. 유효성 검사 (디렉토리 생성 전에 확인)
        for wave_idx, file_list in enumerate(filelists):
            if len(file_list) != total_sequence:
                print(f"{date_target} : Invalid file list length for wave {args.waves[wave_idx]}. Skipping.")
                return False
            for i, file_path in enumerate(file_list):
                if not file_path or not os.path.exists(file_path):
                    print(f"{date_target} : Missing file for wave {args.waves[wave_idx]}, sequence {i}. Skipping.")
                    return False

        # 5. 여기서부터 디렉토리 생성 (모든 파일이 존재함을 확인한 후)
        print(f"{date_target} : All files found. Creating directories and processing...")
        
        for wave in args.waves:
            save_dir = f"{date_dir}/{wave:d}"
            os.makedirs(save_dir, exist_ok=True)

        # 6. 파일 처리
        total_files = total_sequence * len(args.waves)
        processed = 0
        
        for wave_idx, (wave, file_list) in enumerate(zip(args.waves, filelists)):
            save_dir = f"{date_dir}/{wave:d}"
            
            for file_path in file_list:
                file_name = os.path.basename(file_path)
                save_path = f"{save_dir}/{file_name}"
                
                if run_optimized(file_path, save_path, args.skip_existing):
                    processed += 1
                
                # 간헐적 진행률 출력
                if processed % 10 == 0:
                    print(f"{date_target} : {processed/total_files:.1%}", end="\r", flush=True)
        
        success_rate = processed / total_files
        print(f"{date_target} : Done ({processed}/{total_files} files, {success_rate:.1%})")
        return processed == total_files  # 모든 파일이 성공적으로 처리되어야 성공
        
    except Exception as e:
        print(f"Error processing {date_target}: {e}")
        # 처리 실패 시 빈 디렉토리 정리
        try:
            if 'date_dir' in locals() and os.path.exists(date_dir):
                # 디렉토리가 비어있거나 거의 비어있으면 삭제
                total_files = 0
                for wave in args.waves:
                    save_dir = f"{date_dir}/{wave:d}"
                    if os.path.exists(save_dir):
                        total_files += len([f for f in os.listdir(save_dir) if f.endswith('.jp2')])
                
                if total_files < total_sequence * len(args.waves) * 0.1:  # 10% 미만이면 정리
                    import shutil
                    shutil.rmtree(date_dir)
                    print(f"{date_target} : Cleaned up incomplete directory")
        except:
            pass  # 정리 실패는 무시
        
        return False

def process_sequential(start_date, end_date, args):
    """순차 처리"""
    if start_date.hour % args.interval != 0:
        start_date += datetime.timedelta(hours=args.interval - start_date.hour % args.interval)

    # 처리할 모든 날짜 리스트 생성
    dates_to_process = []
    current_date = start_date
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += datetime.timedelta(hours=args.interval)
    
    print(f"Total dates to process: {len(dates_to_process)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Skip existing files: {args.skip_existing}")
    print("Processing mode: Sequential")
    print(f"Sequence composition: {args.num_sequence} images before reference time (including reference)")
    
    # 순차 처리 실행
    start_time = time.time()
    successful = 0
    failed = 0
    
    try:
        for i, date in enumerate(dates_to_process):
            try:
                if main_optimized(date, args):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Exception for date {date}: {e}")
                failed += 1
            
            # 진행률 출력
            total_processed = successful + failed
            if total_processed % 5 == 0 or total_processed == len(dates_to_process):
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining = len(dates_to_process) - total_processed
                eta = remaining / rate if rate > 0 else 0
                
                print(f"\nProgress: {total_processed}/{len(dates_to_process)} "
                      f"({total_processed/len(dates_to_process):.1%}) - "
                      f"Success: {successful}, Failed: {failed} - "
                      f"Rate: {rate:.2f} dates/sec - ETA: {eta/60:.1f}min")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False
    
    # 최종 결과
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Processing completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successfully processed: {successful}/{len(dates_to_process)} dates")
    print(f"Failed: {failed}/{len(dates_to_process)} dates")
    print(f"Average rate: {len(dates_to_process)/total_time:.2f} dates/sec")
    print(f"{'='*50}")
    
    return True

def process_parallel(start_date, end_date, args):
    """병렬 처리"""
    if start_date.hour % args.interval != 0:
        start_date += datetime.timedelta(hours=args.interval - start_date.hour % args.interval)

    # 처리할 모든 날짜 리스트 생성
    dates_to_process = []
    current_date = start_date
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += datetime.timedelta(hours=args.interval)
    
    print(f"Total dates to process: {len(dates_to_process)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Skip existing files: {args.skip_existing}")
    print(f"Sequence composition: {args.num_sequence} images before reference time (including reference)")
    
    # 워커 수 최적화: CPU 집약적 작업이므로 CPU 코어 수와 동일하게
    max_workers = min(args.max_workers or cpu_count(), len(dates_to_process))
    print(f"Processing mode: Parallel ({max_workers} workers)")
    
    # 병렬 처리 실행
    start_time = time.time()
    successful = 0
    failed = 0
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 모든 작업 제출
            future_to_date = {executor.submit(main_optimized, date, args): date for date in dates_to_process}
            
            # 완료된 작업들을 처리
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Exception for date {date}: {e}")
                    failed += 1
                
                # 진행률 출력
                total_processed = successful + failed
                if total_processed % 5 == 0 or total_processed == len(dates_to_process):
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    remaining = len(dates_to_process) - total_processed
                    eta = remaining / rate if rate > 0 else 0
                    
                    print(f"\nProgress: {total_processed}/{len(dates_to_process)} "
                          f"({total_processed/len(dates_to_process):.1%}) - "
                          f"Success: {successful}, Failed: {failed} - "
                          f"Rate: {rate:.2f} dates/sec - ETA: {eta/60:.1f}min")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        return False
    
    # 최종 결과
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Processing completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successfully processed: {successful}/{len(dates_to_process)} dates")
    print(f"Failed: {failed}/{len(dates_to_process)} dates")
    print(f"Average rate: {len(dates_to_process)/total_time:.2f} dates/sec")
    print(f"{'='*50}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='SDO Image Processor')
    
    # Input/Output directories
    parser.add_argument('--input-dir', type=str, 
                       default='/Volumes/usbshare1/data/sdo_jp2/aia',
                       help='Input directory path')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/eunsupark/ap_project/data/new_processed',
                       help='Output directory path')
    
    # Date range
    parser.add_argument('--start-year', type=int, default=2011, help='Start year')
    parser.add_argument('--start-month', type=int, default=1, help='Start month')
    parser.add_argument('--start-day', type=int, default=1, help='Start day')
    parser.add_argument('--start-hour', type=int, default=0, help='Start hour')
    parser.add_argument('--end-year', type=int, default=2024, help='End year')
    parser.add_argument('--end-month', type=int, default=12, help='End month')
    parser.add_argument('--end-day', type=int, default=31, help='End day')
    parser.add_argument('--end-hour', type=int, default=23, help='End hour')
    
    # Processing parameters
    parser.add_argument('--waves', type=int, nargs='+', default=[193, 211],
                       help='Wavelengths to process')
    parser.add_argument('--time-step', type=int, default=6,
                       help='Time step in hours for sequence')
    parser.add_argument('--num-sequence', type=int, default=40,
                       help='Number of images in sequence before reference time (including reference)')
    parser.add_argument('--interval', type=int, default=3,
                       help='Processing interval in hours')
    
    # Processing options
    parser.add_argument('--parallel', action='store_true', 
                       help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip existing processed files')
    parser.add_argument('--no-skip', action='store_true',
                       help='Force reprocess existing files')
    
    args = parser.parse_args()
    
    # Handle skip existing logic
    if args.no_skip:
        args.skip_existing = False
    
    # Create start and end dates
    start_date = datetime.datetime(args.start_year, args.start_month, args.start_day, args.start_hour)
    end_date = datetime.datetime(args.end_year, args.end_month, args.end_day, args.end_hour)
    
    print(f"SDO Image Processor")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Wavelengths: {args.waves}")
    print(f"Time step: {args.time_step} hours")
    print(f"Sequence length: {args.num_sequence} images (before reference time, including reference)")
    print(f"Processing interval: {args.interval} hours")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    
    # Execute processing
    try:
        if args.parallel:
            success = process_parallel(start_date, end_date, args)
        else:
            success = process_sequential(start_date, end_date, args)
        
        if success:
            print("Processing completed successfully!")
        else:
            print("Processing failed!")
            return 1
            
    except KeyboardInterrupt:
        print("Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"Processing failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# # 파일 위치: processors/process_sdo.py

# import os
# import datetime
# from glob import glob
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import cpu_count
# import time
# import argparse

# import cv2
# import numpy as np

# def resize_image_optimized(img_array, target_size=(64, 64)):
#     """이미지 리사이징 최적화 - OpenCV 사용"""
#     if img_array is None:
#         return None
    
#     # OpenCV resize가 bin_ndarray보다 빠름 (네이티브 최적화)
#     # INTER_AREA는 다운샘플링에 최적
#     img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
#     return np.clip(img_resized, 0, 255).astype(np.uint8)

# def run_optimized(file_path, save_path, skip_existing=True):
#     """최적화된 개별 이미지 파일 처리"""
#     if not os.path.exists(file_path):
#         return False
    
#     # 기존 파일 스킵 옵션
#     if skip_existing and os.path.exists(save_path):
#         # 파일 크기가 0이 아니면 정상 처리된 것으로 간주
#         if os.path.getsize(save_path) > 0:
#             return True
    
#     try:
#         # 이미지 읽기
#         img_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
#         if img_array is None:
#             return False
        
#         # 최적화된 리사이징
#         img_resized = resize_image_optimized(img_array)
#         if img_resized is None:
#             return False
        
#         # 압축 저장
#         success = cv2.imwrite(save_path, img_resized, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000])
#         return success
        
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return False

# def main_optimized(date_target, args):
#     """단순하게 최적화된 날짜별 처리 함수"""
#     try:
#         # 1. 출력 디렉토리 설정 (생성하지 않고 경로만 설정)
#         date_start = date_target
#         date_end = date_start + datetime.timedelta(hours=args.time_step * (args.num_sequence - 1))
#         # date_dir = f"{args.output_dir}/{date_start:%Y%m%d%H}-{date_end:%Y%m%d%H}"
#         tmp = date_end + datetime.timedelta(hours=3)
#         date_dir = f"{args.output_dir}/{tmp:%Y%m%d%H}"
        
#         # 기존 완료 체크 (디렉토리가 존재할 경우에만)
#         if args.skip_existing and os.path.exists(date_dir):
#             expected_files = args.num_sequence * len(args.waves)
#             existing_files = 0
            
#             # 정확한 파일 수 체크: 예상 파일 수와 실제 파일 수가 정확히 일치해야 함
#             for wave in args.waves:
#                 save_dir = f"{date_dir}/{wave:d}"
#                 if os.path.exists(save_dir):
#                     valid_files = [f for f in os.listdir(save_dir) 
#                                  if f.endswith('.jp2') and os.path.getsize(f"{save_dir}/{f}") > 0]
#                     existing_files += len(valid_files)
            
#             # 정확히 일치해야 완료된 것으로 간주
#             if existing_files == expected_files:
#                 print(f"{date_target} : Already processed ({existing_files}/{expected_files} files). Skipping.")
#                 return True
#             elif existing_files > 0:
#                 print(f"{date_target} : Incomplete processing found ({existing_files}/{expected_files} files). Will reprocess.")
#                 # 불완전한 디렉토리 삭제
#                 import shutil
#                 shutil.rmtree(date_dir)
#                 print(f"{date_target} : Removed incomplete directory")

#         # 2. 파일 수집 및 유효성 검사 (디렉토리 생성 전에 먼저 확인)
#         filelists = [[] for _ in args.waves]
#         date_current = date_target
        
#         for i in range(args.num_sequence):
#             year = date_current.year
#             month = date_current.month
#             day = date_current.day
#             hour = date_current.hour
            
#             for wave_idx, wave in enumerate(args.waves):
#                 data_path = f"{args.input_dir}/{wave:d}/{year:04d}/{year:04d}{month:02d}{day:02d}"
#                 data_name = f"{year:04d}_{month:02d}_{day:02d}__{hour:02d}_00_*_*__SDO_AIA_AIA_{wave:d}.jp2"
#                 data_list = sorted(glob(f"{data_path}/{data_name}"))
                
#                 found = False
#                 for file_path in data_list:
#                     file_name = file_path.split("/")[-1]
#                     try:
#                         file_date = datetime.datetime.strptime(file_name[:23], "%Y_%m_%d__%H_%M_%S_%f")
#                         if file_date.second < 60:
#                             filelists[wave_idx].append(file_path)
#                             found = True
#                             break
#                     except ValueError:
#                         continue
                
#                 if not found:
#                     filelists[wave_idx].append("")
                    
#             date_current += datetime.timedelta(hours=args.time_step)

#         # 3. 유효성 검사 (디렉토리 생성 전에 확인)
#         for wave_idx, file_list in enumerate(filelists):
#             if len(file_list) != args.num_sequence:
#                 print(f"{date_target} : Invalid file list length for wave {args.waves[wave_idx]}. Skipping.")
#                 return False
#             for i, file_path in enumerate(file_list):
#                 if not file_path or not os.path.exists(file_path):
#                     print(f"{date_target} : Missing file for wave {args.waves[wave_idx]}, sequence {i}. Skipping.")
#                     return False

#         # 4. 여기서부터 디렉토리 생성 (모든 파일이 존재함을 확인한 후)
#         print(f"{date_target} : All files found. Creating directories and processing...")
        
#         for wave in args.waves:
#             save_dir = f"{date_dir}/{wave:d}"
#             os.makedirs(save_dir, exist_ok=True)

#         # 5. 파일 처리
#         total_files = args.num_sequence * len(args.waves)
#         processed = 0
        
#         for wave_idx, (wave, file_list) in enumerate(zip(args.waves, filelists)):
#             save_dir = f"{date_dir}/{wave:d}"
            
#             for file_path in file_list:
#                 file_name = os.path.basename(file_path)
#                 save_path = f"{save_dir}/{file_name}"
                
#                 if run_optimized(file_path, save_path, args.skip_existing):
#                     processed += 1
                
#                 # 간헐적 진행률 출력
#                 if processed % 10 == 0:
#                     print(f"{date_target} : {processed/total_files:.1%}", end="\r", flush=True)
        
#         success_rate = processed / total_files
#         print(f"{date_target} : Done ({processed}/{total_files} files, {success_rate:.1%})")
#         return processed == total_files  # 모든 파일이 성공적으로 처리되어야 성공
        
#     except Exception as e:
#         print(f"Error processing {date_target}: {e}")
#         # 처리 실패 시 빈 디렉토리 정리
#         try:
#             if 'date_dir' in locals() and os.path.exists(date_dir):
#                 # 디렉토리가 비어있거나 거의 비어있으면 삭제
#                 total_files = 0
#                 for wave in args.waves:
#                     save_dir = f"{date_dir}/{wave:d}"
#                     if os.path.exists(save_dir):
#                         total_files += len([f for f in os.listdir(save_dir) if f.endswith('.jp2')])
                
#                 if total_files < args.num_sequence * len(args.waves) * 0.1:  # 10% 미만이면 정리
#                     import shutil
#                     shutil.rmtree(date_dir)
#                     print(f"{date_target} : Cleaned up incomplete directory")
#         except:
#             pass  # 정리 실패는 무시
        
#         return False

# def process_sequential(start_date, end_date, args):
#     """순차 처리"""
#     if start_date.hour % args.interval != 0:
#         start_date += datetime.timedelta(hours=args.interval - start_date.hour % args.interval)

#     # 처리할 모든 날짜 리스트 생성
#     dates_to_process = []
#     current_date = start_date
#     while current_date <= end_date:
#         dates_to_process.append(current_date)
#         current_date += datetime.timedelta(hours=args.interval)
    
#     print(f"Total dates to process: {len(dates_to_process)}")
#     print(f"Date range: {start_date} to {end_date}")
#     print(f"Skip existing files: {args.skip_existing}")
#     print("Processing mode: Sequential")
    
#     # 순차 처리 실행
#     start_time = time.time()
#     successful = 0
#     failed = 0
    
#     try:
#         for i, date in enumerate(dates_to_process):
#             try:
#                 result = main_optimized(date, args)
#                 if result:
#                     successful += 1
#                 else:
#                     failed += 1
#             except Exception as e:
#                 print(f"Exception for date {date}: {e}")
#                 failed += 1
            
#             # 진행률 출력
#             total_processed = successful + failed
#             if total_processed % 5 == 0 or total_processed == len(dates_to_process):
#                 elapsed = time.time() - start_time
#                 rate = total_processed / elapsed if elapsed > 0 else 0
#                 remaining = len(dates_to_process) - total_processed
#                 eta = remaining / rate if rate > 0 else 0
                
#                 print(f"\nProgress: {total_processed}/{len(dates_to_process)} "
#                       f"({total_processed/len(dates_to_process):.1%}) - "
#                       f"Success: {successful}, Failed: {failed} - "
#                       f"Rate: {rate:.2f} dates/sec - ETA: {eta/60:.1f}min")
    
#     except KeyboardInterrupt:
#         print("\nInterrupted by user")
#         return False
    
#     # 최종 결과
#     end_time = time.time()
#     total_time = end_time - start_time
    
#     print(f"\n{'='*50}")
#     print(f"Processing completed!")
#     print(f"Total time: {total_time/60:.1f} minutes")
#     print(f"Successfully processed: {successful}/{len(dates_to_process)} dates")
#     print(f"Failed: {failed}/{len(dates_to_process)} dates")
#     print(f"Average rate: {len(dates_to_process)/total_time:.2f} dates/sec")
#     print(f"{'='*50}")
    
#     return True

# def process_parallel(start_date, end_date, args):
#     """병렬 처리"""
#     if start_date.hour % args.interval != 0:
#         start_date += datetime.timedelta(hours=args.interval - start_date.hour % args.interval)

#     # 처리할 모든 날짜 리스트 생성
#     dates_to_process = []
#     current_date = start_date
#     while current_date <= end_date:
#         dates_to_process.append(current_date)
#         current_date += datetime.timedelta(hours=args.interval)
    
#     print(f"Total dates to process: {len(dates_to_process)}")
#     print(f"Date range: {start_date} to {end_date}")
#     print(f"Skip existing files: {args.skip_existing}")
    
#     # 워커 수 최적화: CPU 집약적 작업이므로 CPU 코어 수와 동일하게
#     max_workers = min(args.max_workers or cpu_count(), len(dates_to_process))
#     print(f"Processing mode: Parallel ({max_workers} workers)")
    
#     # 병렬 처리 실행
#     start_time = time.time()
#     successful = 0
#     failed = 0
    
#     try:
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             # 모든 작업 제출
#             future_to_date = {executor.submit(main_optimized, date, args): date for date in dates_to_process}
            
#             # 완료된 작업들을 처리
#             for future in as_completed(future_to_date):
#                 date = future_to_date[future]
#                 try:
#                     result = future.result()
#                     if result:
#                         successful += 1
#                     else:
#                         failed += 1
#                 except Exception as e:
#                     print(f"Exception for date {date}: {e}")
#                     failed += 1
                
#                 # 진행률 출력
#                 total_processed = successful + failed
#                 if total_processed % 5 == 0 or total_processed == len(dates_to_process):
#                     elapsed = time.time() - start_time
#                     rate = total_processed / elapsed if elapsed > 0 else 0
#                     remaining = len(dates_to_process) - total_processed
#                     eta = remaining / rate if rate > 0 else 0
                    
#                     print(f"\nProgress: {total_processed}/{len(dates_to_process)} "
#                           f"({total_processed/len(dates_to_process):.1%}) - "
#                           f"Success: {successful}, Failed: {failed} - "
#                           f"Rate: {rate:.2f} dates/sec - ETA: {eta/60:.1f}min")
    
#     except KeyboardInterrupt:
#         print("\nInterrupted by user")
#         return False
#     except Exception as e:
#         print(f"Error in parallel processing: {e}")
#         return False
    
#     # 최종 결과
#     end_time = time.time()
#     total_time = end_time - start_time
    
#     print(f"\n{'='*50}")
#     print(f"Processing completed!")
#     print(f"Total time: {total_time/60:.1f} minutes")
#     print(f"Successfully processed: {successful}/{len(dates_to_process)} dates")
#     print(f"Failed: {failed}/{len(dates_to_process)} dates")
#     print(f"Average rate: {len(dates_to_process)/total_time:.2f} dates/sec")
#     print(f"{'='*50}")
    
#     return True

# def main():
#     parser = argparse.ArgumentParser(description='SDO Image Processor')
    
#     # Input/Output directories
#     parser.add_argument('--input-dir', type=str, 
#                        default='/Volumes/usbshare1/data/sdo_jp2/aia',
#                        help='Input directory path')
#     parser.add_argument('--output-dir', type=str,
#                        default='/Users/eunsupark/ap_project/data/new_processed',
#                        help='Output directory path')
    
#     # Date range
#     parser.add_argument('--start-year', type=int, default=2010, help='Start year')
#     parser.add_argument('--start-month', type=int, default=9, help='Start month')
#     parser.add_argument('--start-day', type=int, default=1, help='Start day')
#     parser.add_argument('--start-hour', type=int, default=0, help='Start hour')
#     parser.add_argument('--end-year', type=int, default=2024, help='End year')
#     parser.add_argument('--end-month', type=int, default=12, help='End month')
#     parser.add_argument('--end-day', type=int, default=31, help='End day')
#     parser.add_argument('--end-hour', type=int, default=23, help='End hour')
    
#     # Processing parameters
#     parser.add_argument('--waves', type=int, nargs='+', default=[193, 211],
#                        help='Wavelengths to process')
#     parser.add_argument('--time-step', type=int, default=6,
#                        help='Time step in hours for sequence')
#     parser.add_argument('--num-sequence', type=int, default=40,
#                        help='Number of images in sequence')
#     parser.add_argument('--interval', type=int, default=3,
#                        help='Processing interval in hours')
    
#     # Processing options
#     parser.add_argument('--parallel', action='store_true', 
#                        help='Enable parallel processing')
#     parser.add_argument('--max-workers', type=int, default=None,
#                        help='Maximum number of parallel workers (default: auto)')
#     parser.add_argument('--skip-existing', action='store_true', default=True,
#                        help='Skip existing processed files')
#     parser.add_argument('--no-skip', action='store_true',
#                        help='Force reprocess existing files')
    
#     args = parser.parse_args()
    
#     # Handle skip existing logic
#     if args.no_skip:
#         args.skip_existing = False
    
#     # Create start and end dates
#     start_date = datetime.datetime(args.start_year, args.start_month, args.start_day, args.start_hour)
#     end_date = datetime.datetime(args.end_year, args.end_month, args.end_day, args.end_hour)
    
#     print(f"SDO Image Processor")
#     print(f"Input directory: {args.input_dir}")
#     print(f"Output directory: {args.output_dir}")
#     print(f"Wavelengths: {args.waves}")
#     print(f"Time step: {args.time_step} hours")
#     print(f"Sequence length: {args.num_sequence}")
#     print(f"Processing interval: {args.interval} hours")
#     print(f"Date range: {start_date} to {end_date}")
#     print(f"Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    
#     # Execute processing
#     try:
#         if args.parallel:
#             success = process_parallel(start_date, end_date, args)
#         else:
#             success = process_sequential(start_date, end_date, args)
        
#         if success:
#             print("Processing completed successfully!")
#         else:
#             print("Processing failed!")
#             return 1
            
#     except KeyboardInterrupt:
#         print("Processing interrupted by user")
#         return 1
#     except Exception as e:
#         print(f"Processing failed with error: {e}")
#         return 1
    
#     return 0

# if __name__ == "__main__":
#     exit(main())