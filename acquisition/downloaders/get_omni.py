"""
OMNI Data Processor - Python 3.6+
Downloads and converts NASA OMNI data files to CSV
파일 위치: downloaders/get_omni.py
"""

import sys
from pathlib import Path

# Check Python version
if sys.version_info < (3, 6):
    raise RuntimeError("This script requires Python 3.6 or higher")

from core.utils import (
    ConfigLoader, FixedWidthParser, FilePatternSource, 
    TextFileDownloader, setup_logging
)

class OMNIProcessor:
    """OMNI 데이터 처리기"""
    
    def __init__(self, config_path: str):
        self.config = ConfigLoader.load_config(config_path)
        self.logger = None
    
    def setup_dataset(self, dataset_name: str):
        """데이터셋 설정"""
        if dataset_name not in self.config['datasets']:
            available = list(self.config['datasets'].keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
        
        dataset_config = self.config['datasets'][dataset_name]
        
        # 필드 명세 생성 (50개 컬럼 전체)
        field_specs = []
        for col_def in dataset_config['columns']:
            spec = {
                'name': col_def[0],
                'format': col_def[1],
                'fill_value': col_def[2],
                'description': col_def[3],
                'unit': col_def[4]
            }
            field_specs.append(spec)
        
        # 파서 및 다운로더 생성
        parser = FixedWidthParser(field_specs)
        source = FilePatternSource(dataset_config)
        
        return TextFileDownloader(
            source,
            parser=parser,
            timeout=60,
            max_retries=3,
            overwrite=False
        ), dataset_config
    
    def process_years(self, dataset_name: str, years: list, output_dir: str, 
                     overwrite: bool = False) -> None:
        """연도별 데이터 처리"""
        downloader, dataset_config = self.setup_dataset(dataset_name)
        downloader.overwrite = overwrite
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        failed = 0
        
        for year in years:
            print(f"Processing year {year}")
            output_file = str(output_path / dataset_config['output_pattern'].format(year=year))
            
            if downloader.download_and_convert(output_file, year=year):
                print(f"Completed: {output_file}")
                successful += 1
            else:
                print(f"Failed: {year}")
                failed += 1
        
        print(f"Processing complete: {successful} successful, {failed} failed")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NASA OMNI Data Processor")
    
    # 기본 옵션
    parser.add_argument("--config", type=str, default="omni_config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset name (omni_low_res, omni_high_res)")
    parser.add_argument("--output-dir", type=str, default="./omni_data", 
                       help="Output directory")
    
    # 연도 옵션
    parser.add_argument("--year", type=int, help="Single year to process")
    parser.add_argument("--start-year", type=int, help="Start year")
    parser.add_argument("--end-year", type=int, help="End year")
    
    # 처리 옵션
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, help="Log file path")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level, args.log_file)
    
    try:
        # OMNI 처리기 생성
        processor = OMNIProcessor(args.config)
        
        # 연도 범위 결정
        if args.year:
            years = [args.year]
        else:
            config = processor.config
            start_year = args.start_year or config['default_settings']['default_years']['start_year']
            end_year = args.end_year or config['default_settings']['default_years']['end_year']
            years = list(range(start_year, end_year + 1))
        
        print(f"OMNI Data Processor")
        print(f"Dataset: {args.dataset}")
        print(f"Years: {years[0]} to {years[-1]} ({len(years)} years)")
        print(f"Output directory: {args.output_dir}")
        
        # 데이터 처리
        processor.process_years(
            args.dataset, 
            years, 
            args.output_dir,
            args.overwrite
        )
        
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()