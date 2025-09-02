"""
Generic Download Utilities - Python 3.6+
Reusable components for any data download task
"""

import os
import logging
import time
import yaml
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

@dataclass
class DownloadTask:
    """Represents a single download task"""
    source: str
    destination: str
    retry_count: int = 0
    success: bool = False
    error_message: str = ""

class DataSourceConfig(ABC):
    """Abstract base class for data source configurations"""
    
    @abstractmethod
    def get_url(self, **kwargs) -> str:
        pass
    
    @abstractmethod
    def get_save_path(self, destination_root: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def get_file_list(self, base_url: str, extensions: List[str]) -> List[str]:
        pass

class GenericWebSource(DataSourceConfig):
    """Generic web data source with pattern-based URLs"""
    
    def __init__(self, url_pattern: str, path_pattern: str):
        self.url_pattern = url_pattern
        self.path_pattern = path_pattern
    
    def get_url(self, **kwargs) -> str:
        return self.url_pattern.format(**kwargs)
    
    def get_save_path(self, destination_root: str, **kwargs) -> str:
        relative_path = self.path_pattern.format(**kwargs)
        return str(Path(destination_root) / relative_path)
    
    def get_file_list(self, base_url: str, extensions: List[str]) -> List[str]:
        """Get file list from web directory"""
        try:
            session = requests.Session()
            retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
            session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
            
            response = session.get(f"{base_url}/", verify=False, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            files = [
                link.get('href') for link in soup.find_all('a', href=True)
                if link.get('href') and 
                any(link.get('href').lower().endswith(f".{ext.lower()}") for ext in extensions) and
                not link.get('href').startswith('/') and '?' not in link.get('href')
            ]
            
            return [f for f in files if not any(skip in f.lower() for skip in ['parent', '..', 'index', 'readme'])]
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error fetching file list from {base_url}: {e}")
            return []

class SimpleDownloader:
    """Simple HTTP downloader with retry logic"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, overwrite: bool = False):
        self.timeout = timeout
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.logger = logging.getLogger(__name__)
        
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    
    def download(self, source: str, destination: str) -> bool:
        """Download file from source to destination"""
        if os.path.exists(destination) and not self.overwrite:
            return True
        
        # Create directory
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        
        temp_path = f"{destination}.tmp"
        try:
            response = self.session.get(source, stream=True, verify=False, timeout=self.timeout)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.getsize(temp_path) > 0:
                os.rename(temp_path, destination)
                return True
            else:
                os.remove(temp_path)
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading {source}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

class ConfigLoader:
    """YAML configuration file loader"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load config: {e}")
            raise

class FixedWidthParser:
    """Parser for fixed-width text files"""
    
    def __init__(self, field_specs: List[Dict[str, Any]]):
        """
        Initialize parser with field specifications
        
        Args:
            field_specs: List of field specifications with format info
        """
        self.field_specs = field_specs
        self.field_widths = self._calculate_field_widths()
        self.fill_values = {spec['name']: spec.get('fill_value') 
                           for spec in field_specs 
                           if spec.get('fill_value') is not None}
    
    def _calculate_field_widths(self) -> List[int]:
        """Calculate field widths from format strings"""
        widths = []
        for spec in self.field_specs:
            fmt = spec.get('format', 'F8.2')
            if fmt.startswith('I'):
                width = int(fmt[1:])
            elif fmt.startswith('F'):
                if '.' in fmt:
                    width = int(fmt[1:fmt.index('.')])
                else:
                    width = int(fmt[1:])
            else:
                width = 8  # default
            widths.append(width)
        return widths
    
    def _clean_value(self, value: str, field_spec: Dict[str, Any]) -> Any:
        """Clean and convert field value"""
        if not value or not value.strip():
            return None
        
        value = value.strip()
        name = field_spec['name']
        fmt = field_spec.get('format', 'F8.2')
        fill_value = field_spec.get('fill_value')
        
        try:
            if fmt.startswith('I'):  # Integer
                int_val = int(float(value))
                if fill_value is not None and int_val == fill_value:
                    return None
                return int_val
            elif fmt.startswith('F'):  # Float
                float_val = float(value)
                if fill_value is not None and abs(float_val - fill_value) < 1e-3:
                    return None
                return float_val
        except (ValueError, TypeError):
            return None
        
        return value
    
    def parse_line(self, line: str) -> Dict[str, Any]:
        """Parse a single line of fixed-width data"""
        result = {}
        pos = 0
        
        for i, spec in enumerate(self.field_specs):
            width = self.field_widths[i]
            
            if pos + width <= len(line):
                field_value = line[pos:pos + width]
                result[spec['name']] = self._clean_value(field_value, spec)
            else:
                result[spec['name']] = None
            
            pos += width
        
        return result

class FilePatternSource(DataSourceConfig):
    """Data source with file pattern URLs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['base_url']
        self.file_pattern = config['file_pattern']
        self.output_pattern = config.get('output_pattern', self.file_pattern.replace('.dat', '.csv'))
    
    def get_url(self, **kwargs) -> str:
        filename = self.file_pattern.format(**kwargs)
        return f"{self.base_url}{filename}"
    
    def get_save_path(self, destination_root: str, **kwargs) -> str:
        filename = self.output_pattern.format(**kwargs)
        return str(Path(destination_root) / filename)
    
    def get_file_list(self, base_url: str, extensions: List[str]) -> List[str]:
        """For pattern-based sources, return single file"""
        filename = Path(base_url).name
        return [filename] if any(filename.endswith(ext) for ext in extensions) else []
        
class GenericDataDownloader:
    """Generic data downloader with concurrent processing"""
    
    def __init__(self, config: DataSourceConfig, **kwargs):
        self.config = config
        self.downloader = SimpleDownloader(
            timeout=kwargs.get('timeout', 30),
            max_retries=kwargs.get('max_retries', 3),
            overwrite=kwargs.get('overwrite', False)
        )
        self.logger = logging.getLogger(__name__)
        self.parallel = kwargs.get('parallel', 1)
        self.use_threads = kwargs.get('use_threads', True)
    
    def get_download_tasks(self, **params) -> List[DownloadTask]:
        """Generate download tasks based on parameters"""
        tasks = []
        try:
            # Extract destination_root separately to avoid keyword argument conflicts
            destination_root = params.pop('destination_root', './data')
            
            base_url = self.config.get_url(**params)
            save_dir = self.config.get_save_path(destination_root, **params)
            extensions = params.get('extensions', ['jp2'])
            
            file_list = self.config.get_file_list(base_url, extensions)
            
            for file_name in file_list:
                source = f"{base_url}/{file_name}"
                destination = str(Path(save_dir) / file_name)
                tasks.append(DownloadTask(source, destination))
                
        except Exception as e:
            self.logger.error(f"Error generating tasks: {e}")
        
        return tasks
    
    def _download_task(self, task: DownloadTask) -> DownloadTask:
        """Download a single task"""
        task.success = self.downloader.download(task.source, task.destination)
        return task
    
    def download(self, tasks: List[DownloadTask]) -> List[DownloadTask]:
        """Download tasks with optional parallelism"""
        if not tasks:
            return tasks
        
        if self.parallel < 2:
            return [self._download_task(task) for task in tasks]
        
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        with executor_class(max_workers=self.parallel) as executor:
            return list(executor.map(self._download_task, tasks))
    
    def run_with_retry(self, max_retries: int = 3, **params):
        """Run download with retry logic"""
        start_time = time.time()
        
        tasks = self.get_download_tasks(**params)
        if not tasks:
            return {"success": True, "downloaded": 0, "failed": 0, "duration": 0}
        
        for attempt in range(max_retries + 1):
            self.logger.info(f"Attempt {attempt + 1}: Processing {len(tasks)} files")
            
            completed_tasks = self.download(tasks)
            successful = [t for t in completed_tasks if t.success]
            failed = [t for t in completed_tasks if not t.success]
            
            self.logger.info(f"Success: {len(successful)}, Failed: {len(failed)}")
            
            if not failed or attempt == max_retries:
                break
            
            tasks = failed
            time.sleep(2)
        
        duration = time.time() - start_time
        return {
            "success": len(failed) == 0,
            "downloaded": len(successful),
            "failed": len(failed),
            "duration": duration
        }

class TimeSeriesProcessor:
    """Time series data processor for matching datasets"""
    
    def __init__(self, data_dir: str, columns: Optional[List[str]] = None):
        self.data_dir = data_dir
        self.columns = columns  # None means all columns
        self.logger = logging.getLogger(__name__)
        self._data_cache = {}
    
    def _load_year_data(self, year: int) -> Optional[Any]:
        """Load OMNI data for specific year"""
        if year in self._data_cache:
            return self._data_cache[year]
        
        import pandas as pd
        csv_path = Path(self.data_dir) / f"omni2_{year}.csv"
        
        if not csv_path.exists():
            self.logger.warning(f"Data file not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            self._data_cache[year] = df
            return df
        except Exception as e:
            self.logger.error(f"Error loading {csv_path}: {e}")
            return None
    
    def find_matching_data(self, target_datetime, time_offset_hours: int = -1) -> Optional[Dict[str, Any]]:
        """Find data matching target datetime with offset"""
        import pandas as pd
        from datetime import timedelta
        
        # Apply time offset
        search_datetime = target_datetime + timedelta(hours=time_offset_hours)
        year = search_datetime.year
        
        # Load year data
        df = self._load_year_data(year)
        if df is None:
            return None
        
        # Create datetime from Year, Day, Hour columns
        try:
            df['search_datetime'] = pd.to_datetime(df['Year'], format='%Y') + \
                                  pd.to_timedelta(df['Day'] - 1, unit='D') + \
                                  pd.to_timedelta(df['Hour'], unit='h')
        except KeyError as e:
            self.logger.error(f"Required time columns missing: {e}")
            return None
        
        # Find exact match
        matching_rows = df[df['search_datetime'] == search_datetime]
        
        if len(matching_rows) == 0:
            return None
        
        # Extract columns
        result = {}
        row = matching_rows.iloc[0]
        
        # If no specific columns specified, use all columns (except time-related ones)
        if self.columns is None:
            exclude_cols = {'search_datetime', 'Year', 'Day', 'Hour'}
            data_columns = [col for col in df.columns if col not in exclude_cols]
        else:
            data_columns = self.columns
        
        for col in data_columns:
            if col in df.columns:
                value = row[col]
                # Convert to numeric, NaN if invalid
                result[col] = pd.to_numeric(value, errors='coerce')
            else:
                result[col] = None
        
        return result
    
    def process_time_range(self, start_datetime, end_datetime, interval_hours: int = 3, 
                         time_offset_hours: int = -1) -> List[Dict[str, Any]]:
        """Process time range and extract matching data"""
        from datetime import timedelta
        
        results = []
        current_time = start_datetime
        
        # Get column names from first successful data extraction
        data_columns = None
        
        while current_time <= end_datetime:
            data = self.find_matching_data(current_time, time_offset_hours)
            
            result_row = {
                'base_time': current_time.strftime('%Y-%m-%d %H')#,
                #'target_time': (current_time + timedelta(hours=time_offset_hours)).strftime('%Y-%m-%d %H')
            }
            
            if data:
                result_row.update(data)
                # Store column names from first successful extraction
                if data_columns is None:
                    data_columns = list(data.keys())
            else:
                # Fill with None for missing data (use previously found columns or empty)
                if data_columns is not None:
                    for col in data_columns:
                        result_row[col] = None
            
            results.append(result_row)
            current_time += timedelta(hours=interval_hours)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> bool:
        """Save results to CSV file"""
        if not results:
            return False
        
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return False

class DirectoryTimeExtractor:
    """Extract datetime from directory names"""
    
    @staticmethod
    def extract_from_pattern(dirname: str, pattern: str = '%Y%m%d%H') -> Optional[Any]:
        """Extract datetime from directory name using pattern"""
        from datetime import datetime
        
        try:
            # Extract time part from directory name (before '-' if exists)
            time_str = dirname.split('-')[0]
            return datetime.strptime(time_str, pattern)
        except (ValueError, IndexError):
            return None

class TextFileDownloader(GenericDataDownloader):
    """Downloader for text files with parsing capability"""
    
    def __init__(self, config: DataSourceConfig, parser: FixedWidthParser = None, **kwargs):
        super().__init__(config, **kwargs)
        self.parser = parser
    
    def download_and_convert(self, output_path: str, **params) -> bool:
        """Download file and convert to CSV if parser is available"""
        # Get source file info
        url = self.config.get_url(**params)
        temp_file = Path(output_path).with_suffix('.tmp')
        
        # Download file
        success = self.downloader.download(url, str(temp_file))
        if not success:
            return False
        
        # Convert to CSV if parser available
        if self.parser:
            try:
                import pandas as pd
                
                data_rows = []
                with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parsed = self.parser.parse_line(line)
                            data_rows.append(parsed)
                
                if data_rows:
                    df = pd.DataFrame(data_rows)
                    df.to_csv(output_path, index=False)
                    temp_file.unlink()  # Remove temp file
                    self.logger.info(f"Converted to CSV: {output_path} ({len(df)} rows)")
                    return True
                
            except Exception as e:
                self.logger.error(f"Failed to convert to CSV: {e}")
        
        # If no parser or conversion failed, just move the file
        Path(temp_file).rename(output_path)
        return True