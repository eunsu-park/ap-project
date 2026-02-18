"""Config 로드 유틸리티"""
import yaml
import os
import re
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """YAML 설정 파일 로드 (환경변수 치환 포함)"""
    with open(path) as f:
        config = yaml.safe_load(f)
    
    def substitute(obj):
        if isinstance(obj, str):
            for var in re.findall(r'\$\{(\w+)\}', obj):
                obj = obj.replace(f'${{{var}}}', os.environ.get(var, ''))
            return obj
        elif isinstance(obj, dict):
            return {k: substitute(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute(i) for i in obj]
        return obj
    
    return substitute(config)