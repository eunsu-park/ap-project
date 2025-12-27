#!/bin/bash

# .sh 파일 삭제
pattern="AUTO-*.sh"
files=(./$pattern)
if [ -e "${files[0]}" ]; then
    count=${#files[@]}
    echo "Removing $count files matching pattern: $pattern"
    rm -f ./$pattern
else
    echo "No files matching pattern: $pattern"
fi

# .yaml 파일 삭제
pattern_yaml="AUTO-*.yaml"
yaml_files=(./configs/$pattern_yaml)
if [ -e "${yaml_files[0]}" ]; then
    count_yaml=${#yaml_files[@]}
    echo "Removing $count_yaml files matching pattern: $pattern_yaml"
    rm -f ./configs/$pattern_yaml
else
    echo "No files matching pattern: $pattern_yaml"
fi

# __pycache__ 디렉토리 삭제
echo "Removing __pycache__ directories"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "Cleanup completed."