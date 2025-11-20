#!/bin/bash

# 로그 파일 설정 (선택사항)
LOG_FILE="execution.log"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting download_sdo_fits.py" | tee -a $LOG_FILE
    
    # download_sdo_fits.py 실행 (완료될 때까지 대기)
    /Users/eunsupark/Softwares/miniconda3/envs/ap-backend/bin/python download_sdo_fits.py
    A_EXIT_CODE=$?
    
    # download_sdo_fits.py가 성공적으로 완료되었는지 확인
    if [ $A_EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] download_sdo_fits.py completed successfully. Starting update_database.py" | tee -a $LOG_FILE
        
        # update_database.py 실행
        /Users/eunsupark/Softwares/miniconda3/envs/ap-backend/bin/python update_database.py
        B_EXIT_CODE=$?
        
        if [ $B_EXIT_CODE -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] update_database.py completed successfully" | tee -a $LOG_FILE
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] update_database.py failed with exit code $B_EXIT_CODE" | tee -a $LOG_FILE
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] download_sdo_fits.py failed with exit code $A_EXIT_CODE. Skipping update_database.py" | tee -a $LOG_FILE
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 5 minutes..." | tee -a $LOG_FILE
    # 5분(300초) 대기
    sleep 300
done