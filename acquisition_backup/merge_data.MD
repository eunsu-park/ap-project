# merge_data_improved.py 사용 가이드

## 🎯 개선 사항 요약

### ✅ 주요 개선사항

| 항목 | Before | After |
|------|--------|-------|
| **DB 연결** | 전역 연결 (불안정) | Context Manager (안전) |
| **에러 처리** | `return False, None` | 상세 에러 메시지 + 로깅 |
| **병렬 처리** | ❌ 순차 처리 | ✅ 멀티프로세싱 |
| **중복 방지** | ❌ 없음 | ✅ 파일 체크 + Progress 파일 |
| **재시도** | ❌ 없음 | ✅ 3회 재시도 (설정 가능) |
| **진행률** | ❌ 없음 | ✅ tqdm 진행바 |
| **로깅** | `print()` | logging 모듈 (파일 + 콘솔) |
| **메모리** | 비효율적 | Pre-allocation으로 개선 |

---

## 🚀 빠른 시작

### 1. 기본 실행

```bash
cd /Users/eunsupark/projects/ap
python merge_data_improved.py
```

**출력 예시:**
```
============================================================
SDO + OMNI DATA MERGER
============================================================
DATASET CADENCE: 3 hours
SDO CADENCE: 6 hours
OMNI CADENCE: 3 hours
BEFORE 10 days to AFTER 5 days
SDO_NUM_SEQUENCE: 60
OMNI_NUM_SEQUENCE: 120
NUM_WORKERS: 7
============================================================
Total dates to process: 41,016
Already processed: 0
To process: 41,016

Starting parallel processing...
Processing dates: 45%|████████      | 18450/41016 [2:15:30<2:45:20, 2.27it/s]
```

### 2. 중단 후 재시작

프로그램이 중단되어도 **자동으로 이어서 처리**:

```bash
# Ctrl+C로 중단
^C
⚠ Interrupted by user

# 다시 실행 → 처리된 날짜는 자동 스킵
python merge_data_improved.py

Already processed: 18,450
To process: 22,566
```

---

## 📊 성능 비교

### 예상 처리 시간 (41,016개 샘플)

| 항목 | 기존 (순차) | 개선 (병렬 8 cores) | 속도 개선 |
|------|------------|-------------------|----------|
| **1개 처리 시간** | 2초 | 2초 | - |
| **전체 시간** | 22.8시간 | 2.9시간 | **7.8배 빠름** ⚡ |
| **재시작 후** | 처음부터 | 이어서 진행 | **무한대** ⚡ |

---

## 🔧 설정 변경

### config 섹션 수정

```python
# 시간 간격 변경
OMNI_CADENCE = 3  # OMNI 데이터 간격 (hours)
SDO_CADENCE = 6   # SDO 데이터 간격 (hours)
DATASET_CADENCE = 3  # 샘플 생성 간격 (hours)

# 시퀀스 길이 변경
BEFORE_DAY = 10  # 과거 데이터 (days)
AFTER_DAY = 5    # 미래 데이터 (days)

# 병렬 처리
NUM_WORKERS = 4  # 워커 수 (기본: cpu_count() - 1)

# 재시도
MAX_RETRIES = 3  # 최대 재시도 횟수
RETRY_DELAY = 5  # 재시도 간 대기 시간 (seconds)
```

### 날짜 범위 변경

```python
# main() 함수 내
start_date = datetime.datetime(2015, 1, 1, 0, 0, 0)  # 시작 날짜
end_date = datetime.datetime(2020, 1, 1, 0, 0, 0)    # 종료 날짜
```

---

## 📝 생성 파일

### 1. 데이터 파일

```
/Users/eunsupark/projects/ap/data/
├── 2011010100/
│   ├── aia_193.npy           # (60, H, W) - AIA 193Å 시퀀스
│   ├── aia_211.npy           # (60, H, W) - AIA 211Å 시퀀스
│   ├── hmi_magnetogram.npy   # (60, H, W) - HMI Magnetogram 시퀀스
│   └── omni.csv              # (120 rows) - OMNI 시계열
├── 2011010103/
│   ├── ...
└── ...
```

### 2. 로그 파일

**merge_data.log:**
```
2024-12-01 10:30:15 - INFO - [MainProcess] - Starting parallel processing...
2024-12-01 10:30:16 - INFO - [SpawnProcess-1] - Successfully processed: 2011-01-01 00:00:00
2024-12-01 10:30:17 - WARNING - [SpawnProcess-2] - Attempt 1/3 failed: No SDO data: aia_193 at 2011-02-15
2024-12-01 10:30:22 - INFO - [SpawnProcess-2] - Retrying in 5 seconds...
2024-12-01 10:30:27 - INFO - [SpawnProcess-2] - Successfully processed: 2011-02-15 00:00:00
```

### 3. Progress 파일

**merge_progress.json:**
```json
{
  "processed": [
    "2011-01-01T00:00:00",
    "2011-01-01T03:00:00",
    "2011-01-01T06:00:00",
    ...
  ],
  "total": 18450,
  "last_updated": "2024-12-01T12:30:45"
}
```

---

## 🐛 트러블슈팅

### 1. DB 연결 에러

**증상:**
```
ERROR - Connection failed: connection refused
```

**해결:**
```bash
# PostgreSQL 실행 확인
pg_isready

# 재시작
brew services restart postgresql@14
```

### 2. 메모리 부족

**증상:**
프로세스 종료 또는 매우 느려짐

**해결:**
```python
# merge_data_improved.py 수정
NUM_WORKERS = 2  # 워커 수 감소 (8 → 2)
```

### 3. 디스크 용량 부족

**증상:**
```
ERROR - Save error: [Errno 28] No space left on device
```

**확인:**
```bash
df -h /Users/eunsupark/projects/ap/data
```

**해결:**
- 불필요한 파일 삭제
- 다른 디스크로 SAVE_ROOT 변경

### 4. 특정 날짜 계속 실패

**증상:**
```
ERROR - Failed after 3 retries: 2011-05-15: No SDO data
```

**원인:**
- SDO 데이터 누락
- OMNI 데이터 누락
- 파일 손상

**확인:**
```python
# 특정 날짜 디버깅
target_date = datetime.datetime(2011, 5, 15, 0, 0, 0)
success, error = process_single_date(target_date)
print(f"Success: {success}")
print(f"Error: {error}")
```

### 5. Progress 파일 손상

**증상:**
```
WARNING - Failed to load progress: ...
```

**해결:**
```bash
# Progress 파일 삭제 후 재시작
rm merge_progress.json
python merge_data_improved.py
```

---

## 💡 고급 사용법

### 1. 특정 연도만 처리

```python
# main() 함수 수정
start_date = datetime.datetime(2020, 1, 1, 0, 0, 0)
end_date = datetime.datetime(2021, 1, 1, 0, 0, 0)
```

### 2. 실패한 날짜만 재처리

```python
# 로그에서 실패한 날짜 추출
failed_dates = [
    datetime.datetime(2011, 5, 15, 0, 0, 0),
    datetime.datetime(2012, 3, 20, 0, 0, 0),
]

# 병렬 처리
with Pool(processes=NUM_WORKERS) as pool:
    results = pool.map(process_one_date_wrapper, failed_dates)
```

### 3. 단일 날짜 테스트

```python
# 병렬 처리 주석 처리하고
# 단일 날짜로 테스트
test_date = datetime.datetime(2011, 1, 1, 0, 0, 0)
result = process_one_date_wrapper(test_date)
print(result)
```

### 4. 워커 수 동적 조정

```python
import psutil

# 사용 가능한 메모리에 따라 조정
available_memory_gb = psutil.virtual_memory().available / (1024**3)

if available_memory_gb < 8:
    NUM_WORKERS = 2
elif available_memory_gb < 16:
    NUM_WORKERS = 4
else:
    NUM_WORKERS = cpu_count() - 1
```

### 5. 로깅 레벨 변경

```python
# 더 자세한 로깅
logging.basicConfig(
    level=logging.DEBUG,  # INFO → DEBUG
    ...
)

# 덜 자세한 로깅
logging.basicConfig(
    level=logging.WARNING,  # INFO → WARNING
    ...
)
```

---

## 📊 모니터링

### 실시간 진행 상황

**터미널 1:**
```bash
python merge_data_improved.py
```

**터미널 2 (진행률 확인):**
```bash
# Progress 파일 모니터링
watch -n 5 'cat merge_progress.json | jq ".total"'

# 로그 실시간 확인
tail -f merge_data.log
```

**터미널 3 (리소스 사용량):**
```bash
# CPU/메모리 모니터링
top

# 또는
htop
```

---

## 🎯 체크리스트

### 실행 전
- [ ] PostgreSQL 실행 중
- [ ] SDO 데이터베이스 접근 가능
- [ ] OMNI 데이터베이스 접근 가능
- [ ] 충분한 디스크 용량 (최소 100GB)
- [ ] 충분한 메모리 (최소 8GB)

### 실행 중
- [ ] 진행률 표시 정상
- [ ] 로그 파일 생성 확인
- [ ] Progress 파일 업데이트 확인
- [ ] 에러 발생 시 재시도 확인

### 실행 후
- [ ] 성공률 확인 (> 95%)
- [ ] 실패한 날짜 재처리
- [ ] 데이터 샘플링 확인
- [ ] 파일 무결성 검증

---

## 🔄 기존 코드와 비교

### 호환성
- ✅ 동일한 출력 형식
- ✅ 동일한 디렉토리 구조
- ✅ 동일한 파일명
- ✅ 기존 데이터와 호환

### 전환 방법
```bash
# 기존 코드 백업
cp merge_data.py merge_data_old.py

# 새 코드로 교체
cp merge_data_improved.py merge_data.py

# 실행
python merge_data.py
```

---

## 📚 코드 구조

```python
# 설정
SAVE_ROOT = "..."
DB_CONFIGS = {...}
CADENCE = ...

# 로깅
setup_logging()

# 유틸리티
is_already_processed()
save_progress()
load_progress()
retry_on_failure()

# DB 접근
get_sdo_data()
get_omni_data()

# 이미지 처리
read_png()
read_and_concatenate_images()

# 메인 처리
collect_sdo_files()
collect_omni_data()
save_merged_data()
process_single_date()

# 병렬 처리
process_one_date_wrapper()
main()
```

---

## 🎓 학습 포인트

### 1. Context Manager 패턴
```python
with PostgresManager(**config) as db:
    data = db.select(...)
# 자동으로 연결 해제
```

### 2. 재시도 패턴
```python
def retry_on_failure(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except:
            if attempt < max_retries - 1:
                sleep(delay)
            else:
                raise
```

### 3. 병렬 처리 패턴
```python
with Pool(processes=num_workers) as pool:
    results = pool.map(process_func, items)
```

### 4. Progress 저장 패턴
```python
processed = load_progress()
to_process = [x for x in all if x not in processed]
# 처리...
save_progress(processed + newly_processed)
```

---

## 💬 FAQ

**Q: 기존 코드보다 느린 것 같아요**
A: 첫 실행은 progress 체크로 약간 느릴 수 있지만, 재시작 시 훨씬 빠릅니다.

**Q: 중간에 멈췄는데 다시 처음부터 하나요?**
A: 아니요! 자동으로 이어서 처리합니다.

**Q: 특정 날짜만 다시 처리하고 싶어요**
A: Progress 파일에서 해당 날짜만 제거하거나, failed_dates 리스트로 직접 실행하세요.

**Q: 로그 파일이 너무 커져요**
A: 로그 로테이션 설정하거나, 주기적으로 삭제하세요.

**Q: NUM_WORKERS를 얼마로 설정해야 하나요?**
A: `cpu_count() - 1`이 기본값이지만, 메모리가 부족하면 2-4로 줄이세요.

---

**완료!** 🎉

문제 있으면 알려주세요! 😊