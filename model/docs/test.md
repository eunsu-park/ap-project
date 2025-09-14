# test.py 사용법 가이드

## 개요
`test.py`는 훈련된 멀티모달 딥러닝 모델로 실제 테스트 데이터에 대한 추론을 수행하는 스크립트입니다. Ground truth가 있는 경우 성능 평가를, 없는 경우 순수 예측을 수행합니다.

## 기본 사용법

### 1. 기본 테스트 실행 (validation 데이터 사용)
```bash
python test.py --checkpoint ./results/my_experiment/checkpoint/best_model.pth
```

### 2. 별도 테스트 데이터로 실행
```bash
python test.py --checkpoint best_model.pth --test_list ./data/test_list.csv
```

### 3. 결과 저장 디렉토리 지정
```bash
python test.py --checkpoint model.pth --output_dir ./test_results --test_list test_data.csv
```

## 커맨드라인 옵션

| 옵션 | 타입 | 기본값 | 필수 | 설명 |
|------|------|--------|------|------|
| `--config` | str | 'config_dev.yaml' | 아니오 | 설정 파일 경로 |
| `--checkpoint` | str | None | **예** | 모델 체크포인트 경로 |
| `--test_list` | str | None | 아니오 | 테스트 데이터 목록 파일 경로 |
| `--output_dir` | str | None | 아니오 | 결과 저장 디렉토리 |
| `--seed` | int | None | 아니오 | 랜덤 시드 (config 오버라이드) |

## 설정 파일 요구사항

### 필수 설정
```yaml
# 데이터 경로
data_root: './data/dataset'
validation_list_path: './data/validation_list.csv'  # test_list 미지정시 사용
stat_file_path: './data/statistics.pkl'

# 모델 아키텍처 (훈련 시와 동일해야 함)
input_variables: ['Bx_GSE', 'By_GSM', 'Bz_GSM', ...]
target_variables: ['ap_index', 'DST_index']
input_sequence_length: 40
target_sequence_length: 24
num_linear_output: 256
inception_in_channels: 2
inception_out_channels: 96
inception_in_image_size: 64
inception_in_image_frames: 20
lstm_hidden_size: 512

# 기타 설정
device: 'cuda'
seed: 42
```

## 테스트 데이터 형식

### 1. CSV 파일 형식 (test_list.csv)
```csv
file_name
sample_001.h5
sample_002.h5
sample_003.h5
...
```

### 2. HDF5 데이터 파일 구조
각 HDF5 파일은 다음 데이터셋을 포함해야 합니다:
```
sample_001.h5
├── sdo_193          # SDO 193 채널 이미지 (frames, height, width)
├── sdo_211          # SDO 211 채널 이미지 (frames, height, width)
├── omni_Bx_GSE      # 태양풍 입력 변수들
├── omni_By_GSM
├── ...
├── omni_ap_index    # 타겟 변수들 (ground truth 있는 경우)
└── omni_DST_index
```

## 실행 예시

### 1. Validation 데이터로 테스트 (Ground Truth 있음)
```bash
python test.py --checkpoint ./results/experiment_1/checkpoint/best_model.pth
```

### 2. 새로운 테스트 데이터로 예측 (Ground Truth 없음)
```bash
python test.py \
  --checkpoint best_model.pth \
  --test_list ./data/new_test_list.csv \
  --output_dir ./predictions_new_data
```

### 3. 특정 체크포인트로 비교 테스트
```bash
python test.py \
  --config config_production.yaml \
  --checkpoint model_epoch100.pth \
  --test_list holdout_test.csv \
  --output_dir ./final_evaluation
```

### 4. CPU에서 테스트
```bash
# config 파일에서 device: 'cpu'로 설정
python test.py --checkpoint model.pth --config config_cpu.yaml
```

## 출력 파일 구조

### Ground Truth가 있는 경우
```
{output_dir}/
├── test_20241215_143022/               # 타임스탬프별 결과 디렉토리
│   ├── all_predictions.npz            # 모든 예측 결과 (압축)
│   ├── overall_test_results.png       # 전체 결과 비교 그래프
│   ├── overall_test_results.h5        # 전체 결과 데이터
│   ├── sample_001_comparison.png      # 개별 샘플 비교
│   ├── sample_001_comparison.h5
│   ├── sample_002_comparison.png
│   ├── sample_002_comparison.h5
│   └── ...
```

### Ground Truth가 없는 경우
```
{output_dir}/
├── test_20241215_143022/
│   ├── all_predictions.npz            # 모든 예측 결과
│   ├── sample_001_prediction.png      # 개별 예측 결과
│   ├── sample_001_prediction.h5
│   ├── sample_002_prediction.png
│   ├── sample_002_prediction.h5
│   └── ...
```

## 결과 데이터 형식

### all_predictions.npz 파일 구조
```python
import numpy as np

data = np.load('all_predictions.npz')
print(data.files)
# ['predictions', 'prediction_metadata', 'target_variables', 
#  'prediction_means', 'prediction_stds', 'timestamp']

# Ground truth가 있는 경우 추가로:
# ['targets', 'test_metrics']

predictions = data['predictions']        # (n_samples, seq_len, n_vars)
metadata = data['prediction_metadata']   # 파일명, 배치 인덱스 등
target_vars = data['target_variables']   # ['ap_index', 'DST_index']

if 'test_metrics' in data.files:
    metrics = data['test_metrics'].item()  # RMSE, MAE, R² 등
```

## 실행 출력 예시

### Ground Truth가 있는 경우
```
============================================================
Test inference completed successfully!
Total Predictions: 500
Success Rate: 100.0%
Ground Truth Available: Yes

Test Metrics Summary:
  ap_index:
    RMSE: 0.1234
    MAE: 0.0987
    R²: 0.8456

  DST_index:
    RMSE: 0.2145
    MAE: 0.1678
    R²: 0.7890

Detailed results saved to: ./results/my_experiment/test/test_20241215_143022
============================================================
```

### Ground Truth가 없는 경우
```
============================================================
Test inference completed successfully!
Total Predictions: 300
Success Rate: 100.0%
Ground Truth Available: No

Detailed results saved to: ./predictions/test_20241215_143022
============================================================
```

## 사용 시나리오

### 1. 모델 최종 평가
Hold-out 테스트 세트로 최종 성능 평가:
```bash
python test.py \
  --checkpoint best_model.pth \
  --test_list holdout_test.csv \
  --output_dir final_evaluation
```

### 2. 실제 데이터 예측
새로운 실제 데이터에 대한 예측:
```bash
python test.py \
  --checkpoint production_model.pth \
  --test_list real_world_data.csv \
  --output_dir production_predictions
```

### 3. 모델 비교
여러 모델의 성능 비교:
```bash
for model in model_*.pth; do
  python test.py \
    --checkpoint $model \
    --test_list common_test.csv \
    --output_dir comparison_$(basename $model .pth)
done
```

### 4. 배치 예측
대량 데이터 처리:
```bash
python test.py \
  --checkpoint model.pth \
  --test_list batch_data_list.csv \
  --output_dir batch_predictions
```

## 결과 분석 방법

### 1. Python에서 결과 로드
```python
import numpy as np
import matplotlib.pyplot as plt

# 예측 결과 로드
data = np.load('all_predictions.npz')
predictions = data['predictions']
target_vars = data['target_variables']

# Ground truth가 있는 경우
if 'targets' in data.files:
    targets = data['targets']
    metrics = data['test_metrics'].item()
    
    # RMSE 출력
    for var in target_vars:
        print(f"{var} RMSE: {metrics[var]['rmse']:.4f}")

# 시각화
for i, var in enumerate(target_vars):
    plt.figure()
    plt.plot(predictions[0, :, i], label=f'Predicted {var}')
    if 'targets' in data.files:
        plt.plot(targets[0, :, i], label=f'True {var}')
    plt.legend()
    plt.title(f'{var} Prediction')
    plt.show()
```

### 2. 통계 분석
```python
# 예측 통계
pred_means = np.mean(predictions, axis=0)  # 시간별 평균
pred_stds = np.std(predictions, axis=0)    # 시간별 표준편차

# 변수별 분포 분석
for i, var in enumerate(target_vars):
    var_predictions = predictions[:, :, i].flatten()
    print(f"{var} - Mean: {np.mean(var_predictions):.4f}, "
          f"Std: {np.std(var_predictions):.4f}")
```

## 성능 벤치마크

### 하드웨어별 예상 처리 시간 (1000 샘플 기준)

| GPU | 처리 시간 | 메모리 사용량 |
|-----|-----------|---------------|
| RTX 4090 | ~2분 | ~4GB |
| RTX 3080 | ~3분 | ~6GB |
| RTX 2080 | ~5분 | ~8GB |
| CPU only | ~20분 | ~2GB |

## 문제 해결

### 1. 메모리 부족
```
CUDA out of memory
```
**해결책**:
- CPU로 실행: `device: 'cpu'`
- 배치 크기가 자동으로 1로 설정됨

### 2. 모델 로드 오류
```
RuntimeError: size mismatch for linear_model
```
**해결책**: 모델 아키텍처 설정이 훈련 시와 동일한지 확인

### 3. 데이터 형식 오류
```
KeyError: 'omni_Bx_GSE' not found
```
**해결책**: HDF5 파일에 필요한 변수들이 모두 포함되어 있는지 확인

### 4. 예측 결과가 이상함
- 모델이 올바른 체크포인트인지 확인
- 입력 데이터의 전처리가 훈련 시와 동일한지 확인
- NaN 값이 포함된 데이터는 자동으로 제외됨

## 주의사항

1. **데이터 품질**: NaN 값이 포함된 샘플은 자동으로 제외됩니다
2. **메모리 관리**: 대량 데이터 처리 시 충분한 메모리 확보 필요
3. **결과 저장**: 모든 샘플의 시각화 파일이 생성되므로 디스크 공간 확인
4. **모델 호환성**: 체크포인트와 설정이 훈련 시와 일치해야 함

## 고급 활용

### 1. 불확실성 정량화
예측의 신뢰도 분석:
```python
# 여러 번 예측하여 불확실성 추정
predictions_list = []
for seed in [42, 123, 456, 789, 999]:
    # 각 시드로 예측 실행
    # python test.py --seed {seed} ...
    pass
```

### 2. 시계열 예측 성능 분석
```python
# 시간 단계별 성능 분석
time_rmse = np.sqrt(np.mean((targets - predictions)**2, axis=(0, 2)))
plt.plot(time_rmse)
plt.title('RMSE by Time Step')
plt.xlabel('Time Step')
plt.ylabel('RMSE')
```

### 3. 앙상블 예측
여러 모델의 결과를 결합:
```python
# 여러 모델 예측 결합
ensemble_predictions = (pred1 + pred2 + pred3) / 3
```

test.py를 통해 모델의 실제 성능을 평가하고 실용적인 예측 결과를 얻을 수 있습니다.