# validation.py 사용법 가이드

## 개요
`validation.py`는 훈련된 멀티모달 딥러닝 모델의 성능을 검증하고 평가하기 위한 스크립트입니다. 다양한 평가 지표를 계산하고 시각화 결과를 생성합니다.

## 기본 사용법

### 1. 기본 검증 실행
```bash
python validation.py --checkpoint ./results/my_experiment/checkpoint/best_model.pth
```

### 2. 특정 설정 파일과 함께 실행
```bash
python validation.py --config config_dev.yaml --checkpoint ./results/my_experiment/checkpoint/model_epoch50.pth
```

### 3. 결과 저장 디렉토리 지정
```bash
python validation.py --checkpoint best_model.pth --output_dir ./validation_results
```

## 커맨드라인 옵션

| 옵션 | 타입 | 기본값 | 필수 | 설명 |
|------|------|--------|------|------|
| `--config` | str | 'config_dev.yaml' | 아니오 | 설정 파일 경로 |
| `--checkpoint` | str | None | **예** | 평가할 모델 체크포인트 경로 |
| `--output_dir` | str | None | 아니오 | 결과 저장 디렉토리 |
| `--seed` | int | None | 아니오 | 랜덤 시드 (config 오버라이드) |

## 설정 파일 요구사항

### 필수 설정
```yaml
# 데이터 경로
data_root: './data/dataset'
validation_list_path: './data/validation_list.csv'
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

## 실행 예시

### 1. 기본 검증
```bash
python validation.py --checkpoint ./results/experiment_1/checkpoint/best_model.pth
```

### 2. 특정 에포크 모델 검증
```bash
python validation.py \
  --config config_production.yaml \
  --checkpoint ./results/experiment_1/checkpoint/model_epoch75.pth \
  --output_dir ./validation_epoch75
```

### 3. 다른 시드로 검증
```bash
python validation.py \
  --checkpoint best_model.pth \
  --seed 12345
```

### 4. CPU에서 검증
```bash
# config 파일에서 device: 'cpu'로 설정하거나
python validation.py --checkpoint model.pth --config config_cpu.yaml
```

## 출력 파일 구조

검증 실행 시 다음과 같은 파일들이 생성됩니다:

```
{output_dir}/
├── validation_20241215_143022/      # 타임스탬프별 결과 디렉토리
│   ├── overall_validation_results.png  # 전체 결과 비교 그래프
│   ├── overall_validation_results.h5   # 전체 결과 데이터
│   ├── sample_001_comparison.png       # 개별 샘플 비교 그래프
│   ├── sample_001_comparison.h5        # 개별 샘플 데이터
│   ├── sample_002_comparison.png
│   ├── sample_002_comparison.h5
│   └── ...
```

## 평가 지표

### 자동 계산되는 지표들
1. **RMSE (Root Mean Square Error)**: 예측 정확도의 기본 지표
2. **MAE (Mean Absolute Error)**: 절대 오차의 평균
3. **Correlation**: 예측값과 실제값 간의 상관관계
4. **R² (R-squared)**: 결정계수, 모델의 설명력

### 변수별 지표 계산
각 target 변수(예: ap_index, DST_index)에 대해 개별적으로 모든 지표가 계산됩니다.

## 실행 출력 예시

```
========================================
Validation Results Summary
Average Loss: 0.0234 (±0.0045)
Total Samples: 1250
Failed Batches: 0
Success Rate: 100.0%

Metrics for ap_index:
  RMSE: 0.1456
  MAE: 0.1123
  Correlation: 0.8934
  R²: 0.7891

Metrics for DST_index:
  RMSE: 0.2134
  MAE: 0.1678
  Correlation: 0.8654
  R²: 0.7234

RMSE by Variable:
  ap_index: 0.1456
  DST_index: 0.2134

Results saved to: ./results/my_experiment/validation/validation_20241215_143022
========================================
```

## 결과 분석

### 1. 전체 결과 그래프
- `overall_validation_results.png`: 모든 샘플의 평균적인 예측 성능
- 시계열 형태로 예측값과 실제값을 비교 표시

### 2. 개별 샘플 분석
- 각 validation 샘플에 대한 개별 예측 결과
- 특정 패턴이나 오류를 분석할 때 유용

### 3. HDF5 데이터 파일
- 시각화 그래프의 원본 데이터
- 추가 분석이나 다른 도구에서 활용 가능

## 사용 시나리오

### 1. 모델 선택
여러 에포크의 모델 중 가장 성능이 좋은 것을 선택:
```bash
python validation.py --checkpoint model_epoch25.pth --output_dir val_e25
python validation.py --checkpoint model_epoch50.pth --output_dir val_e50
python validation.py --checkpoint model_epoch75.pth --output_dir val_e75
# 결과 비교 후 최적 모델 선택
```

### 2. 하이퍼파라미터 효과 분석
다른 설정으로 훈련된 모델들 비교:
```bash
python validation.py --checkpoint model_lr1e-3.pth --output_dir val_lr_high
python validation.py --checkpoint model_lr1e-4.pth --output_dir val_lr_mid
python validation.py --checkpoint model_lr1e-5.pth --output_dir val_lr_low
```

### 3. 정기적 성능 모니터링
```bash
# 매주 성능 검증 스크립트
for epoch in 25 50 75 100; do
  python validation.py \
    --checkpoint model_epoch${epoch}.pth \
    --output_dir weekly_val_epoch${epoch}
done
```

## 성능 지표 해석

### RMSE (낮을수록 좋음)
- **< 0.1**: 매우 우수한 성능
- **0.1-0.2**: 좋은 성능
- **0.2-0.3**: 보통 성능
- **> 0.3**: 개선 필요

### Correlation (1.0에 가까울수록 좋음)
- **> 0.9**: 매우 강한 상관관계
- **0.7-0.9**: 강한 상관관계
- **0.5-0.7**: 보통 상관관계
- **< 0.5**: 약한 상관관계

### R² (1.0에 가까울수록 좋음)
- **> 0.8**: 모델이 변동성의 80% 이상 설명
- **0.6-0.8**: 양호한 설명력
- **0.4-0.6**: 보통 설명력
- **< 0.4**: 설명력 부족

## 문제 해결

### 1. 체크포인트 파일 오류
```
Error: Model checkpoint not found: ./model.pth
```
**해결책**: 체크포인트 파일 경로를 확인하세요.

### 2. 설정 불일치 오류
```
Error: RuntimeError: Expected tensor size (batch, 40, 12) but got (batch, 35, 12)
```
**해결책**: 모델 아키텍처 설정이 훈련 시와 동일한지 확인하세요.

### 3. 메모리 부족
```
CUDA out of memory
```
**해결책**: 
- CPU로 실행: `device: 'cpu'`
- 또는 batch_size를 1로 고정 (자동으로 설정됨)

### 4. 데이터 파일 없음
```
FileNotFoundError: validation_list.csv not found
```
**해결책**: validation 데이터 목록 파일이 올바른 경로에 있는지 확인하세요.

## 주의사항

1. **모델 호환성**: 체크포인트와 설정 파일이 호환되는지 확인
2. **데이터 일관성**: 훈련 시와 동일한 전처리 통계 사용
3. **메모리 사용량**: 큰 validation 세트는 상당한 메모리 사용
4. **디스크 공간**: 모든 샘플의 시각화 파일이 생성되므로 충분한 공간 필요

## 고급 사용법

### 1. 스크립트를 통한 자동화
```bash
#!/bin/bash
# validate_all_models.sh

MODELS_DIR="./results/experiment/checkpoint"
OUTPUT_BASE="./validation_results"

for model in ${MODELS_DIR}/model_epoch*.pth; do
    epoch=$(basename $model .pth | sed 's/model_epoch//')
    python validation.py \
        --checkpoint $model \
        --output_dir ${OUTPUT_BASE}/epoch_${epoch}
done
```

### 2. 배치 처리를 위한 Python 스크립트
```python
import subprocess
import glob

models = glob.glob("./results/*/checkpoint/best_model.pth")
for model in models:
    exp_name = model.split('/')[2]
    subprocess.run([
        "python", "validation.py",
        "--checkpoint", model,
        "--output_dir", f"./batch_validation/{exp_name}"
    ])
```

## 결과 활용

1. **논문/리포트**: 생성된 그래프와 지표를 성능 평가 자료로 활용
2. **모델 개선**: 낮은 성능을 보이는 변수나 시점 분석
3. **하이퍼파라미터 튜닝**: 여러 모델 비교를 통한 최적화
4. **운영 모델 선택**: 실제 배포할 모델의 성능 검증

validation.py를 통해 모델의 성능을 종합적으로 평가하고 개선 방향을 찾을 수 있습니다.