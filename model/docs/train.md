# train.py 사용법 가이드

## 개요
`train.py`는 멀티모달 딥러닝 모델을 훈련하기 위한 메인 스크립트입니다. 태양풍 시계열 데이터와 SDO 이미지 시퀀스를 결합하여 예측 모델을 훈련합니다.

## 기본 사용법

### 1. 기본 훈련 실행
```bash
python train.py --config config_dev.yaml
```

### 2. 커맨드라인 옵션으로 설정 override
```bash
python train.py --config config_dev.yaml --epochs 200 --lr 1e-4 --seed 12345
```

## 커맨드라인 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | str | 'config_dev.yaml' | 설정 파일 경로 |
| `--resume` | str | None | 재개할 체크포인트 경로 |
| `--seed` | int | None | 랜덤 시드 (config 오버라이드) |
| `--epochs` | int | None | 에포크 수 (config 오버라이드) |
| `--lr` | float | None | 학습률 (config 오버라이드) |

## 설정 파일 요구사항

훈련을 위해 다음 설정들이 config 파일에 정의되어야 합니다:

### 필수 설정
```yaml
# 기본 설정
seed: 42
device: 'cuda'  # 'cuda', 'cpu', 또는 'mps'
experiment_name: 'my_experiment'

# 데이터 경로
data_root: './data/dataset'
train_list_path: './data/train_list.csv'
validation_list_path: './data/validation_list.csv'
stat_file_path: './data/statistics.pkl'

# 훈련 설정
batch_size: 4
num_epochs: 100
learning_rate: 2e-4
loss_type: 'mse'  # 'mse', 'mae', 'huber', etc.

# 로깅 설정
report_freq: 100      # 진행 상황 리포팅 주기
model_save_freq: 5    # 모델 저장 주기
```

### 모델 아키텍처 설정
```yaml
# 입력/출력 변수
input_variables: ['Bx_GSE', 'By_GSM', 'Bz_GSM', ...]
target_variables: ['ap_index', 'DST_index']
input_sequence_length: 40
target_sequence_length: 24

# 네트워크 구조
num_linear_output: 256
inception_in_channels: 2
inception_out_channels: 96
inception_in_image_size: 64
inception_in_image_frames: 20
lstm_hidden_size: 512
```

## 실행 예시

### 1. 개발 환경에서 짧은 훈련
```bash
python train.py --config config_dev.yaml --epochs 10 --lr 1e-3
```

### 2. 프로덕션 환경에서 전체 훈련
```bash
python train.py --config config_production.yaml --seed 42
```

### 3. 중단된 훈련 재개
```bash
python train.py --config config_dev.yaml --resume ./results/my_experiment/checkpoint/model_epoch50.pth
```

### 4. GPU 메모리가 부족한 경우
```bash
python train.py --config config_dev.yaml --batch_size 2
```

## 출력 파일 구조

훈련 실행 시 다음과 같은 디렉토리 구조가 생성됩니다:

```
results/
└── {experiment_name}/
    ├── checkpoint/
    │   ├── best_model.pth      # 가장 낮은 loss의 모델
    │   ├── model_epoch5.pth    # 주기적 체크포인트
    │   ├── model_epoch10.pth
    │   └── model_final.pth     # 최종 모델
    ├── log/
    │   ├── training_20241215_143022.log  # 훈련 로그
    │   ├── training_history.json         # 훈련 이력
    │   └── training_curve.png           # 훈련 곡선
    ├── snapshot/
    │   ├── iteration_100_epoch_1.png    # 훈련 중 예측 샘플
    │   └── iteration_100_epoch_1.h5     # 예측 데이터
    └── tensorboard/
        └── events.out.tfevents.*         # TensorBoard 로그
```

## 훈련 모니터링

### 1. 콘솔 출력
```
[Epoch   1, Batch  100/1250, Iteration    100] loss: 0.045231 | Time: 12.34s | Progress: 8.0%
[Epoch   1, Batch  200/1250, Iteration    200] loss: 0.042156 | Time: 11.87s | Progress: 16.0%
```

### 2. 로그 파일
- 상세한 훈련 로그가 `{experiment_name}/log/` 디렉토리에 저장
- JSON 형태의 훈련 이력 파일 생성

### 3. 훈련 곡선
- PNG 형태의 loss curve 자동 생성
- 에포크별 loss 변화 추이 확인 가능

## 주요 기능

### 1. 자동 학습률 조정
- ReduceLROnPlateau 스케줄러 사용
- validation loss가 개선되지 않으면 학습률 자동 감소

### 2. Gradient Clipping
- exploding gradient 방지를 위한 자동 gradient clipping (max_norm=1.0)

### 3. Best Model 저장
- 가장 낮은 loss를 달성한 모델을 `best_model.pth`로 자동 저장

### 4. 중단 처리
- Ctrl+C로 훈련 중단 시 graceful shutdown
- 현재 상태까지의 모델 저장

## 문제 해결

### 1. 메모리 부족
```bash
# 배치 크기 줄이기
python train.py --config config.yaml --batch_size 2

# worker 수 줄이기 (config 파일에서 num_workers: 2)
```

### 2. 훈련이 너무 느림
```bash
# GPU 사용 확인
python train.py --config config.yaml --device cuda

# 더 많은 worker 사용 (config 파일에서 num_workers: 8)
```

### 3. loss가 수렴하지 않음
```bash
# 학습률 조정
python train.py --config config.yaml --lr 1e-5

# 다른 loss function 사용 (config 파일에서 loss_type: 'huber')
```

### 4. NaN loss 발생
- 데이터에 NaN 값이 있는지 확인
- 학습률이 너무 높지 않은지 확인
- gradient clipping이 활성화되어 있는지 확인

## 성능 최적화 팁

### 1. 데이터 로딩 최적화
```yaml
num_workers: 4      # CPU 코어 수에 맞게 조정
pin_memory: true    # GPU 사용 시 활성화
```

### 2. 배치 크기 최적화
- GPU 메모리에 맞는 최대 배치 크기 사용
- 일반적으로 4-16 사이가 적절

### 3. Mixed Precision Training (선택사항)
- PyTorch AMP 사용 고려
- 메모리 사용량 감소 및 속도 향상

## 주의사항

1. **데이터 품질**: NaN 값이 포함된 데이터는 자동으로 제외됩니다
2. **디스크 공간**: 체크포인트와 로그 파일이 상당한 용량을 차지할 수 있습니다
3. **재현성**: 동일한 결과를 위해서는 seed 값을 고정해야 합니다
4. **모델 크기**: 체크포인트 파일은 모델 크기에 따라 수백MB가 될 수 있습니다

## 예상 훈련 시간

하드웨어 사양에 따른 대략적인 훈련 시간:

| GPU | 배치 크기 | 에포크당 시간 | 100 에포크 |
|-----|-----------|---------------|------------|
| RTX 4090 | 8 | ~5분 | ~8시간 |
| RTX 3080 | 4 | ~8분 | ~13시간 |
| RTX 2080 | 4 | ~12분 | ~20시간 |
| CPU only | 2 | ~45분 | ~75시간 |

*실제 시간은 데이터 크기와 모델 복잡도에 따라 달라질 수 있습니다.*