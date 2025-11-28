# Solar Wind Prediction - Multimodal Deep Learning

태양풍 예측을 위한 멀티모달 딥러닝 모델입니다. Transformer (시계열 데이터)와 ConvLSTM (위성 이미지)을 결합하여 지구 자기권 교란을 예측합니다.

## 🎯 주요 특징

### 모델 아키텍처
- **Transformer**: OMNI 태양풍 시계열 데이터 처리
- **ConvLSTM**: SDO 위성 이미지 시퀀스 처리
- **Cross-Modal Fusion**: 두 모달리티 정보 통합
- **Contrastive Learning**: 멀티모달 특성 정렬

### 코드 품질
- ✅ **모듈화**: 명확한 책임 분리 (Trainer, Validator, DataProcessor 등)
- ✅ **재현성**: 완전한 시드 제어 및 결정적 알고리즘
- ✅ **효율성**: 메모리 최적화 및 DataLoader 최적화
- ✅ **확장성**: 새로운 기능 추가 용이

## 📁 프로젝트 구조

```
project/
├── configs/
│   └── local_dev.yaml        # 설정 파일
│
├── Core Components
│   ├── pipeline.py           # 데이터 파이프라인
│   ├── losses.py             # Loss 함수들
│   ├── trainers.py           # 학습 로직
│   └── validators.py         # 검증 로직
│
├── Scripts
│   ├── train.py              # 학습 스크립트
│   └── validation.py         # 검증 스크립트
│
├── Model & Utilities
│   ├── networks.py           # 모델 정의
│   └── utils.py              # 유틸리티 함수
│
└── Documentation
    ├── README.md             # 이 파일
    └── CONFIG_GUIDE.md       # 설정 가이드 (선택)
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필수 패키지 설치
pip install torch torchvision
pip install numpy pandas h5py
pip install hydra-core matplotlib

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### 2. 설정 파일 준비

`configs/local_dev.yaml` 파일에서 경로를 본인 환경에 맞게 수정:

```yaml
environment:
  data_root: '/YOUR/PATH/TO/datasets'      # 데이터셋 경로
  save_root: '/YOUR/PATH/TO/results'       # 결과 저장 경로

validation:
  checkpoint_path: /YOUR/PATH/TO/model.pth # 체크포인트 경로
  output_dir: /YOUR/PATH/TO/output         # 검증 결과 경로
```

### 3. 학습

```bash
python train.py --config-name=local_dev
```

### 4. 검증

```bash
python validation.py --config-name=local_dev
```

## 📚 주요 컴포넌트

### 1. pipeline.py - 데이터 파이프라인

**주요 클래스:**

```python
Normalizer                  # SDO/OMNI 데이터 정규화
OnlineStatistics           # 메모리 효율적 통계 계산
SamplingStrategy           # Under/Oversampling 전략
HDF5Reader                 # HDF5 파일 읽기
DataProcessor              # 데이터 전처리
MultimodalDataset          # PyTorch Dataset
```

**사용 예시:**
```python
from pipeline import create_dataloader

dataloader = create_dataloader(config, logger)
for batch in dataloader:
    sdo = batch['sdo']          # (B, C, T, H, W)
    inputs = batch['inputs']     # (B, T, C)
    targets = batch['targets']   # (B, T, C)
    labels = batch['labels']     # (B, 1, 1)
```

### 2. losses.py - Loss 함수

**주요 클래스:**

```python
MultiModalContrastiveLoss   # InfoNCE contrastive loss
MultiModalMSELoss          # MSE consistency loss
create_loss_functions()    # Config에서 loss 생성
```

**사용 예시:**
```python
from losses import create_loss_functions

reg_criterion, cont_criterion = create_loss_functions(config)

# 학습 시
reg_loss = reg_criterion(outputs, targets)
cont_loss = cont_criterion(transformer_features, convlstm_features)
total_loss = reg_loss + lambda_contrastive * cont_loss
```

### 3. trainers.py - 학습 관리

**주요 클래스:**

```python
MetricsTracker             # 학습 metrics 자동 추적
CheckpointManager          # 모델 체크포인트 관리
Trainer                    # 전체 학습 루프 관리
```

**사용 예시:**
```python
from trainers import Trainer

trainer = Trainer(
    config, model, optimizer, scheduler,
    criterion, contrastive_criterion,
    device, logger
)

history = trainer.fit(dataloader, num_epochs=10)
```

### 4. validators.py - 검증 관리

**주요 클래스:**

```python
MetricsAggregator          # 검증 metrics 집계
ResultsWriter              # 결과 파일 저장
Validator                  # 전체 검증 루프 관리
```

**사용 예시:**
```python
from validators import Validator

validator = Validator(config, model, criterion, device, logger)
results = validator.validate(dataloader)
```

## ⚙️ 설정 가이드

### 필수 경로 설정

```yaml
environment:
  seed: 250104                              # 재현성을 위한 랜덤 시드
  device: 'cuda'                            # 'cuda', 'cpu', or 'mps'
  data_root: '/path/to/datasets'            # ⚠️ 수정 필요
  save_root: '/path/to/results'             # ⚠️ 수정 필요
```

### 실험 설정

```yaml
experiment:
  experiment_name: 'my_experiment'          # 실험 이름
  phase: 'train'                            # 'train' or 'validation'
  batch_size: 8                             # 배치 크기
  num_workers: 4                            # DataLoader worker 수
```

### 샘플링 전략

```yaml
experiment:
  # Undersampling (negative 샘플 줄이기)
  enable_undersampling: false
  num_subsample: 12                         # Fold 개수
  subsample_index: 0                        # 사용할 fold (0~11)
  
  # Oversampling (positive 샘플 늘리기)
  enable_oversampling: true
  num_oversample: 13                        # 복제 횟수
```

### 모델 설정

```yaml
model:
  # Transformer
  transformer_d_model: 256
  transformer_nhead: 8
  transformer_num_layers: 3
  
  # ConvLSTM
  convlstm_hidden_channels: 64
  convlstm_num_layers: 2
  
  # Cross-Modal Fusion
  fusion_num_heads: 4
```

### 학습 설정

```yaml
training:
  num_epochs: 10
  learning_rate: 0.0002
  optimizer: 'adam'                         # 'adam' or 'sgd'
  
  # Regression loss
  loss_type: 'mse'                          # 'mse' or 'mae'
  
  # Contrastive loss
  contrastive_type: 'mse'                   # 'mse' or 'infonce'
  contrastive_temperature: 0.3              # InfoNCE temperature (if used)
  lambda_contrastive: 0.1                   # Contrastive loss weight
  
  # Logging
  report_freq: 100                          # 로그 출력 빈도 (배치 단위)
  model_save_freq: 1                        # 체크포인트 저장 빈도 (epoch 단위)
```

### 검증 설정

```yaml
validation:
  checkpoint_path: /path/to/model.pth       # ⚠️ 수정 필요
  output_dir: /path/to/output               # ⚠️ 수정 필요
  compute_alignment: true                   # Feature alignment 계산 여부
```

## 📊 출력 결과

### 학습 (`train.py`)

```
results/{experiment_name}/
├── checkpoint/
│   ├── best_model.pth                    # Best model (loss 기준)
│   ├── model_epoch1.pth                  # Epoch별 체크포인트
│   ├── model_epoch2.pth
│   └── model_final.pth                   # 최종 모델
├── log/
│   ├── training_YYYYMMDD_HHMMSS.log     # 학습 로그
│   ├── training_history.json            # Epoch별 metrics
│   └── training_curves.png              # 학습 곡선 그래프
└── snapshot/
    └── (training snapshots if enabled)
```

### 검증 (`validation.py`)

```
{output_dir}/
├── validation_results.txt                # 요약 텍스트
└── validation_results.csv                # 상세 결과 CSV
```

## 🔍 주요 기능

### 1. 자동 Metrics 추적

학습 중 metrics가 자동으로 추적되고 로깅됩니다:

```python
trainer = Trainer(...)
history = trainer.fit(dataloader, num_epochs)

# 자동으로:
# - 배치별 metrics 계산
# - 주기적 로깅 (report_freq)
# - Epoch별 요약
# - Best model 자동 저장
# - 학습 곡선 자동 플롯
```

### 2. 체계적인 Checkpoint 관리

```python
# Best model 자동 저장
checkpoint_manager.save_if_best(model, optimizer, epoch, loss)

# 주기적 저장 (model_save_freq)
checkpoint_manager.save_periodic(model, optimizer, epoch, loss)

# 최종 모델 저장
checkpoint_manager.save(..., 'model_final.pth')
```

### 3. 완전한 재현성

```python
# Config에서 시드 설정
environment:
  seed: 250104

# 자동으로 모든 랜덤성 제어:
# - Python random
# - NumPy random
# - PyTorch random (CPU & GPU)
# - CUDA backends
# - DataLoader shuffling
# - Sampling strategies
# - Data augmentation
```

### 4. 유연한 Loss 설정

```yaml
# Config만 수정하면 즉시 적용
training:
  contrastive_type: 'mse'           # MSE consistency loss
  # 또는
  contrastive_type: 'infonce'       # InfoNCE contrastive loss
```

**Contrastive Loss 선택 가이드:**
- **MSE**: 작은 배치 사이즈 (<32), 더 안정적
- **InfoNCE**: 큰 배치 사이즈 (>32), 더 강력한 표현 학습

## 🐛 트러블슈팅

### 1. Import 오류

```bash
# 오류
ModuleNotFoundError: No module named 'pipeline'

# 해결
# 프로젝트 루트 디렉토리에서 실행
cd /path/to/project
python train.py --config-name=local_dev
```

### 2. Config 파일을 찾을 수 없음

```bash
# 오류
ConfigCompositionException: Cannot find primary config 'local_dev'

# 해결
# configs/ 디렉토리 확인
ls configs/local_dev.yaml

# 또는 절대 경로 사용
python train.py --config-path=/absolute/path/to/configs --config-name=local_dev
```

### 3. GPU 메모리 부족

```yaml
# Config 수정
experiment:
  batch_size: 1              # 배치 크기 줄이기
  num_workers: 0             # Worker 수 줄이기

model:
  transformer_d_model: 128   # 모델 크기 줄이기
  convlstm_hidden_channels: 32
```

### 4. 학습 속도 느림

```yaml
# Config 최적화
experiment:
  num_workers: 4             # CPU 코어 수에 맞게 조정
  batch_size: 8              # GPU 메모리 허용 범위 내 최대

# DataLoader 자동 최적화:
# - persistent_workers=True
# - prefetch_factor=2
# - pin_memory=True (CUDA 사용 시)
```

### 5. 재현성이 보장되지 않음

```yaml
# 1. 시드가 올바르게 설정되었는지 확인
environment:
  seed: 250104

# 2. num_workers 확인
experiment:
  num_workers: 0  # 완전한 재현성을 위해 0으로 설정
  # 또는 num_workers > 0이면 worker_init_fn이 자동 적용됨
```

## 📈 성능 최적화 팁

### 1. DataLoader 최적화

```yaml
experiment:
  batch_size: 16             # GPU에 맞게 최대한 크게
  num_workers: 4             # CPU 코어 수 (보통 4-8)
```

### 2. 모델 크기 조정

```yaml
# 큰 GPU (V100, A100)
model:
  transformer_d_model: 512
  convlstm_hidden_channels: 128

# 작은 GPU (RTX 2080, 3070)
model:
  transformer_d_model: 256
  convlstm_hidden_channels: 64

# CPU only
model:
  transformer_d_model: 128
  convlstm_hidden_channels: 32
```

### 3. 혼합 정밀도 학습 (AMP)

현재 코드에는 포함되지 않았지만, 필요시 추가 가능:

```python
# trainers.py의 Trainer.train_step()에 추가
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs, sdo)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🔧 개발 가이드

### 새 Loss 함수 추가

```python
# losses.py에 추가
class MyCustomLoss(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def forward(self, pred, target):
        # 구현
        loss = ...
        return loss

# create_loss_functions() 수정
if config.training.loss_type == 'my_custom':
    criterion = MyCustomLoss(param1=..., param2=...)
```

### 새 Metric 추가

```python
# trainers.py의 MetricsTracker.reset() 수정
def reset(self):
    self.metrics = {
        'loss': [],
        'mae': [],
        'my_metric': [],  # 추가
        # ...
    }

# Trainer.train_step()에서 계산
def train_step(self, data_dict):
    # ...
    my_metric = compute_my_metric(outputs, targets)
    
    return {
        'loss': total_loss.item(),
        'my_metric': my_metric,  # 추가
        # ...
    }
```

### 새 데이터 증강 추가

```python
# pipeline.py의 DataProcessor.apply_augmentation() 수정
def apply_augmentation(self, input_array, sample_idx, file_name):
    rng = np.random.RandomState(seed)
    
    # 기존 증강
    factors = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    augmented = input_array * rng.choice(factors, size=input_array.shape)
    
    # 새 증강 추가
    if rng.random() > 0.5:
        augmented = my_custom_augmentation(augmented, rng)
    
    return augmented
```

## 📊 데이터 포맷

### HDF5 파일 구조

```
data_file.h5
├── sdo_193              # SDO 193Å 채널 (T, H, W)
├── sdo_211              # SDO 211Å 채널 (T, H, W)
├── sdo_magnetogram      # SDO Magnetogram (T, H, W)
├── omni_Bx_GSE_GSM_nT   # OMNI 변수들 (T,)
├── omni_By_GSM_nT
└── ...
```

### CSV 파일 구조

```csv
file_name,class_day1,class_day2,...
sample_001.h5,0,1,...
sample_002.h5,1,1,...
```

## 🤝 기여 가이드

1. **코드 스타일**: PEP 8 준수
2. **Docstring**: Google style docstring 사용
3. **타입 힌트**: 가능한 모든 함수에 타입 힌트 추가
4. **테스트**: 주요 기능에 대한 단위 테스트 작성

## 📄 라이선스

(프로젝트 라이선스 명시)

## 👥 저자

- Eunsu Park (KASI - Korea Astronomy and Space Science Institute)

## 📧 문의

 - eunsupark@kasi.re.kr (work)
 - phd.choux@gmail.com (home)

## 🙏 감사의 글

- SDO 데이터: NASA Solar Dynamics Observatory
- OMNI 데이터: NASA OMNIWeb

## 📚 참고 문헌

- 업데이트 예정