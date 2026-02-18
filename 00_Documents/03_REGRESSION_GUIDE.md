# 03_Regression 사용 설명서

## 목차

1. [환경 설정](#1-환경-설정)
2. [설정 시스템 (Hydra)](#2-설정-시스템-hydra)
3. [훈련 실행](#3-훈련-실행)
4. [검증 실행](#4-검증-실행)
5. [추론 (테스트) 실행](#5-추론-테스트-실행)
6. [분석 도구](#6-분석-도구)
7. [설정 변수 레퍼런스](#7-설정-변수-레퍼런스)
8. [실험 설정 예제](#8-실험-설정-예제)
9. [셸 스크립트](#9-셸-스크립트)
10. [문제 해결](#10-문제-해결)

---

## 1. 환경 설정

### 실행 환경

| 환경 | 설정 파일 | 디바이스 | 설명 |
|------|-----------|----------|------|
| 로컬 (macOS) | `local.yaml` | MPS | 개발 및 디버깅용 |
| HPC 클러스터 | `wulver.yaml` | CUDA | 전체 훈련용 |

### 작업 디렉터리

모든 스크립트는 `03_Regression/` 디렉터리에서 실행한다.

```bash
cd /opt/projects/04_NJIT/01_AP/03_Regression
```

### 데이터 경로

- **로컬**: `environment.data_root = /opt/projects/10_Harim/01_AP/03_Dataset`
- **Wulver**: `environment.data_root = /mmfs1/home/hl545/ap/final/datasets`

데이터 디렉터리 내에 다음 CSV 파일이 필요하다:
```
{data_root}/
├── original_64_full_train.csv
└── original_64_full_validation.csv
```

CSV 파일명 규칙: `{dataset_name}_{dataset_suffix}_{phase}.csv`

---

## 2. 설정 시스템 (Hydra)

### 설정 파일 구조

```
configs/
├── base.yaml           # 모든 기본값 정의
├── local.yaml          # 로컬 환경 (base.yaml 상속)
├── wulver.yaml         # HPC 환경 (base.yaml 상속)
└── experiments/        # 실험별 오버라이드
    ├── exp_day3_only.yaml
    ├── exp_short_input.yaml
    ├── exp_high_capacity.yaml
    ├── exp_contrastive_ablation.yaml
    ├── exp_transformer_only.yaml
    └── exp_convlstm_only.yaml
```

### 설정 상속 구조

```
base.yaml (기본값)
  ├── local.yaml (로컬 오버라이드)
  │     └── +experiments/exp_xxx.yaml (실험별 오버라이드)
  └── wulver.yaml (HPC 오버라이드)
        └── +experiments/exp_xxx.yaml (실험별 오버라이드)
```

### CLI 오버라이드 방법

Hydra는 커맨드라인에서 `key=value` 형식으로 설정값을 오버라이드할 수 있다.

```bash
# 단일 값 변경
python scripts/train.py --config-name=local model.model_type=transformer

# 여러 값 동시 변경
python scripts/train.py --config-name=local \
    model.model_type=baseline \
    training.learning_rate=0.001 \
    training.epochs=50

# 실험 설정 파일 추가 적용
python scripts/train.py --config-name=local +experiments=exp_high_capacity
```

---

## 3. 훈련 실행

### 방법 1: Python 직접 실행

```bash
# 로컬 (기본 fusion 모델)
python scripts/train.py --config-name=local

# 실험명 지정
python scripts/train.py --config-name=local experiment.name=fusion_v11

# 모델 타입 변경
python scripts/train.py --config-name=local \
    experiment.name=baseline_v11 \
    model.model_type=baseline

# HPC 클러스터
python scripts/train.py --config-name=wulver experiment.name=fusion_v11
```

### 방법 2: train.sh 스크립트

`train.sh`는 실험명과 모델 타입을 인자로 받아 사전 정의된 설정으로 훈련을 실행한다.

```bash
# 사용법: ./train.sh <experiment> [model_type]
# model_type 기본값: fusion

./train.sh v11                  # fusion_v11 (fusion 모델)
./train.sh v11 baseline         # baseline_v11 (baseline 모델)
./train.sh v11 transformer      # transformer_v11
./train.sh v11b baseline        # baseline_v11b (변형 실험)
```

`train.sh` 내부에는 실험 버전별 사전 정의 설정이 포함되어 있다:

| 실험 | 설명 | 주요 설정 |
|------|------|-----------|
| `v11` / `v11a` | 오버피팅 수정 (전체) | LR Warmup + Cosine + Grad Accum |
| `v11b` | 오버피팅 수정 (일부) | LR Warmup + Cosine (Grad Accum 없음) |
| `v11c` | Cosine Annealing만 | Cosine Annealing만 적용 |
| `v12` | 데이터 증강 | LR Warmup + Cosine |
| `v13` / `v13a` | TCN 기본 | 3레이어, kernel=3 |
| `v13b` | TCN 심층 | 4레이어, kernel=3 |
| `v13c` | TCN 큰 커널 | 3레이어, kernel=5 |

### 훈련 출력 구조

```
{save_root}/{experiment_name}/
├── checkpoint/
│   ├── model_epoch_0005.pth
│   ├── model_epoch_0010.pth
│   ├── model_best.pth          # 최적 검증 손실
│   └── model_final.pth         # 마지막 에폭
├── log/
│   └── training_history.json
└── plots/
    └── training_curves.png
```

### 2단계 훈련 (사전학습 체크포인트)

이전 실험의 체크포인트를 로드하여 이어서 훈련할 수 있다:

```bash
python scripts/train.py --config-name=local \
    experiment.name=fusion_stage2 \
    training.pretrained_checkpoint="fusion_stage1/checkpoint/model_best.pth"
```

경로는 `save_root` 기준 상대 경로이다.

---

## 4. 검증 실행

### 방법 1: Python 직접 실행

```bash
# 에폭 기반 (권장) - 경로 자동 생성
python scripts/validate.py --config-name=local \
    experiment.name=baseline_v7 \
    validation.epoch=best

# 특정 에폭
python scripts/validate.py --config-name=local \
    experiment.name=fusion_v11 \
    validation.epoch=10

# 명시적 경로 지정 (우선)
python scripts/validate.py --config-name=local \
    validation.checkpoint_path=/path/to/model.pth \
    validation.output_dir=/path/to/output
```

`validation.epoch` 가능한 값:
- 정수 (예: `10`, `20`) → `model_epoch_0010.pth`
- `"best"` → `model_best.pth`
- `"final"` → `model_final.pth`

### 방법 2: validation.sh 스크립트

여러 모델/에폭을 일괄 검증한다.

```bash
# 사용법: ./validation.sh <version> [model_types...]
# 기본 모델: baseline, fusion

./validation.sh 7                     # baseline_v7 + fusion_v7
./validation.sh 9 transformer         # transformer_v9만
./validation.sh 8 baseline fusion     # baseline_v8 + fusion_v8
```

이 스크립트는 에폭 5, 10, 15, 20, 25, best에 대해 순차적으로 검증을 실행한다.

### 검증 출력 구조

```
{save_root}/{experiment_name}/validation/
├── epoch_0010/
│   ├── validation_results.txt     # 메트릭 요약
│   ├── sample_0001.npz            # 예측값 + 실측값
│   ├── sample_0001.png            # 시각화 플롯
│   └── ...
└── best/
    └── ...
```

### 검증 메트릭

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (결정계수)
- **Cosine Similarity** (시계열 유사도)
- **Average Loss**

---

## 5. 추론 (테스트) 실행

정답 없이 예측만 수행할 때 사용한다.

```bash
# 에폭 기반
python scripts/test.py --config-name=local \
    experiment.name=fusion_v11 \
    test.epoch=best

# 명시적 경로
python scripts/test.py --config-name=local \
    test.checkpoint_path=/path/to/model.pth \
    test.output_dir=/path/to/output
```

---

## 6. 분석 도구

### run_analysis.sh (통합 분석)

모델 타입에 따라 적용 가능한 분석을 자동으로 선택하여 실행한다.

```bash
# 사용법: ./run_analysis.sh <model_type> <epoch> [experiment_name]

./run_analysis.sh fusion 10              # fusion/ 에 결과 저장
./run_analysis.sh fusion 10 fusion_v11   # fusion_v11/ 에 결과 저장
./run_analysis.sh baseline best baseline_v7
```

모델별 분석 지원 여부:

| 분석 | fusion | baseline | transformer | convlstm | tcn | linear |
|------|--------|----------|-------------|----------|-----|--------|
| Validation | O | O | O | O | O | O |
| Attention | O | - | O | - | - | - |
| Saliency | O | O | - | O | - | - |
| MCD (불확실성) | O | O | O | O | O | O |

---

## 7. 설정 변수 레퍼런스

### experiment (실험 설정)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `experiment.name` | `"base"` | 실험 이름 (결과 디렉터리명) |
| `experiment.seed` | `250104` | 랜덤 시드 |
| `experiment.batch_size` | `4` | 배치 크기 |

### model (모델 아키텍처)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `model.model_type` | `"fusion"` | 모델 선택 (아래 표 참조) |
| `model.d_model` | `128` | 공유 임베딩 차원 |
| `model.output_seq_len` | `24` | 출력 시퀀스 길이 (3일 x 8 = 24) |

**모델 타입**

| 타입 | 입력 | 아키텍처 | 설명 |
|------|------|----------|------|
| `fusion` | SDO + OMNI | Transformer + ConvLSTM + CrossModalFusion | 교차 모달 어텐션 융합 |
| `baseline` | SDO + OMNI | Conv3D + Linear + 결합 | Son et al. 2023 재현 |
| `transformer` | OMNI만 | Transformer Encoder | 시계열 모델 |
| `tcn` | OMNI만 | Temporal Convolutional Network | 확장 인과 합성곱 |
| `linear` | OMNI만 | Linear Encoder | 단순 기준선 |
| `convlstm` | SDO만 | ConvLSTM | 시공간 모델 |

**Transformer 설정** (`model.model_type = "transformer"` 또는 `"fusion"`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `model.transformer_nhead` | `4` | 어텐션 헤드 수 (`d_model`의 약수) |
| `model.transformer_num_layers` | `2` | 인코더 레이어 수 |
| `model.transformer_dim_feedforward` | `256` | FFN 히든 차원 |
| `model.transformer_dropout` | `0.1` | 드롭아웃 비율 |

**ConvLSTM 설정** (`model.model_type = "convlstm"` 또는 `"fusion"`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `model.convlstm_input_channels` | `3` | 입력 채널 (SDO 3파장) |
| `model.convlstm_hidden_channels` | `32` | 히든 채널 수 |
| `model.convlstm_kernel_size` | `3` | 합성곱 커널 크기 |
| `model.convlstm_num_layers` | `2` | ConvLSTM 레이어 수 |

**Cross-Modal Fusion 설정** (`model.model_type = "fusion"`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `model.fusion_num_heads` | `4` | 교차 어텐션 헤드 수 |
| `model.fusion_dropout` | `0.1` | 퓨전 드롭아웃 비율 |

**TCN 설정** (`model.model_type = "tcn"`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `model.tcn_channels` | `[64, 128, 256]` | 각 레이어의 채널 수 |
| `model.tcn_kernel_size` | `3` | 합성곱 커널 크기 (홀수) |
| `model.tcn_dropout` | `0.1` | 드롭아웃 비율 |

### training (훈련 설정)

**손실 함수**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `training.regression_loss_type` | `"solar_wind_weighted"` | 회귀 손실 (아래 표 참조) |
| `training.lambda_contrastive` | `0.5` | 대조 손실 가중치 (0.0 = 비활성) |
| `training.contrastive_loss_type` | `"consistency"` | 대조 손실 타입 (`consistency`, `infonce`) |
| `training.huber_delta` | `10.0` | Huber 손실 델타 (type=`huber`일 때) |

회귀 손실 타입:

| 타입 | 설명 |
|------|------|
| `mse` | 평균 제곱 오차 |
| `mae` | 평균 절대 오차 |
| `huber` | Huber 손실 (MSE와 MAE의 결합) |
| `weighted_mse` | 임계값 기반 가중 MSE |
| `solar_wind_weighted` | NOAA G-Scale 기반 가중 손실 (권장) |

**SolarWindWeightedLoss 세부 설정** (`training.solar_wind_weighted.*`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `base_loss` | `"mse"` | 기본 손실 (`mse`, `mae`, `huber`) |
| `weighting_mode` | `"multi_tier"` | 가중 방식 (`threshold`, `continuous`, `multi_tier`) |
| `combine_temporal` | `true` | 시간적 가중치 결합 여부 |
| `denormalize` | `true` | 역정규화 (원본 Ap 값으로 변환) |

**옵티마이저**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `training.optimizer` | `"adam"` | 옵티마이저 (`adam`, `sgd`) |
| `training.learning_rate` | `0.0002` | 학습률 |
| `training.weight_decay` | `0.0` | L2 정규화 |

**학습률 스케줄러**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `training.scheduler_type` | `"reduce_on_plateau"` | 스케줄러 타입 |

스케줄러 타입별 설정:

| 타입 | 동작 | 관련 설정 |
|------|------|-----------|
| `reduce_on_plateau` | 검증 손실 정체 시 LR 감소 | `scheduler_factor` (0.5), `scheduler_patience` (5) |
| `cosine_annealing` | 주기적 코사인 감쇠 + 재시작 | `cosine_annealing.T_0` (10), `T_mult` (2), `eta_min` (1e-6) |

**LR Warmup**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `training.lr_warmup.enable` | `false` | 워밍업 활성화 |
| `training.lr_warmup.warmup_epochs` | `5` | 워밍업 에폭 수 |
| `training.lr_warmup.warmup_start_factor` | `0.1` | 시작 학습률 비율 (기본 LR의 10%) |

**경사 누적 (Gradient Accumulation)**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `training.gradient_accumulation_steps` | `1` | 누적 스텝 (1 = 비활성, 4 = 4배치 누적) |

실질적 배치 크기 = `batch_size` × `gradient_accumulation_steps`

**훈련 루프**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `training.epochs` | `100` (wulver), `30` (local) | 총 에폭 수 |
| `training.report_freq` | `100` | 로그 출력 주기 (배치 단위) |
| `training.model_save_freq` | `10` (wulver), `5` (local) | 체크포인트 저장 주기 (에폭 단위) |
| `training.gradient_clip_max_norm` | `1.0` | 경사 클리핑 최대 노름 |
| `training.enable_plot` | `true` | 훈련 중 플롯 생성 여부 |

**조기 종료 (Early Stopping)**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `training.early_stopping_patience` | `10` | 개선 없이 허용하는 에폭 수 |
| `training.early_stopping_min_delta` | `0.0` | 개선으로 인정하는 최소 변화량 |

### sampling (샘플링 설정)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `sampling.enable_undersampling` | `false` | 언더샘플링 활성화 |
| `sampling.input_days` | `[-7, -6, ..., -1]` | 입력 윈도우 (기준 시간 이전 N일) |
| `sampling.target_days` | `[1, 2, 3]` | 예측 대상 (기준 시간 이후 N일) |

### data (데이터 설정)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `data.dataset_name` | `"original_64"` | 데이터셋 이름 |
| `data.dataset_suffix` | `"full"` | 데이터셋 접미사 |
| `data.normalization.default` | `"zscore"` | 기본 정규화 방법 |

정규화 방법:

| 방법 | 적용 대상 | 설명 |
|------|-----------|------|
| `zscore` | 양/음수 대칭 값 (Bx, By, Bz, Dst) | 표준 Z-score |
| `log_zscore` | 양수, 긴 꼬리 분포 (유속, 밀도, 온도) | log 변환 후 Z-score |
| `log1p_zscore` | 0 포함 지수 (Ap, Kp, 흑점수) | log(1+x) 후 Z-score |

### validation (검증 설정)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `validation.epoch` | `null` | 에폭 지정 (정수, `"best"`, `"final"`) |
| `validation.checkpoint_path` | `""` | 명시적 체크포인트 경로 (우선) |
| `validation.output_dir` | `""` | 명시적 출력 경로 (우선) |
| `validation.save_plots` | `true` | 예측 플롯 저장 |
| `validation.save_npz` | `true` | NPZ 파일 저장 |

---

## 8. 실험 설정 예제

### 예제 1: Transformer 모델로 OMNI 데이터만 사용

```bash
python scripts/train.py --config-name=local \
    experiment.name=transformer_exp01 \
    model.model_type=transformer
```

또는 실험 설정 파일 사용:
```bash
python scripts/train.py --config-name=local \
    +experiments=exp_transformer_only \
    experiment.name=transformer_exp01
```

### 예제 2: 고용량 모델 (HPC)

```bash
python scripts/train.py --config-name=wulver \
    +experiments=exp_high_capacity \
    experiment.name=high_cap_01
```

### 예제 3: Day 3 예측만 수행

```bash
python scripts/train.py --config-name=local \
    +experiments=exp_day3_only \
    experiment.name=day3_exp01
```

### 예제 4: 대조 손실 비활성화 (Ablation)

```bash
python scripts/train.py --config-name=local \
    experiment.name=no_contrastive_01 \
    training.lambda_contrastive=0.0
```

### 예제 5: 짧은 입력 윈도우 (3일)

```bash
python scripts/train.py --config-name=local \
    +experiments=exp_short_input \
    experiment.name=short_3d_01
```

### 예제 6: 학습률 및 배치 크기 조정

```bash
python scripts/train.py --config-name=local \
    experiment.name=lr_test_01 \
    training.learning_rate=0.001 \
    experiment.batch_size=8 \
    training.gradient_accumulation_steps=2
```

### 예제 7: Cosine Annealing + LR Warmup

```bash
python scripts/train.py --config-name=local \
    experiment.name=cosine_warmup_01 \
    training.scheduler_type=cosine_annealing \
    training.lr_warmup.enable=true \
    training.lr_warmup.warmup_epochs=5
```

### 예제 8: 훈련 후 검증 → 분석 (전체 파이프라인)

```bash
# 1. 훈련
python scripts/train.py --config-name=local \
    experiment.name=fusion_v11 \
    model.model_type=fusion

# 2. 검증 (best 에폭)
python scripts/validate.py --config-name=local \
    experiment.name=fusion_v11 \
    model.model_type=fusion \
    validation.epoch=best

# 3. 통합 분석
./run_analysis.sh fusion best fusion_v11
```

---

## 9. 셸 스크립트

### train.sh

사전 정의된 실험 설정으로 훈련을 실행한다.

```bash
./train.sh <experiment> [model_type]

# model_type 기본값: fusion
# 사용 가능한 model_type: baseline, fusion, transformer, linear, tcn
```

**주의**: `train.sh` 내부의 `cd` 경로와 결과 경로가 현재 프로젝트 경로와 다를 수 있다.
현재 스크립트는 `/opt/projects/10_Harim/01_AP/02_Regression`을 참조하고 있으므로, 필요 시 경로를 수정해야 한다.

### validation.sh

여러 에폭/모델을 일괄 검증한다.

```bash
./validation.sh <version> [model_types...]

# 기본 에폭: 5, 10, 15, 20, 25, best
# 기본 모델: baseline, fusion
```

### run_analysis.sh

검증 + 어텐션 분석 + 현저성 분석 + MCD를 순차 실행한다.

```bash
./run_analysis.sh <model_type> <epoch> [experiment_name]
```

---

## 10. 문제 해결

### 자주 발생하는 문제

**MPS 메모리 부족 (로컬)**
```bash
# 배치 크기 줄이기
python scripts/train.py --config-name=local experiment.batch_size=2

# 또는 경사 누적으로 실질 배치 크기 유지
python scripts/train.py --config-name=local \
    experiment.batch_size=2 \
    training.gradient_accumulation_steps=2
```

**체크포인트를 찾을 수 없음**
- `validation.epoch` 또는 `test.epoch`에 지정한 에폭에 해당하는 체크포인트가 존재하는지 확인
- `training.model_save_freq`에 의해 저장되지 않은 에폭은 체크포인트가 없음
- `best`와 `final`은 항상 존재

**transformer_nhead가 d_model의 약수가 아님**
- `model.transformer_nhead`와 `model.fusion_num_heads`는 반드시 `model.d_model`의 약수여야 한다
- 예: `d_model=128`이면 nhead는 1, 2, 4, 8, 16, 32, 64, 128 중 선택

**CSV 파일을 찾을 수 없음**
- CSV 파일명이 `{dataset_name}_{dataset_suffix}_{phase}.csv` 규칙을 따르는지 확인
- 기본값: `original_64_full_train.csv`, `original_64_full_validation.csv`
- `data.dataset_name`과 `data.dataset_suffix`가 실제 파일과 일치하는지 확인
