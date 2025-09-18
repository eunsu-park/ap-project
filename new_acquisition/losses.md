# Time Series Loss Functions Documentation

이 문서는 이상치와 급격한 변화가 중요하고, 미래 시점에 더 높은 가중치를 주는 시계열 회귀 작업을 위한 6가지 우선순위별 손실함수에 대한 상세한 설명을 제공합니다.

## 목차
1. [Priority 1: Huber Multi-criteria Loss](#priority-1-huber-multi-criteria-loss)
2. [Priority 2: MAE Outlier-focused Loss](#priority-2-mae-outlier-focused-loss)
3. [Priority 3: Adaptive Weight Loss](#priority-3-adaptive-weight-loss)
4. [Priority 4: Gradient-based Weight Loss](#priority-4-gradient-based-weight-loss)
5. [Priority 5: Quantile Loss](#priority-5-quantile-loss)
6. [Priority 6: Multi-task Loss](#priority-6-multi-task-loss)
7. [사용 가이드](#사용-가이드)
8. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)

---

## Priority 1: Huber Multi-criteria Loss

### 🥇 가장 추천되는 손실함수

**클래스명**: `Priority1_HuberMultiCriteriaLoss`

### 개요
Huber Loss를 기반으로 하여 시간적 가중치(미래 강조)와 변화율 기반 가중치(급변 강조)를 결합한 손실함수입니다. 이상치에 robust하면서도 요구사항을 모두 만족하는 균형잡힌 접근법입니다.

### 수학적 정의

**기본 Huber Loss:**
```
HuberLoss(x) = {
    0.5 * x² / β,           if |x| ≤ β
    |x| - 0.5 * β,          if |x| > β
}
```

**전체 손실함수:**
```
L = Σᵢⱼ [w_temporal(j) × w_gradient(i,j) × HuberLoss(pred(i,j) - target(i,j))]
```

### 주요 특징

#### 1. **Huber Loss 기반**
- **장점**: MSE와 MAE의 장점을 결합
- **소규모 오차**: L2 손실처럼 동작 (부드러운 그래디언트)
- **대규모 오차**: L1 손실처럼 동작 (이상치에 robust)
- **β 파라미터**: 전환점을 조정 (기본값: 0.3)

#### 2. **시간적 가중치 (Temporal Weighting)**
```python
temporal_weights = torch.linspace(start_weight, end_weight, seq_len)
```
- **목적**: 미래 시점에 더 높은 중요도 부여
- **구현**: 선형적으로 증가하는 가중치
- **기본값**: (0.3, 1.0) - 시작점 30%, 끝점 100%

#### 3. **변화율 기반 가중치 (Gradient-based Weighting)**
```python
grad = torch.diff(target, dim=1)  # 시간적 변화율
grad_magnitude = torch.norm(grad, dim=2)  # 변화의 크기
grad_weights = torch.softmax(grad_magnitude * scale, dim=1)
```
- **목적**: 급격한 변화 구간에 더 높은 중요도 부여
- **구현**: 시간적 그래디언트의 크기에 비례
- **정규화**: Softmax로 가중치 분포 조정

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 튜닝 범위 |
|---------|--------|------|-----------|
| `beta` | 0.3 | Huber loss 전환점 | 0.1 - 1.0 |
| `temporal_weight_range` | (0.3, 1.0) | 시간 가중치 범위 | (0.1-0.5, 0.8-1.0) |
| `gradient_weight_scale` | 2.0 | 변화율 가중치 스케일 | 1.0 - 5.0 |

### 언제 사용하나?
- **첫 번째 선택**: 대부분의 경우에 권장
- **균형적 접근**: 안정성과 요구사항 달성의 균형
- **실용적**: 하이퍼파라미터 튜닝이 상대적으로 쉬움

### 장단점

**장점:**
- ✅ 이상치에 robust
- ✅ 급변 구간 강조
- ✅ 미래 시점 강조
- ✅ 안정적인 학습
- ✅ 해석 가능한 가중치

**단점:**
- ❌ 매우 극단적인 이상치에는 여전히 민감할 수 있음
- ❌ 두 개의 가중치 스케일 조정 필요

---

## Priority 2: MAE Outlier-focused Loss

### 🥈 이상치 집중 손실함수

**클래스명**: `Priority2_MAEOutlierFocusedLoss`

### 개요
MAE Loss를 기반으로 Z-score를 이용한 명시적 이상치 탐지와 함께 미래 시점 가중치를 결합한 손실함수입니다. 이상치 구간에 가장 직접적으로 높은 중요도를 부여합니다.

### 수학적 정의

**기본 MAE Loss:**
```
MAELoss(x) = |x|
```

**이상치 탐지:**
```
Z-score = |target - mean| / std
outlier_mask = Z-score > threshold
outlier_weight = 1 + mask × (multiplier - 1)
```

**전체 손실함수:**
```
L = Σᵢⱼ [w_temporal(j) × w_outlier(i,j) × MAELoss(pred(i,j) - target(i,j))]
```

### 주요 특징

#### 1. **MAE Loss 기반**
- **완전한 이상치 내성**: 제곱항이 없어 극값에 덜 민감
- **선형 패널티**: 오차에 비례하는 일정한 그래디언트
- **Robust 특성**: 가장 안정적인 기본 손실함수

#### 2. **Z-score 기반 이상치 탐지**
```python
mean = target.mean(dim=2, keepdim=True)
std = target.std(dim=2, keepdim=True) + 1e-8
z_scores = torch.abs((target - mean) / std)
max_z_score = z_scores.max(dim=2)[0]
outlier_mask = (max_z_score > threshold).float()
```
- **통계적 접근**: 표준편차 기반 이상치 정의
- **동적 탐지**: 배치별로 이상치 기준 조정
- **차원별 고려**: 모든 feature 차원을 고려한 이상치 판정

#### 3. **이상치 가중치 증폭**
```python
outlier_weights = 1.0 + outlier_mask * (multiplier - 1.0)
```
- **선택적 강조**: 이상치로 판정된 시점만 가중치 증가
- **배수 조정**: multiplier로 강조 정도 조절

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 튜닝 범위 |
|---------|--------|------|-----------|
| `outlier_threshold` | 2.0 | Z-score 이상치 임계값 | 1.5 - 3.0 |
| `outlier_weight_multiplier` | 3.0 | 이상치 가중치 배수 | 2.0 - 5.0 |
| `temporal_weight_range` | (0.3, 1.0) | 시간 가중치 범위 | (0.1-0.5, 0.8-1.0) |

### 언제 사용하나?
- **이상치가 매우 중요한 경우**
- **노이즈가 심한 데이터**
- **Priority 1으로 이상치 성능이 부족한 경우**
- **해석 가능한 이상치 탐지가 필요한 경우**

### 장단점

**장점:**
- ✅ 이상치에 가장 robust
- ✅ 명시적 이상치 탐지
- ✅ 해석 가능한 가중치
- ✅ 안정적인 학습
- ✅ 극한 상황에서도 안정

**단점:**
- ❌ 정상 구간에서 성능이 다소 떨어질 수 있음
- ❌ Z-score 기반 탐지의 한계
- ❌ 이상치 기준이 고정적

---

## Priority 3: Adaptive Weight Loss

### 🥉 적응형 가중치 손실함수

**클래스명**: `Priority3_AdaptiveWeightLoss`

### 개요
오차의 크기에 따라 동적으로 가중치를 조정하는 손실함수입니다. 큰 오차(이상치/급변)일수록 자동으로 더 높은 가중치를 부여하여 학습 과정에서 적응적으로 중요도를 조절합니다.

### 수학적 정의

**적응형 가중치:**
```
error_magnitude = ||pred - target||₂
adaptive_weight = (error_magnitude + ε)^power
normalized_weight = adaptive_weight / mean(adaptive_weight)
```

**전체 손실함수:**
```
L = Σᵢⱼ [w_temporal(j) × w_adaptive(i,j) × BaseLoss(pred(i,j) - target(i,j))]
```

### 주요 특징

#### 1. **동적 가중치 조정**
```python
error_magnitude = torch.norm(errors, dim=2)
adaptive_weights = torch.pow(error_magnitude + 1e-8, adaptive_power)
adaptive_weights = adaptive_weights / (adaptive_weights.mean(dim=1, keepdim=True) + 1e-8)
```
- **자동 탐지**: 별도의 임계값 설정 없이 오차 크기로 자동 판단
- **연속적 가중치**: 이진 마스크가 아닌 연속적인 가중치
- **정규화**: 시퀀스별 정규화로 안정성 확보

#### 2. **Power Law 스케일링**
- **Adaptive Power**: 가중치 증가 속도 조절
- **Power < 1**: 부드러운 가중치 증가
- **Power > 1**: 급격한 가중치 증가
- **Power = 1**: 선형 증가

#### 3. **기본 손실함수 선택**
```python
if base_loss_type == 'mse':
    self.base_loss_fn = nn.MSELoss(reduction='none')
elif base_loss_type == 'mae':
    self.base_loss_fn = nn.L1Loss(reduction='none')
elif base_loss_type == 'huber':
    self.base_loss_fn = nn.SmoothL1Loss(beta=beta, reduction='none')
```

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 튜닝 범위 |
|---------|--------|------|-----------|
| `base_loss_type` | 'huber' | 기본 손실함수 타입 | 'mse', 'mae', 'huber' |
| `beta` | 0.5 | Huber loss 파라미터 | 0.1 - 1.0 |
| `adaptive_power` | 1.5 | 적응형 가중치 지수 | 1.0 - 2.5 |
| `temporal_weight_range` | (0.3, 1.0) | 시간 가중치 범위 | (0.1-0.5, 0.8-1.0) |

### 언제 사용하나?
- **자동화된 가중치 조정이 필요한 경우**
- **이상치의 정의가 명확하지 않은 경우**
- **복잡한 패턴의 데이터**
- **하이퍼파라미터 튜닝 시간이 제한적인 경우**

### 장단점

**장점:**
- ✅ 자동 가중치 조정
- ✅ 연속적인 중요도 부여
- ✅ 다양한 기본 손실함수 지원
- ✅ 데이터 적응적

**단점:**
- ❌ 구현 복잡도 높음
- ❌ 해석이 상대적으로 어려움
- ❌ 가중치 계산 오버헤드

---

## Priority 4: Gradient-based Weight Loss

### 변화율 기반 가중치 손실함수

**클래스명**: `Priority4_GradientBasedWeightLoss`

### 개요
시간적 변화율(gradient)의 크기에 따라 가중치를 부여하는 손실함수입니다. 급격한 변화가 일어나는 시점에 집중적으로 높은 중요도를 부여합니다.

### 수학적 정의

**시간적 그래디언트:**
```
grad(t) = target(t+1) - target(t)
grad_magnitude = ||grad(t)||₂
```

**그래디언트 가중치:**
```
gradient_weight = exp(grad_magnitude × scale / max(grad_magnitude))
```

**전체 손실함수:**
```
L = Σᵢⱼ [w_temporal(j) × w_gradient(i,j) × BaseLoss(pred(i,j) - target(i,j))]
```

### 주요 특징

#### 1. **시간적 그래디언트 계산**
```python
grad = torch.diff(target, dim=1)  # 연속 시점 간 차이
grad_magnitude = torch.norm(grad, dim=2)  # 변화량 크기
grad_magnitude = F.pad(grad_magnitude, (1, 0), value=0)  # 패딩으로 길이 맞춤
```
- **차분 계산**: 연속된 시점 간의 변화량 계산
- **노름 계산**: 다차원 변화량을 스칼라로 변환
- **패딩 처리**: 원래 시퀀스 길이와 맞춤

#### 2. **지수적 가중치**
```python
gradient_weights = torch.exp(grad_magnitude × scale / (max_grad + ε))
```
- **지수 함수**: 급변 구간을 강하게 강조
- **정규화**: 최대값으로 나누어 안정성 확보
- **클램핑**: 극단적인 가중치 방지

#### 3. **직접적 급변 타겟팅**
- **명확한 목적**: 변화율에만 집중
- **즉각적 반응**: 변화가 감지되는 즉시 가중치 증가
- **시간적 연속성**: 연속된 시점의 관계 고려

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 튜닝 범위 |
|---------|--------|------|-----------|
| `base_loss_type` | 'mae' | 기본 손실함수 타입 | 'mse', 'mae', 'huber' |
| `beta` | 0.5 | Huber loss 파라미터 | 0.1 - 1.0 |
| `gradient_weight_scale` | 3.0 | 그래디언트 가중치 스케일 | 1.0 - 5.0 |
| `temporal_weight_range` | (0.3, 1.0) | 시간 가중치 범위 | (0.1-0.5, 0.8-1.0) |

### 언제 사용하나?
- **급격한 변화가 가장 중요한 특징인 경우**
- **시계열의 변화점 탐지가 목적인 경우**
- **이상치보다 변화율이 더 중요한 경우**
- **시간적 패턴이 중요한 경우**

### 장단점

**장점:**
- ✅ 급변 구간 직접 타겟팅
- ✅ 시간적 연속성 고려
- ✅ 명확한 목적성
- ✅ 변화점 탐지에 효과적

**단점:**
- ❌ 변화율이 작은 이상치 놓칠 수 있음
- ❌ 노이즈에 민감할 수 있음
- ❌ 첫 번째 시점 가중치 항상 0

---

## Priority 5: Quantile Loss

### 분위수 회귀 손실함수

**클래스명**: `Priority5_QuantileLoss`

### 개요
예측 구간을 제공하는 분위수 회귀를 위한 손실함수입니다. 불확실성이 높은 구간에 더 높은 가중치를 부여하며, 점 추정값뿐만 아니라 예측 구간도 함께 제공합니다.

### 수학적 정의

**분위수 손실:**
```
QuantileLoss(τ) = max(τ × (y - ŷ), (τ - 1) × (y - ŷ))
```
여기서 τ는 분위수 (0 < τ < 1)

**불확실성 가중치:**
```
uncertainty = ||q_high - q_low||₂  # 분위수 간 거리
uncertainty_weight = 1 + uncertainty × scale
```

**전체 손실함수:**
```
L = Σᵢⱼτ [w_temporal(j) × w_uncertainty(i,j) × QuantileLoss_τ(pred_τ(i,j), target(i,j))]
```

### 주요 특징

#### 1. **다중 분위수 예측**
```python
for i, quantile in enumerate(self.quantiles):
    pred_q = pred[..., i]
    q_loss = self._quantile_loss(pred_q, target, quantile)
    total_loss += q_loss
```
- **동시 학습**: 여러 분위수를 한 번에 학습
- **예측 구간**: 신뢰구간 제공 가능
- **기본 분위수**: [0.1, 0.5, 0.9] (10%, 50%, 90%)

#### 2. **비대칭 손실함수**
```python
def _quantile_loss(self, pred, target, quantile):
    errors = target - pred
    return torch.maximum(quantile * errors, (quantile - 1) * errors)
```
- **분위수별 특성**: 각 분위수에 맞는 비대칭 패널티
- **상한/하한**: 과소/과대 예측에 다른 패널티
- **확률적 해석**: 분위수의 확률적 의미 반영

#### 3. **불확실성 기반 가중치**
```python
uncertainty = torch.norm(q_high - q_low, dim=2)  # IQR
uncertainty_weights = 1.0 + uncertainty * scale
```
- **구간 너비**: 분위수 간 거리로 불확실성 측정
- **적응적 가중치**: 불확실한 구간에 더 높은 중요도
- **해석 가능성**: 넓은 예측 구간 = 높은 불확실성

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 튜닝 범위 |
|---------|--------|------|-----------|
| `quantiles` | [0.1, 0.5, 0.9] | 예측할 분위수 목록 | 다양한 조합 가능 |
| `temporal_weight_range` | (0.3, 1.0) | 시간 가중치 범위 | (0.1-0.5, 0.8-1.0) |
| `uncertainty_weight_scale` | 2.0 | 불확실성 가중치 스케일 | 1.0 - 5.0 |

### 언제 사용하나?
- **예측 불확실성이 중요한 경우**
- **신뢰구간이 필요한 경우**
- **리스크 관리가 중요한 응용 분야**
- **확률적 예측이 필요한 경우**

### 장단점

**장점:**
- ✅ 예측 구간 제공
- ✅ 불확실성 정량화
- ✅ 확률적 해석 가능
- ✅ 리스크 관리 용이

**단점:**
- ❌ 모델 출력 형태 변경 필요
- ❌ 단순 회귀보다 복잡
- ❌ 분위수 선택의 어려움
- ❌ 계산 오버헤드 증가

---

## Priority 6: Multi-task Loss

### 다중 작업 학습 손실함수

**클래스명**: `Priority6_MultiTaskLoss`

### 개요
회귀 작업과 이상치 탐지 작업을 동시에 학습하는 손실함수입니다. 보조 작업(이상치 탐지)을 통해 이상치 구간에서의 회귀 성능을 향상시킵니다.

### 수학적 정의

**회귀 손실:**
```
L_regression = BaseLoss(pred_regression, target)
```

**이상치 탐지 손실:**
```
L_outlier = BCEWithLogitsLoss(pred_outlier, outlier_labels)
```

**전체 손실함수:**
```
L = w_temporal × (L_regression + λ × L_outlier)
```

### 주요 특징

#### 1. **동시 다중 작업**
```python
# 회귀 손실
regression_loss = self.regression_loss_fn(pred, target)

# 이상치 탐지 손실
if outlier_logits is not None:
    outlier_labels = self._generate_outlier_labels(target)
    outlier_loss = self.outlier_loss_fn(outlier_logits, outlier_labels)
```
- **주 작업**: 시계열 회귀
- **보조 작업**: 이상치 이진 분류
- **상호 보완**: 두 작업이 서로 도움

#### 2. **자동 이상치 라벨 생성**
```python
def _generate_outlier_labels(self, target):
    mean = target.mean(dim=2, keepdim=True)
    std = target.std(dim=2, keepdim=True) + 1e-8
    z_scores = torch.abs((target - mean) / std)
    max_z_score = z_scores.max(dim=2)[0]
    outlier_labels = (max_z_score > self.outlier_threshold).float()
    return outlier_labels
```
- **Z-score 기반**: 통계적 이상치 정의
- **동적 생성**: 학습 중 자동으로 라벨 생성
- **이진 분류**: 이상치/정상 구분

#### 3. **가중치 균형**
```python
total_loss = weighted_regression_loss + self.outlier_loss_weight * weighted_outlier_loss
```
- **가중치 조정**: 두 작업 간 균형 조절
- **주/보조 관계**: 회귀가 주, 이상치 탐지가 보조
- **안정화**: 보조 작업이 주 작업 안정화

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 튜닝 범위 |
|---------|--------|------|-----------|
| `regression_loss_type` | 'huber' | 회귀 손실함수 타입 | 'mse', 'mae', 'huber' |
| `beta` | 0.5 | Huber loss 파라미터 | 0.1 - 1.0 |
| `outlier_loss_weight` | 0.3 | 이상치 탐지 손실 가중치 | 0.1 - 0.5 |
| `temporal_weight_range` | (0.3, 1.0) | 시간 가중치 범위 | (0.1-0.5, 0.8-1.0) |
| `outlier_threshold` | 2.0 | 이상치 Z-score 임계값 | 1.5 - 3.0 |

### 언제 사용하나?
- **이상치 탐지 능력이 필요한 경우**
- **복잡한 모델로 실험할 여유가 있는 경우**
- **이상치 구간 식별이 부차적 목표인 경우**
- **다양한 출력이 필요한 경우**

### 장단점

**장점:**
- ✅ 포괄적인 접근
- ✅ 이상치 탐지 능력 향상
- ✅ 상호 보완적 학습
- ✅ 다양한 출력 제공

**단점:**
- ❌ 모델 복잡도 증가
- ❌ 하이퍼파라미터 튜닝 복잡
- ❌ 학습 시간 증가
- ❌ 구현 복잡도 높음

---

## 사용 가이드

### 📋 선택 가이드

#### 1. **빠른 시작 (권장)**
```python
# 대부분의 경우에 적합
loss_fn = Priority1_HuberMultiCriteriaLoss(
    beta=0.3,
    temporal_weight_range=(0.3, 1.0),
    gradient_weight_scale=2.0
)
```

#### 2. **이상치가 매우 중요한 경우**
```python
# 이상치 집중
loss_fn = Priority2_MAEOutlierFocusedLoss(
    outlier_threshold=2.0,
    outlier_weight_multiplier=3.0
)
```

#### 3. **자동화를 원하는 경우**
```python
# 적응형 가중치
loss_fn = Priority3_AdaptiveWeightLoss(
    adaptive_power=1.5,
    base_loss_type='huber'
)
```

### 🔄 실험 순서

1. **Priority 1**으로 시작하여 베이스라인 설정
2. **이상치 성능**이 부족하면 **Priority 2** 시도
3. **복잡한 패턴**이면 **Priority 3** 시도
4. **급변이 핵심**이면 **Priority 4** 시도
5. **불확실성**이 중요하면 **Priority 5** 시도
6. **포괄적 접근**이 필요하면 **Priority 6** 시도

### 📊 성능 평가

#### 전체 성능 지표
- **전체 MSE/MAE**: 일반적인 회귀 성능
- **가중 손실값**: 사용자 정의 손실함수 값

#### 구간별 성능 지표
- **이상치 구간 성능**: 이상치로 분류된 시점들의 성능
- **급변 구간 성능**: 높은 그래디언트 시점들의 성능
- **미래 시점 성능**: 시퀀스 후반부의 성능
- **정상 구간 성능**: 일반적인 시점들의 성능

#### 성능 평가 코드 예시
```python
def evaluate_performance(loss_fn, pred, target):
    """손실함수별 세부 성능 평가"""
    
    # 전체 성능
    total_loss = loss_fn(pred, target)
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    
    # 이상치 구간 식별
    mean = target.mean(dim=2, keepdim=True)
    std = target.std(dim=2, keepdim=True) + 1e-8
    z_scores = torch.abs((target - mean) / std)
    outlier_mask = z_scores.max(dim=2)[0] > 2.0
    
    # 급변 구간 식별
    grad = torch.diff(target, dim=1)
    grad_magnitude = torch.norm(grad, dim=2)
    grad_magnitude = F.pad(grad_magnitude, (1, 0), value=0)
    rapid_change_mask = grad_magnitude > grad_magnitude.quantile(0.8, dim=1, keepdim=True)
    
    # 미래 시점 (후반 30%)
    seq_len = target.shape[1]
    future_mask = torch.zeros_like(outlier_mask)
    future_mask[:, int(seq_len*0.7):] = 1
    
    # 구간별 성능 계산
    results = {
        'total_loss': total_loss.item(),
        'total_mse': mse.item(),
        'total_mae': mae.item(),
        'outlier_mse': F.mse_loss(pred[outlier_mask], target[outlier_mask]).item() if outlier_mask.any() else 0,
        'rapid_change_mse': F.mse_loss(pred[rapid_change_mask], target[rapid_change_mask]).item() if rapid_change_mask.any() else 0,
        'future_mse': F.mse_loss(pred[future_mask.bool()], target[future_mask.bool()]).item(),
        'outlier_ratio': outlier_mask.float().mean().item(),
        'rapid_change_ratio': rapid_change_mask.float().mean().item()
    }
    
    return results
```

---

## 하이퍼파라미터 튜닝

### 🎯 튜닝 전략

#### 1단계: 기본 설정
```python
# 모든 손실함수의 공통 시작점
common_config = {
    'temporal_weight_range': (0.3, 1.0),  # 미래 강조 기본값
    'reduction': 'mean'
}
```

#### 2단계: 손실함수별 핵심 파라미터 튜닝

**Priority 1 - Huber Multi-criteria**
```python
# 그리드 서치 범위
param_grid = {
    'beta': [0.1, 0.3, 0.5, 0.7, 1.0],
    'gradient_weight_scale': [1.0, 2.0, 3.0, 4.0, 5.0],
    'temporal_weight_range': [(0.2, 1.0), (0.3, 1.0), (0.4, 1.0)]
}

# 튜닝 순서
# 1. beta 먼저 조정 (이상치 민감도)
# 2. gradient_weight_scale 조정 (급변 강조)
# 3. temporal_weight_range 미세 조정
```

**Priority 2 - MAE Outlier-focused**
```python
param_grid = {
    'outlier_threshold': [1.5, 2.0, 2.5, 3.0],
    'outlier_weight_multiplier': [2.0, 3.0, 4.0, 5.0],
    'temporal_weight_range': [(0.2, 1.0), (0.3, 1.0), (0.4, 1.0)]
}

# 튜닝 순서
# 1. outlier_threshold 조정 (이상치 탐지 민감도)
# 2. outlier_weight_multiplier 조정 (이상치 강조 정도)
# 3. temporal_weight_range 미세 조정
```

**Priority 3 - Adaptive Weight**
```python
param_grid = {
    'base_loss_type': ['mse', 'mae', 'huber'],
    'adaptive_power': [1.0, 1.2, 1.5, 1.8, 2.0],
    'beta': [0.3, 0.5, 0.7],  # huber인 경우만
}

# 튜닝 순서
# 1. base_loss_type 선택
# 2. adaptive_power 조정 (적응형 강도)
# 3. beta 조정 (huber인 경우)
```

#### 3단계: 세부 튜닝

**시간 가중치 세부 조정**
```python
# 미래 강조 정도에 따른 옵션
temporal_options = {
    'weak_future_emphasis': (0.4, 1.0),     # 약한 미래 강조
    'medium_future_emphasis': (0.3, 1.0),   # 중간 미래 강조 (기본)
    'strong_future_emphasis': (0.2, 1.0),   # 강한 미래 강조
    'very_strong_emphasis': (0.1, 1.0),     # 매우 강한 미래 강조
}

# 지수적 가중치 옵션 (고급)
def exponential_weights(seq_len, decay_rate=0.95):
    """지수적으로 증가하는 시간 가중치"""
    weights = torch.exp(torch.linspace(0, -np.log(decay_rate), seq_len))
    return weights / weights.min()  # 최소값을 1로 정규화
```

### 🔍 성능 모니터링

#### 학습 중 모니터링
```python
class LossMonitor:
    """손실함수 성능 모니터링 클래스"""
    
    def __init__(self):
        self.history = {
            'total_loss': [],
            'outlier_loss': [],
            'rapid_change_loss': [],
            'future_loss': [],
            'normal_loss': []
        }
    
    def update(self, loss_fn, pred, target, step):
        """각 단계별 세부 손실 기록"""
        
        # 전체 손실
        total_loss = loss_fn(pred, target)
        self.history['total_loss'].append((step, total_loss.item()))
        
        # 구간별 손실 계산 및 기록
        outlier_mask = self._detect_outliers(target)
        rapid_mask = self._detect_rapid_changes(target)
        future_mask = self._get_future_mask(target)
        normal_mask = ~(outlier_mask | rapid_mask)
        
        # 각 구간별 손실 기록
        if outlier_mask.any():
            outlier_loss = F.mse_loss(pred[outlier_mask], target[outlier_mask])
            self.history['outlier_loss'].append((step, outlier_loss.item()))
        
        # ... 다른 구간들도 동일하게 처리
    
    def plot_progress(self):
        """학습 진행 상황 시각화"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (loss_type, values) in enumerate(self.history.items()):
            if values:
                steps, losses = zip(*values)
                ax = axes[i//3, i%3]
                ax.plot(steps, losses, label=loss_type)
                ax.set_title(f'{loss_type} Progress')
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.legend()
        
        plt.tight_layout()
        plt.show()
```

#### 검증 전략
```python
def cross_validate_loss_functions(X, y, loss_functions, cv_folds=5):
    """여러 손실함수에 대한 교차 검증"""
    
    results = {}
    
    for name, loss_fn in loss_functions.items():
        fold_results = []
        
        for fold in range(cv_folds):
            # 데이터 분할
            train_X, val_X, train_y, val_y = train_test_split(X, y, fold)
            
            # 모델 학습
            model = YourModel()
            optimizer = torch.optim.Adam(model.parameters())
            
            for epoch in range(num_epochs):
                pred = model(train_X)
                loss = loss_fn(pred, train_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 검증 성능
            with torch.no_grad():
                val_pred = model(val_X)
                val_metrics = evaluate_performance(loss_fn, val_pred, val_y)
                fold_results.append(val_metrics)
        
        # 폴드별 결과 집계
        results[name] = {
            metric: np.mean([fold[metric] for fold in fold_results])
            for metric in fold_results[0].keys()
        }
    
    return results
```

### 📈 고급 튜닝 기법

#### 1. **베이지안 최적화**
```python
from skopt import gp_minimize
from skopt.space import Real, Categorical

def objective(params):
    """베이지안 최적화를 위한 목적 함수"""
    
    # 파라미터 언팩
    beta, gradient_scale, temp_start = params
    
    # 손실함수 설정
    loss_fn = Priority1_HuberMultiCriteriaLoss(
        beta=beta,
        gradient_weight_scale=gradient_scale,
        temporal_weight_range=(temp_start, 1.0)
    )
    
    # 모델 학습 및 평가
    val_loss = train_and_evaluate(loss_fn)
    
    return val_loss

# 탐색 공간 정의
space = [
    Real(0.1, 1.0, name='beta'),
    Real(1.0, 5.0, name='gradient_scale'),
    Real(0.1, 0.5, name='temp_start')
]

# 최적화 실행
result = gp_minimize(objective, space, n_calls=50)
best_params = result.x
```

#### 2. **동적 가중치 스케줄링**
```python
class DynamicWeightScheduler:
    """학습 과정에서 가중치를 동적으로 조정"""
    
    def __init__(self, loss_fn, schedule_type='cosine'):
        self.loss_fn = loss_fn
        self.schedule_type = schedule_type
        self.initial_params = self._get_current_params()
    
    def step(self, epoch, total_epochs):
        """에포크에 따른 파라미터 조정"""
        
        if self.schedule_type == 'cosine':
            # 코사인 스케줄링
            factor = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        elif self.schedule_type == 'linear':
            # 선형 감소
            factor = 1 - epoch / total_epochs
        
        # 파라미터 업데이트
        if hasattr(self.loss_fn, 'gradient_weight_scale'):
            self.loss_fn.gradient_weight_scale = self.initial_params['gradient_scale'] * (1 + factor)
        
        if hasattr(self.loss_fn, 'outlier_weight_multiplier'):
            self.loss_fn.outlier_weight_multiplier = self.initial_params['outlier_multiplier'] * (1 + factor)

# 사용 예시
scheduler = DynamicWeightScheduler(loss_fn, 'cosine')

for epoch in range(num_epochs):
    scheduler.step(epoch, num_epochs)
    
    # 일반적인 학습 루프
    for batch in dataloader:
        pred = model(batch.x)
        loss = loss_fn(pred, batch.y)
        # ... 역전파 등
```

#### 3. **앙상블 손실함수**
```python
class EnsembleLoss(nn.Module):
    """여러 손실함수의 가중 평균"""
    
    def __init__(self, loss_functions, weights=None):
        super().__init__()
        self.loss_functions = loss_functions
        self.weights = weights or [1.0] * len(loss_functions)
        
    def forward(self, pred, target):
        total_loss = 0
        
        for loss_fn, weight in zip(self.loss_functions, self.weights):
            loss = loss_fn(pred, target)
            total_loss += weight * loss
        
        return total_loss / sum(self.weights)

# 사용 예시
ensemble_loss = EnsembleLoss([
    Priority1_HuberMultiCriteriaLoss(beta=0.3),
    Priority2_MAEOutlierFocusedLoss(outlier_threshold=2.0)
], weights=[0.7, 0.3])
```

### 🚨 주의사항 및 팁

#### 일반적인 실수들

1. **과도한 가중치 설정**
   ```python
   # 잘못된 예시
   loss_fn = Priority1_HuberMultiCriteriaLoss(
       gradient_weight_scale=10.0,  # 너무 큰 값
       temporal_weight_range=(0.01, 1.0)  # 너무 극단적
   )
   
   # 올바른 예시
   loss_fn = Priority1_HuberMultiCriteriaLoss(
       gradient_weight_scale=2.0,   # 적절한 값
       temporal_weight_range=(0.3, 1.0)  # 균형잡힌 범위
   )
   ```

2. **가중치 정규화 무시**
   - 대부분의 손실함수에는 내장 정규화가 있지만, 극단적인 파라미터 설정 시 주의
   - 가중치의 범위를 모니터링하고 clipping 확인

3. **검증 데이터 편향**
   - 이상치와 급변이 포함된 대표적인 검증 세트 구성
   - 시간적 분할 시 미래 시점 포함 확인

#### 성능 최적화 팁

1. **배치 크기 조정**
   ```python
   # 가중치 계산으로 인한 메모리 사용량 증가 고려
   # 기존 배치 크기의 70-80% 권장
   ```

2. **그래디언트 클리핑**
   ```python
   # 가중치로 인한 그래디언트 폭발 방지
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **학습률 조정**
   ```python
   # 가중치 손실함수는 일반적으로 더 낮은 학습률 필요
   optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 기존의 50-70%
   ```

### 📚 추가 자료 및 참고문헌

#### 관련 논문
- Huber Loss: "Robust Estimation of a Location Parameter" (Huber, 1964)
- Quantile Regression: "Regression Quantiles" (Koenker & Bassett, 1978)
- Multi-task Learning: "A Survey on Multi-Task Learning" (Zhang & Yang, 2017)

#### 구현 참고자료
- PyTorch 공식 문서: Loss Functions
- Scikit-learn: Robust Regression
- TensorFlow Probability: Quantile Regression

#### 응용 분야별 가이드
- **금융 시계열**: Priority 2 (MAE Outlier-focused) 추천
- **IoT 센서 데이터**: Priority 1 (Huber Multi-criteria) 추천
- **의료 신호**: Priority 5 (Quantile Loss) 추천
- **제조업 모니터링**: Priority 4 (Gradient-based) 추천

---

## 결론

이 6가지 우선순위별 손실함수는 이상치와 급격한 변화가 중요하고, 미래 시점에 더 높은 가중치를 주는 시계열 회귀 작업을 위해 특별히 설계되었습니다. 

**선택 가이드라인:**
- 🚀 **빠른 시작**: Priority 1 (Huber Multi-criteria)
- 🎯 **이상치 집중**: Priority 2 (MAE Outlier-focused)  
- 🤖 **자동화**: Priority 3 (Adaptive Weight)
- ⚡ **급변 집중**: Priority 4 (Gradient-based)
- 📊 **불확실성**: Priority 5 (Quantile)
- 🔬 **포괄적**: Priority 6 (Multi-task)

각 손실함수는 특정 상황에 최적화되어 있으므로, 데이터의 특성과 요구사항에 맞게 선택하고 하이퍼파라미터를 조정하여 사용하시기 바랍니다.