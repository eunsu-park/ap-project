# Saliency 분석 결과 및 개선 방안

## 📊 분석 결과 요약

### ✅ 정상적인 부분

1. **채널 중요도가 물리적으로 타당함**
   ```
   193Å:        1.0000  (고온 코로나)
   211Å:        0.9952  (활동 영역)
   magnetogram: 0.8477  (자기장)
   ```
   - 고온 코로나가 지자기 활동 예측에 가장 중요
   - 모든 채널이 적절히 기여 (0.85+)

2. **Grad-CAM이 특정 영역을 집중적으로 주목**
   - 왼쪽 상단 대형 구조
   - 오른쪽 여러 활동 영역
   - 하단 중앙 구조

### 🔴 심각한 문제점

#### 문제 1: 시간 정보를 제대로 사용하지 않음

**증거:**

1. **Grad-CAM이 모든 시점에서 동일**
   ```
   t=0, t=14, t=27의 Grad-CAM이 완전히 동일
   원본 이미지는 명백히 다른데 saliency는 불변
   ```

2. **중간 시점을 거의 무시**
   ```
   Temporal Importance:
   t=0-3:    0.2-0.4  (약간 사용)
   t=4-14:   0.0-0.2  (거의 무시!) ← 12시간 분량
   t=15-19:  0.7-0.75 (사용)
   t=20-22:  0.1      (무시)
   t=23-27:  0.8-1.0  (주로 사용)
   ```

3. **예측값이 거의 일정**
   ```
   모든 시점: -0.2 ± 0.01
   ```

**의미:**
- ConvLSTM이 **시간 정보를 통합하지 못함**
- 마치 **단일 이미지 분류기처럼** 작동
- 초기 + 최근만 보고 중간 진화 과정 무시

#### 문제 2: 공간적 패턴이 고정됨

**증거:**
- 원본이 회전하고 변해도 saliency는 불변
- 특정 위치만 항상 주목

**의미:**
- 위치 기반 휴리스틱 학습
- "왼쪽 상단이 항상 중요하다" 같은 단순 규칙
- 실제 태양 물리 현상과 무관

#### 문제 3: 모델 예측이 입력과 독립적

**증거:**
- 모든 입력에 대해 -0.2 출력
- verify_image_usage 결과와 일치

**의미:**
- 학습 실패
- 평균값만 출력

---

## 🔍 근본 원인 진단

### 가설 1: ConvLSTM Hidden State 미전달

**가능성: 70%**

```python
# 예상되는 문제
for t in range(seq_len):
    frame = images[:, :, t, :, :]
    output, hidden = convlstm(frame, hidden)
    # ← hidden이 제대로 업데이트/전달되지 않음
```

**결과:**
- 각 프레임을 독립적으로 처리
- 시간적 맥락 손실

### 가설 2: Feature Magnitude Imbalance

**가능성: 80%**

```python
# 예상되는 상황
transformer_features: mean = 10.0
convlstm_features:    mean = 0.01  ← 100배 작음!

# Fusion
fused = transformer_feat + convlstm_feat
      ≈ transformer_feat  (convlstm 무시됨)
```

**결과:**
- Cross-modal fusion에서 ConvLSTM 무시
- Transformer만으로 예측

### 가설 3: Gradient Vanishing

**가능성: 50%**

```python
# Long sequence (28 steps) → vanishing gradients
# 초기 시점에 gradient가 도달하지 못함
```

**결과:**
- ConvLSTM이 학습되지 않음
- 초기 가중치 상태 유지

---

## 💡 해결 방안

### 🎯 Phase 1: 진단 (즉시 실행)

#### Step 1: ConvLSTM 작동 확인

```bash
python diagnose_convlstm.py --config-name saliency
```

**확인 사항:**
- [ ] Hidden state가 시간에 따라 변하는가?
- [ ] 각 시점의 output이 다른가?
- [ ] Gradient가 역전파되는가?

#### Step 2: Fusion 가중치 확인

```bash
python diagnose_fusion.py --config-name saliency
```

**확인 사항:**
- [ ] Transformer와 ConvLSTM의 feature 크기 비율
- [ ] Fusion에서 각 modality의 기여도
- [ ] Ablation test 결과

---

### 🔧 Phase 2: 빠른 수정 (1-2일)

#### 수정 1: Feature Normalization 추가

**문제:** ConvLSTM features가 너무 작음

**해결:**
```python
class CrossModalFusion(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 추가!
        self.transformer_norm = nn.LayerNorm(transformer_dim)
        self.convlstm_norm = nn.LayerNorm(convlstm_dim)
        
    def forward(self, transformer_feat, convlstm_feat):
        # Normalize!
        transformer_feat = self.transformer_norm(transformer_feat)
        convlstm_feat = self.convlstm_norm(convlstm_feat)
        
        # Then fuse
        fused = self.fusion_layer(torch.cat([transformer_feat, convlstm_feat], dim=-1))
        return fused
```

**기대 효과:**
- 두 modality의 크기 균형
- ConvLSTM의 기여도 증가

#### 수정 2: ConvLSTM Learning Rate 증가

**문제:** ConvLSTM이 학습되지 않음

**해결:**
```python
# Separate optimizer groups
optimizer = optim.AdamW([
    {'params': model.transformer_model.parameters(), 'lr': 1e-4},
    {'params': model.convlstm_model.parameters(), 'lr': 5e-4},  # 5배!
    {'params': model.cross_modal_fusion.parameters(), 'lr': 1e-4},
    {'params': model.regression_head.parameters(), 'lr': 1e-4}
])
```

**기대 효과:**
- ConvLSTM 가중치 빠른 업데이트
- Transformer에 catch up

#### 수정 3: Gradient Clipping 조정

**문제:** Gradient vanishing/exploding

**해결:**
```python
# 더 관대한 clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 1.0 → 5.0

# 또는 per-module clipping
torch.nn.utils.clip_grad_norm_(model.convlstm_model.parameters(), max_norm=10.0)
```

---

### 🏗️ Phase 3: 구조 개선 (1주)

#### 개선 1: Auxiliary Loss 추가

**목적:** ConvLSTM이 의미 있는 features를 학습하도록 강제

```python
class MultiModalModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing modules ...
        
        # Auxiliary head for ConvLSTM
        self.convlstm_auxiliary_head = nn.Sequential(
            nn.Linear(convlstm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, solar_wind, images):
        transformer_feat = self.transformer_model(solar_wind)
        convlstm_feat = self.convlstm_model(images)
        
        # Auxiliary prediction (only during training)
        if self.training:
            aux_pred = self.convlstm_auxiliary_head(convlstm_feat)
        
        # Main prediction
        fused = self.cross_modal_fusion(transformer_feat, convlstm_feat)
        main_pred = self.regression_head(fused)
        
        if self.training:
            return main_pred, aux_pred
        else:
            return main_pred

# Loss
main_loss = criterion(main_pred, target)
aux_loss = criterion(aux_pred, target)
total_loss = main_loss + 0.3 * aux_loss  # 30% weight on auxiliary
```

**기대 효과:**
- ConvLSTM이 직접 예측을 수행하도록 학습
- 의미 있는 visual features 추출

#### 개선 2: Attention-based Fusion

**목적:** 동적으로 modality 가중치 결정

```python
class AttentionFusion(nn.Module):
    def __init__(self, transformer_dim, convlstm_dim, hidden_dim):
        super().__init__()
        
        # Query, Key, Value projections
        self.q_transformer = nn.Linear(transformer_dim, hidden_dim)
        self.k_convlstm = nn.Linear(convlstm_dim, hidden_dim)
        self.v_transformer = nn.Linear(transformer_dim, hidden_dim)
        self.v_convlstm = nn.Linear(convlstm_dim, hidden_dim)
        
        self.scale = hidden_dim ** -0.5
        self.output = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, transformer_feat, convlstm_feat):
        # Cross-attention
        q_t = self.q_transformer(transformer_feat)
        k_c = self.k_convlstm(convlstm_feat)
        
        # Attention weights
        attn = torch.softmax(q_t @ k_c.T * self.scale, dim=-1)
        
        # Weighted combination
        v_t = self.v_transformer(transformer_feat)
        v_c = self.v_convlstm(convlstm_feat)
        
        fused_t = attn @ v_c
        fused_c = attn.T @ v_t
        
        # Concatenate and project
        fused = self.output(torch.cat([fused_t, fused_c], dim=-1))
        
        return fused
```

**기대 효과:**
- 입력에 따라 adaptive하게 fusion
- 각 modality의 기여도 관찰 가능

#### 개선 3: Temporal Attention in ConvLSTM

**목적:** 시간 정보를 더 효과적으로 활용

```python
class ConvLSTMWithAttention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.convlstm = ConvLSTM(...)
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4
        )
        
    def forward(self, images):
        # images: (B, C, T, H, W)
        B, C, T, H, W = images.shape
        
        # ConvLSTM
        features = []
        for t in range(T):
            feat = self.convlstm(images[:, :, t, :, :])  # (B, hidden_dim)
            features.append(feat)
        
        features = torch.stack(features, dim=0)  # (T, B, hidden_dim)
        
        # Temporal attention
        attn_out, attn_weights = self.temporal_attention(
            features, features, features
        )
        
        # Aggregate
        output = attn_out.mean(dim=0)  # (B, hidden_dim)
        
        return output
```

**기대 효과:**
- 중요한 시점에 집중
- 전체 시퀀스 활용

---

### 🧪 Phase 4: 재학습 전략 (2주)

#### 전략 1: Progressive Training

**Stage 1 (Epoch 1-20): ConvLSTM만 학습**
```python
# Freeze other modules
for param in model.transformer_model.parameters():
    param.requires_grad = False
for param in model.cross_modal_fusion.parameters():
    param.requires_grad = False

# Train only ConvLSTM
optimizer = optim.AdamW(model.convlstm_model.parameters(), lr=1e-3)
```

**Stage 2 (Epoch 21-50): Fusion 학습**
```python
# Unfreeze fusion
for param in model.cross_modal_fusion.parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {'params': model.convlstm_model.parameters(), 'lr': 5e-4},
    {'params': model.cross_modal_fusion.parameters(), 'lr': 1e-3}
])
```

**Stage 3 (Epoch 51-100): Fine-tuning**
```python
# Unfreeze all
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
```

#### 전략 2: Curriculum Learning

**Easy → Hard**
```python
# Stage 1: Short sequences (T=7)
dataloader = create_dataloader(config, seq_len=7)

# Stage 2: Medium sequences (T=14)
dataloader = create_dataloader(config, seq_len=14)

# Stage 3: Full sequences (T=28)
dataloader = create_dataloader(config, seq_len=28)
```

**기대 효과:**
- LSTM이 gradients를 더 잘 전파
- 점진적 학습

#### 전략 3: Data Augmentation

**시간적 augmentation:**
```python
def temporal_augmentation(images):
    # Random temporal shift
    shift = random.randint(-2, 2)
    images = torch.roll(images, shifts=shift, dims=2)
    
    # Random temporal flip
    if random.random() > 0.5:
        images = torch.flip(images, dims=[2])
    
    # Random frame dropout
    if random.random() > 0.5:
        mask = torch.rand(images.shape[2]) > 0.1
        images = images[:, :, mask, :, :]
    
    return images
```

**기대 효과:**
- 더 robust한 시간 정보 학습
- Overfitting 방지

---

## 📋 실행 체크리스트

### Week 1: 진단

- [ ] Day 1: `diagnose_convlstm.py` 실행 및 결과 분석
- [ ] Day 2: `diagnose_fusion.py` 실행 및 결과 분석
- [ ] Day 3: 문제점 정리 및 해결 방안 우선순위 결정

### Week 2: 빠른 수정

- [ ] Day 1-2: Feature normalization 추가
- [ ] Day 3-4: Learning rate 조정 및 재학습 시작
- [ ] Day 5: 중간 결과 확인 (saliency 재생성)

### Week 3: 구조 개선

- [ ] Day 1-2: Auxiliary loss 구현
- [ ] Day 3-4: Attention fusion 구현
- [ ] Day 5: 성능 비교

### Week 4: 재학습 및 평가

- [ ] Day 1-3: Progressive training
- [ ] Day 4: 최종 saliency 분석
- [ ] Day 5: 논문 figure 작성

---

## 🎯 예상 결과

### 성공 시 (Good Case)

**Temporal Importance:**
```
부드러운 곡선, 여러 시점에 분산
모든 시점이 일정 수준 이상 기여
```

**Grad-CAM:**
```
시간에 따라 변화하는 패턴
원본 이미지의 구조 변화를 반영
```

**예측:**
```
입력에 따라 변하는 예측값
더 높은 정확도
```

### 중간 성과 (Moderate)

**Temporal Importance:**
```
일부 시점에 집중
하지만 여전히 의미 있는 분포
```

**Grad-CAM:**
```
약간의 시간적 변화
주요 구조는 추적
```

### 실패 시 (Bad Case)

**여전히 동일한 패턴:**
```
→ 더 근본적인 문제 (아키텍처 재설계 필요)
```

---

## 📚 참고 자료

**Papers:**
1. "Attention is All You Need" - Multi-head attention
2. "ConvLSTM" - Spatial-temporal modeling
3. "Grad-CAM++" - Improved saliency
4. "Progressive Neural Networks" - Transfer learning

**Debugging:**
1. Hidden state visualization
2. Gradient magnitude tracking
3. Feature distribution analysis
4. Ablation studies

---

**작성일**: 2025-01-12  
**상태**: 진단 완료, 개선 대기  
**우선순위**: 🔴 High (모델 성능에 critical)
