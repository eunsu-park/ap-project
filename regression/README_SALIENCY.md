# Multi-Channel Analysis 사용 가이드

## 🎯 새로운 기능

**모든 플롯을 채널별로 자동 생성!**

이제 193Å, 211Å, magnetogram 각 파장에 대해 모든 분석이 자동으로 생성됩니다.

## 🚀 빠른 시작

### 1. 스크립트 실행

```bash
python example_saliency_fixed.py --config-name saliency
```

### 2. 생성되는 구조

```
saliency_outputs/
└── batch_0000/
    ├── channel_0_193A/
    │   ├── grad_cam_193Å.png
    │   ├── full_sequence_193Å.png
    │   └── comprehensive_193Å.png
    ├── channel_1_211A/
    │   ├── grad_cam_211Å.png
    │   ├── full_sequence_211Å.png
    │   └── comprehensive_211Å.png
    ├── channel_2_304A/
    │   ├── grad_cam_magnetogram.png
    │   ├── full_sequence_magnetogram.png
    │   └── comprehensive_magnetogram.png
    ├── channel_comparison.png         ← 모든 채널 비교!
    ├── temporal_importance_all_channels.png
    └── channel_importance.npz
```

## 📊 각 채널별 생성 파일

### 1. `grad_cam_XXX.png`
- 3개 시점(처음/중간/끝)의 Grad-CAM
- 원본 + Saliency + Overlay

### 2. `full_sequence_XXX.png`
- 전체 시퀀스를 한 눈에
- 4개 패널: 원본/Grad-CAM/Temporal/Prediction

### 3. `comprehensive_XXX.png`
- 종합 분석
- Grad-CAM + IG + 통계

## 🔬 채널 비교 분석

### `channel_comparison.png`

**모든 채널을 한 화면에!**

```
          t=0              t=important        t=last
193Å  [원본][CAM]      [원본][CAM]      [원본][CAM]
211Å  [원본][CAM]      [원본][CAM]      [원본][CAM]
magnetogram  [원본][CAM]      [원본][CAM]      [원본][CAM]
```

**활용:**
- 어떤 파장이 가장 민감하게 반응하는가?
- 파장별 주목 영역의 차이
- 시간에 따른 파장별 역할 변화

### `channel_importance.npz`

**수치 데이터:**
```python
import numpy as np

data = np.load('channel_importance.npz')
importance = data['channel_importance']
names = data['channel_names']

print(f"193Å: {importance[0]:.3f}")
print(f"211Å: {importance[1]:.3f}")
print(f"magnetogram: {importance[2]:.3f}")
```

## 📈 과학적 해석 예시

### 예시 1: 193Å이 가장 중요

```
Channel Importance:
  193Å: 1.000  ████████████
  211Å: 0.650  ████████
  magnetogram: 0.420  █████

→ 해석: 코로나 플라즈마 (1-2 MK)가 가장 중요
→ 의미: 고온 활동 영역이 지자기 활동의 주 원인
```

### 예시 2: magnetogram이 두드러짐

```
Channel Importance:
  193Å: 0.450  █████
  211Å: 0.520  ██████
  magnetogram: 1.000  ████████████

→ 해석: 크로모스피어/천이 영역이 중요
→ 의미: 저온 구조물(필라멘트 등)이 영향
```

### 예시 3: 고른 분포

```
Channel Importance:
  193Å: 0.950  ███████████
  211Å: 1.000  ████████████
  magnetogram: 0.880  ██████████

→ 해석: 모든 온도 범위가 골고루 기여
→ 의미: 복잡한 다층 구조의 영향
```

## 🔍 채널별 차이 분석

### Spatial Patterns (Grad-CAM)

**193Å vs 211Å:**
```
193Å: 작고 밝은 영역 집중
     → 활동 영역 코어
     
211Å: 더 넓은 영역
     → 활동 영역 + 주변 루프

→ 211Å가 더 넓은 컨텍스트 포착
```

**211Å vs magnetogram:**
```
211Å: 활동 영역 중심
     → 코로나 루프 구조
     
magnetogram: 더 큰 스케일 구조
     → 필라멘트, 거대 루프

→ 스케일이 다른 현상 포착
```

### Temporal Patterns

**Time-dependent importance:**
```
193Å: t=5-10에 피크
211Å: t=8-12에 피크
magnetogram: t=3-8에 피크

→ 각 파장이 다른 시점에 정보 제공
→ 온도별 진화 타임스케일 차이
```

## 💡 고급 활용

### 1. 파장 조합 실험

```python
# 특정 파장만 사용
image_input_193_only = image_input.clone()
image_input_193_only[:, [1, 2], :, :, :] = 0  # 211, 304 제거

# 예측 비교
pred_all = model(solar_wind, image_input)
pred_193_only = model(solar_wind, image_input_193_only)

print(f"Difference: {(pred_all - pred_193_only).abs().mean()}")
```

### 2. 시간 윈도우별 분석

```python
# 초기 vs 후기
for start_t in [0, 14]:
    end_t = start_t + 14
    
    partial_images = image_input[:, :, start_t:end_t, :, :]
    
    extractor.visualize_all_channels_analysis(
        solar_wind, partial_images,
        output_dir=f'analysis_t{start_t}_{end_t}'
    )
```

### 3. 이벤트별 비교

```python
# 플레어 있는 케이스 vs 없는 케이스
flare_samples = [0, 3, 7]
quiet_samples = [1, 2, 5]

for samples, name in [(flare_samples, 'flare'), (quiet_samples, 'quiet')]:
    # 분석 및 비교
    ...
```

## 🎨 커스터마이징

### 채널 이름 변경

```python
# example_saliency_fixed.py에서
channel_names = ['Cool', 'Hot', 'Chromosphere']  # 원하는 이름

extractor.visualize_all_channels_analysis(
    ...,
    channel_names=channel_names
)
```

### 다른 파장 조합

```python
# 다른 SDO 파장 사용 시
channel_names = ['171Å', '193Å', '211Å']

# 또는 다른 관측소
channel_names = ['STEREO-A', 'SDO', 'STEREO-B']
```

## 📝 체크리스트

실행 전 확인사항:

- [ ] 모델이 제대로 학습되었는가?
- [ ] 이미지 채널 수가 맞는가? (3개)
- [ ] 채널 이름이 정확한가?
- [ ] 출력 디렉토리에 쓰기 권한이 있는가?
- [ ] 충분한 디스크 공간이 있는가? (배치당 ~10-20MB)

## 🐛 문제 해결

### 에러: "index out of range"
```
→ 채널 수 확인: image_input.shape[1]
→ channel_names 길이와 일치해야 함
```

### 메모리 부족
```
→ MAX_BATCHES 줄이기 (기본 3)
→ 한 번에 1개 채널씩 분석
```

### 플롯이 비어있음
```
→ 모델이 이미지를 사용하는지 확인
→ verify_image_usage.py 실행
```

---

**업데이트**: 2025-01-11  
**버전**: 3.0 - Multi-Channel Support