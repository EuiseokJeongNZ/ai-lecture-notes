# 강의_9기_AI개론_6차시_정리  
## 손실함수 Loss 설계 · MSE/MAE/Huber · BCEWithLogits · CrossEntropy · Label Smoothing · Class Imbalance

## 1. 개요

이번 6차시는 모델 학습에서 손실함수를 어떻게 설계하고 선택하는지 정리하는 강의다.

이전 강의까지는 PyTorch 학습 루프의 전체 구조를 봤다.

```text
Forward → Loss → Backward → Update
```

이번 강의는 이 중에서 **Loss**에 집중한다.

손실함수는 모델의 예측값과 실제 정답 사이의 차이를 수치화하는 도구다.  
단순히 틀린 정도를 알려주는 숫자가 아니라, Optimizer가 어느 방향으로 파라미터를 수정해야 하는지 알려주는 학습의 기준이다.

강의 흐름은 다음이다.

```text
회귀 손실함수
→ MSE, MAE, Huber

이진 분류 손실함수
→ BCE, BCEWithLogitsLoss

다중 분류 손실함수
→ CrossEntropyLoss

일반화와 과신 방지
→ Label Smoothing

클래스 불균형 대응
→ Weighted CE, pos_weight, Focal Loss

학습 안정성
→ Loss Surface, 좋은 손실 곡면 만들기
```

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| Loss Function | 예측값과 정답의 차이를 수치화하는 함수 |
| Cost Function | Loss Function과 거의 같은 의미로 쓰이는 경우가 많다 |
| MSE | 오차를 제곱해 평균낸 회귀 손실 |
| MAE | 오차의 절댓값을 평균낸 회귀 손실 |
| Huber Loss | 작은 오차에는 MSE, 큰 오차에는 MAE처럼 동작하는 절충형 손실 |
| BCE | 이진 분류에서 확률값과 정답 차이를 보는 손실 |
| BCEWithLogitsLoss | Sigmoid와 BCE를 내부에서 안정적으로 처리하는 손실 |
| CrossEntropyLoss | 다중 분류에서 class별 raw logits와 class index를 비교하는 손실 |
| Label Smoothing | one-hot label을 부드럽게 만들어 모델 과신을 줄이는 기법 |
| Class Imbalance | 특정 class가 지나치게 많은 데이터 불균형 상황 |
| Weighted CE | class별 손실에 가중치를 주는 방식 |
| pos_weight | BCEWithLogitsLoss에서 양성 class에 주는 가중치 |
| Focal Loss | 쉬운 샘플보다 어려운 샘플에 집중하는 손실 |
| Loss Surface | 파라미터 공간에서 손실값이 변화하는 지형 |
| Gradient Clipping | gradient 폭발을 막기 위해 gradient 크기를 제한하는 방법 |
| Warm-up | 초반 learning rate를 천천히 키우는 전략 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_6차시_00_실전프로젝트_tobecontinued.ipynb` | 실전 프로젝트 예고 및 손실함수 적용 흐름 |
| `강의_9기_AI개론_6차시_01_MSE vs BCEWithLogits 비교와 Label Smoothing_ok.ipynb` | 이진 분류에서 MSE와 BCEWithLogits 비교, MNIST 기반 Label Smoothing/Weighted CE 예시 |
| `강의_9기_AI개론_6차시_02_회귀손실함수 비교_ok.ipynb` | MSE, MAE, Huber 비교, 이상치 민감도, Huber delta 영향 |
| `강의_9기_AI개론_6차시_03_라벨스무딩.ipynb` | Label Smoothing 직접 구현, smoothing alpha 비교, confidence 비교 |
| `강의_9기_AI개론_6차시_04_클래스불균형대응과 손실곡면.ipynb` | 클래스 불균형, pos_weight, Focal Loss, Loss Surface 개념 |
| `06. 손실함수(Loss) 설계 (1).srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_6차시_(비용함수정의).pdf` | 손실함수 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. 손실함수의 목적 정리
3. 간단한 정답/예측 Tensor 생성
4. MSE 직접 계산과 nn.MSELoss 비교
5. MAE 직접 계산과 nn.L1Loss 비교
6. Huber Loss 계산
7. 이상치가 있을 때 MSE, MAE, Huber 반응 비교
8. RMSE 계산
9. make_regression으로 회귀 데이터 생성
10. 이상치 추가
11. RegressionModel 정의
12. MSE, MAE, Huber로 각각 모델 학습
13. 학습 곡선과 Test MAE 비교
14. Huber Loss의 delta 값별 그래프 확인
15. BCE와 BCEWithLogitsLoss 차이 정리
16. 이진 분류에서 MSE와 BCEWithLogitsLoss 비교
17. CrossEntropyLoss 입력 형식 정리
18. logits.argmax(dim=1)로 다중 분류 예측 class 확인
19. Label Smoothing 공식 적용
20. CrossEntropyLoss(label_smoothing=0.1) 사용
21. LabelSmoothingCrossEntropy 직접 구현
22. smoothing alpha별 학습 결과 비교
23. 클래스 불균형 데이터 생성
24. pos_weight 계산
25. 일반 BCEWithLogitsLoss와 pos_weight 적용 결과 비교
26. Binary Focal Loss 구현
27. Weighted CrossEntropyLoss 사용
28. Loss Surface contour plot 출력
29. 좋은 손실 곡면과 안정적 학습 전략 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 MSE

```python
mse_loss = nn.MSELoss()
loss = mse_loss(y_pred, y_true)
```

예측값과 정답의 차이를 제곱한 뒤 평균낸다.

큰 오차에 큰 패널티를 주므로 이상치에 민감하다.

---

### 5.2 MAE

```python
mae_loss = nn.L1Loss()
loss = mae_loss(y_pred, y_true)
```

예측값과 정답 차이의 절댓값을 평균낸다.

PyTorch에서는 MAE를 `L1Loss`라고 부른다.

---

### 5.3 Huber Loss

```python
huber_loss = nn.HuberLoss(delta=1.0)
loss = huber_loss(y_pred, y_true)
```

작은 오차에는 MSE처럼, 큰 오차에는 MAE처럼 동작한다.

`delta`는 두 방식이 전환되는 경계값이다.

---

### 5.4 이상치 비교

```python
y_pred_outlier = torch.tensor([11.0, 19.0, 31.0, 100.0, 51.0])
```

예측값 하나를 크게 틀리게 만들어 MSE, MAE, Huber가 얼마나 다르게 반응하는지 확인한다.

MSE는 큰 오차 하나에 매우 민감하다.

---

### 5.5 RMSE

```python
rmse = np.sqrt(mse)
```

MSE에 제곱근을 씌워 원래 단위와 비슷하게 해석한다.

---

### 5.6 회귀 모델

```python
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
```

10개 feature를 입력받아 숫자 하나를 예측하는 회귀 모델이다.

---

### 5.7 손실함수별 학습 비교

```python
loss_functions = {
    "MSE": nn.MSELoss(),
    "MAE": nn.L1Loss(),
    "Huber": nn.HuberLoss(delta=1.0)
}
```

같은 모델 구조를 손실함수만 바꿔 학습한다.

손실값의 단위가 서로 다르므로 공통 지표인 Test MAE로 비교하는 것이 좋다.

---

### 5.8 BCE

```python
bce = nn.BCELoss()
loss = bce(prob, target)
```

이진 분류에서 확률값과 정답을 비교한다.

`prob`는 반드시 0~1 사이 확률값이어야 한다.

---

### 5.9 BCEWithLogitsLoss

```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, target)
```

Sigmoid 전 raw logits를 직접 받는다.

내부에서 Sigmoid와 BCE를 안정적으로 처리하므로 실무에서 더 권장된다.

---

### 5.10 CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
```

다중 분류에 사용한다.

`logits`는 `[batch, class_count]` 형태이고, `target`은 `[batch]` 형태의 class index다.

Softmax를 먼저 적용하면 안 된다.

---

### 5.11 argmax

```python
pred = logits.argmax(dim=1)
```

class 방향에서 가장 큰 점수를 가진 위치를 예측 class로 선택한다.

다중 분류에서 자주 쓰는 패턴이다.

---

### 5.12 Label Smoothing 공식

```python
smooth_label = original_label * (1 - alpha) + alpha / num_classes
```

정답 class에 1을 몰아주지 않고 전체 class에 조금씩 확률을 나눠준다.

모델의 과신을 줄이는 효과가 있다.

---

### 5.13 PyTorch Label Smoothing

```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

PyTorch 내장 방식으로 Label Smoothing을 적용한다.

`alpha=0.1`이 대표적인 표준값이다.

---

### 5.14 Label Smoothing 직접 구현

```python
log_probs = torch.log_softmax(pred, dim=1)
true_dist.scatter_(1, target.unsqueeze(1), confidence)
loss = (-true_dist * log_probs).sum(dim=1).mean()
```

log softmax와 smoothed target distribution을 이용해 직접 Cross Entropy를 계산한다.

---

### 5.15 pos_weight

```python
pos_weight = torch.tensor([negative_count / positive_count])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

이진 불균형 분류에서 양성 class에 더 큰 가중치를 준다.

소수 class를 틀렸을 때 더 큰 손실을 부여한다.

---

### 5.16 Focal Loss

```python
focal_weight = alpha * (1 - pt) ** gamma
loss = focal_weight * bce_loss
```

이미 잘 맞히는 쉬운 샘플의 손실은 줄이고, 어려운 샘플에 더 집중한다.

객체 탐지나 극심한 불균형 문제에서 자주 언급된다.

---

### 5.17 Weighted CrossEntropyLoss

```python
weights = torch.tensor([1.0, 1.5, 3.0])
criterion = nn.CrossEntropyLoss(weight=weights)
```

다중 분류에서 class별 손실 가중치를 다르게 줄 수 있다.

---

### 5.18 Loss Surface

```python
plt.contourf(W_grid, B_grid, Z, levels=40)
```

파라미터 공간에서 Loss가 어떻게 변하는지 등고선으로 본다.

낮은 계곡을 찾는 것이 학습의 목표다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `loss` | 손실 | 예측과 정답의 차이를 수치화한다 |
| `criterion` | 손실함수 객체 | `criterion(pred, target)` |
| `MSE` | 평균 제곱 오차 | `nn.MSELoss()` |
| `MAE` | 평균 절대 오차 | `nn.L1Loss()` |
| `Huber` | MSE와 MAE 절충 | `nn.HuberLoss(delta=1.0)` |
| `BCE` | 이진 교차 엔트로피 | `nn.BCELoss()` |
| `BCEWithLogitsLoss` | 안정적인 BCE | raw logits를 직접 입력한다 |
| `CrossEntropyLoss` | 다중 분류 손실 | raw logits와 class index를 입력한다 |
| `logits` | Sigmoid/Softmax 전 점수 | 모델 마지막 출력 |
| `target` | 정답 | 회귀값 또는 class index |
| `label_smoothing` | 라벨 스무딩 | `CrossEntropyLoss(label_smoothing=0.1)` |
| `alpha` | smoothing 또는 focal 계수 | 보정 정도를 조절한다 |
| `gamma` | focal 조절 계수 | 쉬운 샘플을 덜 보게 만든다 |
| `pos_weight` | 양성 class 가중치 | `BCEWithLogitsLoss(pos_weight=...)` |
| `weight` | class별 가중치 | `CrossEntropyLoss(weight=...)` |
| `argmax` | 가장 큰 class 선택 | `logits.argmax(dim=1)` |
| `softmax` | 다중 class 확률 변환 | `torch.softmax(logits, dim=1)` |
| `sigmoid` | 이진 확률 변환 | `torch.sigmoid(logits)` |
| `reduction` | loss 집계 방식 | `mean`, `sum`, `none` |
| `Loss Surface` | 손실 곡면 | 파라미터별 손실 지형 |

---

## 7. 그래프/출력 결과 해석

### 7.1 회귀 손실함수 학습 곡선

MSE, MAE, Huber는 손실값의 단위가 다르므로 절대값만 비교하면 안 된다.

각 손실함수 안에서 loss가 줄어드는 흐름을 확인한다.

### 7.2 Test MAE 비교

공통 지표로 Test MAE를 사용하면 손실함수별 최종 성능을 비교하기 쉽다.

이상치가 있는 데이터에서는 MAE나 Huber가 더 안정적일 수 있다.

### 7.3 Huber Delta 그래프

`delta`가 작으면 MAE에 가까워지고, `delta`가 크면 MSE에 가까워진다.

### 7.4 Label Smoothing 학습 곡선

smoothing 값에 따라 train loss가 다르게 나타난다.

smoothing을 적용하면 loss가 조금 높아 보여도 일반화 성능이 좋아질 수 있다.

### 7.5 Accuracy와 Confidence 비교

Label Smoothing은 평균 confidence를 낮출 수 있다.

confidence가 낮아져도 test accuracy가 유지되거나 좋아지면 과신 완화 효과로 볼 수 있다.

### 7.6 Confusion Matrix

불균형 데이터에서는 정확도만 보지 않고 Confusion Matrix로 소수 class를 잡았는지 확인한다.

### 7.7 Loss Surface

색이 낮은 영역은 loss가 낮은 지점이다.

부드럽고 넓은 계곡은 학습이 안정적으로 진행되기 좋은 손실 곡면이다.

---

## 8. 실습에서 배운 점

- 손실함수는 모델의 채점 기준이다.
- 손실함수 선택에 따라 학습 방향이 달라진다.
- 회귀에서는 MSE, MAE, Huber를 상황에 맞게 선택한다.
- MSE는 큰 오차에 민감하다.
- MAE는 이상치에 비교적 강건하다.
- Huber는 MSE와 MAE의 절충안이다.
- BCE는 이진 분류 손실함수다.
- 실무에서는 BCEWithLogitsLoss가 수치적으로 더 안정적이다.
- CrossEntropyLoss는 다중 분류에 사용한다.
- CrossEntropyLoss에는 Softmax를 먼저 적용하지 않는다.
- Label Smoothing은 모델의 과신을 줄인다.
- 클래스 불균형에서는 accuracy만 보면 위험하다.
- 이진 불균형에는 pos_weight를 사용할 수 있다.
- 다중 분류 불균형에는 class weight를 사용할 수 있다.
- Focal Loss는 어려운 샘플에 집중하는 손실이다.
- Loss Surface는 파라미터 공간에서 손실이 어떻게 변하는지 보여준다.
- 안정적 학습을 위해 BatchNorm, Skip Connection, 초기화, gradient clipping, warm-up 등을 고려한다.

---

## 9. 시험용 요약

```text
손실함수 = 모델의 채점 기준 + 학습 방향 가이드
```

- 손실함수는 예측값과 정답의 차이를 수치화한다.
- Loss가 작을수록 모델이 정답에 가깝다.
- MSE는 오차 제곱 평균이다.
- MSE는 이상치에 매우 민감하다.
- MAE는 오차 절댓값 평균이다.
- MAE는 이상치에 더 강건하다.
- Huber Loss는 작은 오차에는 MSE, 큰 오차에는 MAE처럼 동작한다.
- BCE는 이진 분류 손실함수다.
- BCELoss에는 확률값을 넣는다.
- BCEWithLogitsLoss에는 raw logits를 넣는다.
- BCEWithLogitsLoss는 Sigmoid와 BCE를 내부에서 안정적으로 처리한다.
- CrossEntropyLoss는 다중 분류에 사용한다.
- CrossEntropyLoss에는 Softmax를 먼저 적용하지 않는다.
- CrossEntropyLoss의 정답은 one-hot이 아니라 class index다.
- Label Smoothing은 과신을 줄인다.
- Label Smoothing의 대표 alpha는 0.1이다.
- 불균형 데이터에서는 Accuracy만 보면 위험하다.
- pos_weight는 이진 분류에서 양성 class 가중치를 준다.
- CrossEntropyLoss의 weight는 다중 분류에서 class별 가중치를 준다.
- Focal Loss는 어려운 샘플에 더 집중한다.
- Loss Surface는 손실값이 파라미터 공간에서 어떻게 변하는지 나타낸다.
- 좋은 손실 곡면은 부드럽고 넓은 최소값을 가진다.