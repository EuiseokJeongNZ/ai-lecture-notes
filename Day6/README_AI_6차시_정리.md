# 강의_9기_AI개론_6차시_정리  
## 손실함수(Loss) 설계

## 1. 개요

이번 강의는 모델 학습에서 매우 중요한 **손실함수 Loss Function**를 다룬다.

손실함수는 모델의 예측값과 실제 정답 사이의 차이를 수치화하는 도구이다. 이 값이 작을수록 모델이 더 정확하게 예측하고 있다는 의미이다.

손실함수의 핵심 역할은 다음과 같다.

```text
1. 모델의 오류 정도 측정
2. 학습 방향 결정
3. Optimizer가 따라갈 기준 제공
```

즉, 손실함수는 모델이 어디를 고쳐야 하는지 알려주는 나침반이다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 설명 |
|---|---|
| Loss Function | 예측값과 정답의 차이를 수치화하는 함수 |
| MSE | 평균 제곱 오차, 큰 오차에 큰 패널티 |
| MAE | 평균 절대 오차, 이상치에 상대적으로 강건 |
| Huber Loss | MSE와 MAE의 절충안 |
| BCE | 이진 분류용 손실함수 |
| BCEWithLogitsLoss | Sigmoid + BCE를 수치적으로 안정하게 결합 |
| CrossEntropyLoss | 다중 분류용 손실함수 |
| Label Smoothing | 정답 라벨을 부드럽게 만들어 모델 과신을 줄이는 기법 |
| Weighted CE | 클래스 불균형 대응을 위해 클래스별 가중치를 주는 손실 |
| Focal Loss | 쉬운 샘플보다 어려운 샘플에 집중하는 손실 |
| Loss Surface | 가중치 공간에서 손실값의 변화를 나타내는 지형 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_6차시_01_MSE vs BCEWithLogits 비교와 Label Smoothing_ok.ipynb` | 이진 분류 손실 비교, BCEWithLogits, Label Smoothing, Weighted CE |
| `강의_9기_AI개론_6차시_02_회귀손실함수 비교_ok.ipynb` | MSE, MAE, Huber 회귀 손실 비교 |
| `6차-1.srt` | 강사님 설명 스크립트 1 |
| `6차-2.srt` | 강사님 설명 스크립트 2 |
| `강의_9기_AI개론_6차시_(비용함수정의).pdf` | 강의 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. MSE, MAE, Huber 기본 계산
3. 이상치가 있을 때 손실함수별 민감도 비교
4. 회귀 데이터 생성 및 이상치 추가
5. 회귀 모델 정의
6. MSE, MAE, Huber로 각각 학습
7. 학습 곡선과 테스트 MAE 비교
8. Huber delta 값에 따른 손실 곡선 확인
9. 이진 분류 데이터 생성
10. Binary Classification 모델 정의
11. MSE와 BCEWithLogitsLoss 비교
12. BCEWithLogitsLoss 사용법 확인
13. CrossEntropyLoss 입력 형식 확인
14. 다중 분류 데이터 준비
15. CNN 모델 정의
16. Label Smoothing 적용
17. Weighted Cross Entropy 적용
18. Focal Loss 개념 구현
19. Loss Surface와 안정적 학습 전략 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 MSE

```python
mse_loss = nn.MSELoss()
mse_value = mse_loss(y_pred, y_true)
```

예측값과 실제값의 차이를 제곱한 뒤 평균낸다.

큰 오차에 큰 패널티를 주므로 이상치에 민감하다.

---

### 5.2 MAE

```python
mae_loss = nn.L1Loss()
mae_value = mae_loss(y_pred, y_true)
```

오차의 절댓값 평균을 계산한다.

이상치에 상대적으로 강건하고 해석이 직관적이다.

---

### 5.3 Huber Loss

```python
huber_loss = nn.HuberLoss(delta=1.0)
```

작은 오차에는 MSE처럼, 큰 오차에는 MAE처럼 동작한다.

`delta`는 두 방식이 바뀌는 기준값이다.

---

### 5.4 회귀 모델

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

10개의 feature를 입력받아 숫자 하나를 예측하는 회귀 모델이다.

---

### 5.5 이진 분류 모델

```python
class BinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.m(x)
```

출력은 확률이 아니라 raw logit 1개이다.

---

### 5.6 BCEWithLogitsLoss

```python
loss = nn.BCEWithLogitsLoss()(out, y)
```

Sigmoid와 BCE를 내부에서 안정적으로 처리한다.

모델 마지막에 Sigmoid를 붙이지 않는 것이 핵심이다.

---

### 5.7 CrossEntropyLoss

```python
loss = nn.CrossEntropyLoss()(logits, target)
```

다중 분류에서 사용한다.

- 예측값: `[batch, class_count]` raw logits
- 정답값: `[batch]` 클래스 인덱스

Softmax는 내부에 포함되어 있다.

---

### 5.8 Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

정답을 100% 확신하지 않도록 부드럽게 만들어 모델 과신을 줄인다.

---

### 5.9 Weighted Cross Entropy

```python
weights = torch.tensor([...])
criterion = nn.CrossEntropyLoss(weight=weights)
```

특정 클래스의 손실 가중치를 높여 클래스 불균형 문제에 대응한다.

---

### 5.10 Focal Loss

```python
focal_weight = alpha * (1 - pt) ** gamma
loss = focal_weight * bce
```

쉬운 샘플의 영향은 줄이고 어려운 샘플에 더 집중한다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 의미 | 설명 |
|---|---|---|
| `Loss` | 손실 | 모델이 틀린 정도 |
| `Cost Function` | 비용 함수 | 손실함수와 거의 같은 의미 |
| `MSE` | Mean Squared Error | 평균 제곱 오차 |
| `MAE` | Mean Absolute Error | 평균 절대 오차 |
| `L1Loss` | MAE | PyTorch의 MAE 손실 |
| `Huber` | Huber Loss | MSE와 MAE 절충 |
| `delta` | Huber 경계값 | MSE/MAE 전환 기준 |
| `BCE` | Binary Cross Entropy | 이진 분류 손실 |
| `BCEWithLogitsLoss` | 안정적 BCE | Sigmoid와 BCE 결합 |
| `logit` | raw score | Sigmoid/Softmax 전 원시 출력 |
| `CrossEntropyLoss` | 다중 분류 손실 | Softmax 포함 |
| `Label Smoothing` | 라벨 스무딩 | 정답 라벨을 부드럽게 만듦 |
| `Class Weight` | 클래스 가중치 | 소수 클래스 손실을 크게 반영 |
| `Focal Loss` | 어려운 샘플 집중 손실 | 불균형 분류에 사용 |
| `alpha` | Focal 가중치 | 클래스 불균형 보정 |
| `gamma` | Focal 집중도 | 쉬운 샘플 손실 감소 정도 |
| `Loss Surface` | 손실 곡면 | 가중치 공간의 손실 지형 |

---

## 7. 그래프/출력 결과 해석

### 7.1 회귀 손실함수 학습 곡선

MSE, MAE, Huber의 손실 감소 흐름을 비교한다.

손실함수마다 값의 스케일이 다르므로 절대값만 단순 비교하면 안 된다.

### 7.2 테스트 MAE 비교

평균적으로 얼마나 틀렸는지 직관적으로 확인할 수 있다.

이상치가 있는 문제에서는 MSE뿐 아니라 MAE도 함께 확인하는 것이 좋다.

### 7.3 Huber delta 그래프

`delta`가 작으면 MAE처럼 빠르게 전환되고, `delta`가 크면 MSE에 가까운 구간이 넓어진다.

### 7.4 BCEWithLogits 비교

이진 분류에서는 MSE보다 BCEWithLogitsLoss가 더 적합하다.

### 7.5 Label Smoothing 결과

모델이 정답에 지나치게 확신하지 않게 하여 일반화 성능을 높이는 데 도움을 줄 수 있다.

---

## 8. 실습에서 배운 점

- 손실함수는 모델 학습 방향을 결정하는 핵심 기준이다.
- 회귀 문제와 분류 문제는 사용하는 손실함수가 다르다.
- MSE는 큰 오차에 민감하다.
- MAE는 이상치에 강건하다.
- Huber는 MSE와 MAE의 절충안이다.
- 이진 분류에는 BCEWithLogitsLoss가 안정적이다.
- 다중 분류에는 CrossEntropyLoss를 사용한다.
- CrossEntropyLoss에는 Softmax가 내부적으로 포함되어 있다.
- Label Smoothing은 모델 과신을 줄인다.
- Class Weight와 Focal Loss는 클래스 불균형 문제에 대응한다.
- 좋은 학습에는 손실함수뿐 아니라 초기화, 학습률, BatchNorm, gradient clipping도 중요하다.

---

## 9. 시험용 요약

```text
손실함수 = 오차 측정 + 학습 방향 결정
```

- MSE는 평균 제곱 오차이며 이상치에 민감하다.
- MAE는 평균 절대 오차이며 이상치에 상대적으로 강건하다.
- Huber Loss는 작은 오차에는 MSE, 큰 오차에는 MAE처럼 동작한다.
- 이진 분류에는 BCEWithLogitsLoss를 사용한다.
- BCEWithLogitsLoss를 사용할 때는 모델 마지막에 Sigmoid를 붙이지 않는다.
- 다중 분류에는 CrossEntropyLoss를 사용한다.
- CrossEntropyLoss의 입력은 raw logits이다.
- CrossEntropyLoss의 정답은 one-hot이 아니라 클래스 인덱스이다.
- Label Smoothing은 정답 라벨을 부드럽게 만들어 모델 과신을 줄인다.
- Weighted CE는 클래스별 손실 가중치를 다르게 준다.
- Focal Loss는 어려운 샘플에 집중한다.
- Loss Surface는 가중치 공간에서 손실값의 지형이다.
- 안정적인 학습에는 손실함수, 초기화, 학습률, BatchNorm, gradient clipping이 함께 중요하다.