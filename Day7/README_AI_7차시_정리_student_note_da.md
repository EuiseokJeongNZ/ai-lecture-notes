# 강의_9기_AI개론_7차시_정리  
## 선형회귀 · OLS · nn.Linear · 단순회귀/중회귀 · R² Score · 정규화

## 1. 개요

이번 7차시는 머신러닝에서 가장 기본이 되는 선형회귀를 정리하는 강의다.

선형회귀는 입력값을 이용해 연속적인 수치값을 예측하는 회귀 모델이다.  
딥러닝 관점에서는 활성화 함수가 없는 뉴런 1개짜리 모델로 볼 수 있다.

핵심 흐름은 다음이다.

```text
회귀 문제 정의
→ nn.Linear 구조 이해
→ 단순 선형 함수 테스트
→ 커스텀 nn.Module 모델 정의
→ MSELoss로 오차 계산
→ Boston 주택 가격 데이터 흐름 이해
→ RM 하나로 가격 예측
→ 경사하강법 학습
→ 학습률에 따른 수렴/발산 확인
→ RM + LSTAT 중회귀 확장
→ R² Score로 설명력 평가
→ Ridge, Lasso, ElasticNet 정규화 이해
```

이번 강의의 핵심은 단순히 회귀식 하나를 외우는 것이 아니라, `nn.Linear`, `MSELoss`, `optimizer.step()`이 회귀 문제에서 어떻게 연결되는지 이해하는 것이다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| 회귀 Regression | 연속적인 실수값을 예측하는 문제 |
| 분류 Classification | 정해진 class label 중 하나를 고르는 문제 |
| 선형회귀 | 입력과 출력 사이의 선형 관계를 가정하는 모델 |
| OLS | 오차 제곱합을 최소화하는 최소제곱법 |
| nn.Linear | PyTorch의 선형 변환 Layer |
| Weight | 입력 feature의 영향력을 나타내는 학습 파라미터 |
| Bias | 기본 출력값을 보정하는 학습 파라미터 |
| nn.Module | PyTorch 커스텀 모델의 부모 클래스 |
| __init__ | 모델이 사용할 Layer를 선언하는 곳 |
| forward | 입력 데이터가 어떤 흐름으로 출력이 되는지 정의하는 곳 |
| MSELoss | 회귀 문제에서 자주 쓰는 평균 제곱 오차 |
| Gradient Descent | 손실을 줄이는 방향으로 파라미터를 업데이트하는 방법 |
| Learning Rate | 한 번에 파라미터를 얼마나 움직일지 정하는 값 |
| 단순회귀 | 입력 변수가 하나인 회귀 |
| 중회귀 | 입력 변수가 둘 이상인 회귀 |
| R² Score | 모델이 데이터 변동성을 얼마나 설명하는지 나타내는 지표 |
| Overfitting | 훈련 데이터에는 잘 맞지만 새로운 데이터에는 약한 상태 |
| Ridge | L2 정규화를 사용하는 회귀 |
| Lasso | L1 정규화를 사용하는 회귀 |
| ElasticNet | L1과 L2를 함께 사용하는 회귀 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_7차시_01_regression_ok.ipynb` | nn.Linear 실습, 커스텀 Net, Boston RM 단순회귀, 중회귀, 학습률 변경 |
| `강의_9기_AI개론_7차시_02_기본_딥러닝 및 머신러닝 이론.ipynb` | AI/ML/DL 개념, 지도/비지도/강화학습, DNN/CNN/RNN/Transformer/GAN 기본 설명 |
| `07. 선형회귀 OLS (Boston 데이터).srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_7차시_(선형회귀).pdf` | 선형회귀, OLS, R², 정규화 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. 회귀와 분류 차이 정리
3. 선형회귀 기본 수식 y = Wx + b 확인
4. OLS 최소제곱법 개념 정리
5. nn.Linear(2, 3) 생성 및 weight/bias 확인
6. nn.Linear(1, 1)로 y = 2x + 1 함수 만들기
7. x.view(-1, 1)로 Linear 입력 shape 맞추기
8. 2입력 1출력 선형 함수 테스트
9. 2입력 3출력 선형 함수 테스트
10. nn.Module을 상속한 Net 클래스 정의
11. net(inputs) 호출과 forward 흐름 확인
12. MSELoss로 손실 계산
13. loss.backward()로 gradient 계산
14. Boston 구조를 흉내 낸 housing 데이터 생성
15. RM 하나만 선택하여 단순회귀 문제 구성
16. 산점도로 RM과 PRICE 관계 확인
17. LinearRegressionNet 모델 정의
18. Tensor 변환 및 labels.view(-1, 1) 적용
19. criterion과 optimizer 설정
20. 단일 step으로 backward와 optimizer.step 확인
21. 반복 학습 루프 구현
22. loss history로 학습 곡선 출력
23. 학습된 회귀 직선 출력
24. R² Score 직접 계산
25. scikit-learn OLS와 PyTorch 결과 비교
26. RM + LSTAT 중회귀로 확장
27. 큰 학습률에서 발산 가능성 확인
28. 작은 학습률로 안정적 학습 확인
29. StandardScaler로 입력 표준화 후 학습
30. 단순회귀와 중회귀 성능 비교
31. R² Score 해석
32. 과적합 learning curve 확인
33. Ridge, Lasso, ElasticNet 정규화 비교
34. MSE, R², 정규화의 역할 구분
```

---

## 5. 주요 코드 블록 설명

### 5.1 nn.Linear 생성

```python
l = nn.Linear(2, 3)
```

입력 feature 2개를 받아 출력 feature 3개를 만드는 선형 함수다.

`weight` shape은 `[3, 2]`, `bias` shape은 `[3]`이다.

---

### 5.2 1입력 1출력 선형 함수

```python
l1 = nn.Linear(1, 1)
nn.init.constant_(l1.weight, 2.0)
nn.init.constant_(l1.bias, 1.0)
```

`y = 2x + 1` 함수를 직접 만든다.

---

### 5.3 view(-1, 1)

```python
x = x.view(-1, 1)
```

1차원 Tensor를 `[N, 1]` 형태의 2차원 Tensor로 바꾼다.

`nn.Linear`는 `[batch, feature]` 형태 입력을 기대한다.

---

### 5.4 2입력 1출력

```python
l2 = nn.Linear(2, 1)
```

입력 2개로 출력 1개를 만든다.

수식으로는 `y = w1*x1 + w2*x2 + b`다.

---

### 5.5 2입력 3출력

```python
l3 = nn.Linear(2, 3)
```

서로 다른 3개의 선형식을 동시에 계산하는 것처럼 볼 수 있다.

---

### 5.6 커스텀 모델 Net

```python
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.l1(x)
```

`__init__`에서는 Layer를 선언하고, `forward`에서는 데이터 흐름을 정의한다.

---

### 5.7 net(inputs)

```python
outputs = net(inputs)
```

모델 인스턴스를 함수처럼 호출한다.

내부적으로 `nn.Module`의 `__call__`이 실행되고, 그 안에서 `forward()`가 호출된다.

---

### 5.8 MSELoss

```python
criterion = nn.MSELoss()
loss = criterion(outputs, labels)
```

예측값과 정답 사이의 평균 제곱 오차를 계산한다.

회귀 문제에서 표준적으로 많이 쓰인다.

---

### 5.9 loss.backward()

```python
loss.backward()
```

손실을 기준으로 각 파라미터의 gradient를 계산한다.

결과는 `weight.grad`, `bias.grad`에 저장된다.

---

### 5.10 학습 루프

```python
optimizer.zero_grad()
outputs = net(inputs)
loss = criterion(outputs, labels1)
loss.backward()
optimizer.step()
```

PyTorch 회귀 학습의 기본 구조다.

이전 gradient를 지우고, 예측하고, 손실을 계산하고, 역전파한 뒤 파라미터를 수정한다.

---

### 5.11 Boston 구조 데이터

```python
RM = ...
LSTAT = ...
PRICE = ...
```

원본 강의에서는 Boston Housing 데이터에서 RM과 LSTAT를 사용한다.

Summary에서는 인터넷 없이 실행되도록 같은 구조의 synthetic housing 데이터를 사용한다.

---

### 5.12 단순회귀

```python
x = x_org[:, feature_names == "RM"]
```

RM 하나만 사용해 PRICE를 예측한다.

입력 변수가 하나이므로 단순회귀다.

---

### 5.13 중회귀

```python
x_multi = np.hstack([x, x_add])
```

RM과 LSTAT 두 feature를 함께 사용한다.

입력 변수가 둘 이상이므로 중회귀다.

---

### 5.14 학습률 발산

```python
optimizer = optim.SGD(net.parameters(), lr=0.1)
```

학습률이 너무 크면 손실이 발산해 `inf`나 `nan`이 나올 수 있다.

---

### 5.15 StandardScaler

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_multi)
```

feature 평균을 0, 표준편차를 1에 가깝게 만든다.

입력 스케일을 맞추면 학습이 안정적일 수 있다.

---

### 5.16 R² Score

```python
R² = 1 - SS_res / SS_tot
```

모델이 전체 데이터 변동성을 얼마나 설명하는지 나타낸다.

1에 가까울수록 좋고, 0은 평균 모델과 비슷하며, 음수는 평균보다 못한 모델일 수 있다.

---

### 5.17 Ridge

```python
Ridge(alpha=1.0)
```

L2 정규화를 적용한다.

가중치 제곱합을 패널티로 줘서 계수가 너무 커지는 것을 막는다.

PyTorch에서는 `weight_decay`가 L2 정규화와 연결된다.

---

### 5.18 Lasso

```python
Lasso(alpha=0.05)
```

L1 정규화를 적용한다.

덜 중요한 feature의 계수를 0으로 만들 수 있어 feature selection 효과가 있다.

---

### 5.19 ElasticNet

```python
ElasticNet(alpha=0.05, l1_ratio=0.5)
```

L1과 L2를 함께 사용하는 정규화 방식이다.

Lasso의 희소성과 Ridge의 안정성을 같이 노린다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `x` | 입력 데이터 | feature 값을 담는다 |
| `y`, `yt` | 정답 데이터 | 예측해야 하는 target |
| `RM` | 방 평균 개수 | 주택 가격 예측 feature |
| `LSTAT` | 저소득자 비율 | 추가 feature |
| `W`, `weight` | 가중치 | feature 영향력 |
| `b`, `bias` | 편향 | 기본 출력값 |
| `nn.Linear` | 선형 Layer | `nn.Linear(in_features, out_features)` |
| `in_features` | 입력 feature 수 | 입력 열 개수 |
| `out_features` | 출력 feature 수 | 예측값 개수 |
| `nn.Module` | PyTorch 모델 부모 클래스 | 커스텀 모델 정의 시 상속 |
| `__init__` | 모델 뼈대 정의 | Layer 선언 |
| `forward` | 데이터 흐름 정의 | 순전파 로직 |
| `criterion` | 손실 함수 | `nn.MSELoss()` |
| `optimizer` | 최적화 함수 | `optim.SGD(...)` |
| `lr` | learning rate | 학습률 |
| `epoch` | 반복 횟수 | 전체 학습 반복 단위 |
| `loss` | 손실 | 예측과 정답의 차이 |
| `MSE` | 평균 제곱 오차 | 회귀 표준 손실 |
| `OLS` | 최소제곱법 | 오차 제곱합 최소화 |
| `R²` | 결정계수 | 설명력 평가 |
| `Ridge` | L2 정규화 | `weight_decay`와 연결 |
| `Lasso` | L1 정규화 | feature selection 효과 |
| `ElasticNet` | L1+L2 정규화 | Ridge와 Lasso 결합 |

---

## 7. 그래프/출력 결과 해석

### 7.1 RM과 PRICE 산점도

RM이 증가할수록 PRICE가 대체로 증가하는 양의 관계를 확인한다.

이 관계를 가장 잘 대표하는 직선을 찾는 것이 단순회귀의 목표다.

### 7.2 학습 곡선

Loss가 epoch가 증가할수록 감소하면 학습이 정상적으로 진행되고 있다는 뜻이다.

Loss가 `inf`나 `nan`이 되면 학습률이 너무 크거나 입력 스케일 문제가 있을 수 있다.

### 7.3 회귀 직선

산점도 위에 학습된 직선을 그려 모델이 데이터의 큰 흐름을 따라가는지 확인한다.

### 7.4 중회귀 학습 곡선

RM과 LSTAT를 함께 사용하면 더 많은 정보를 이용할 수 있다.

다만 feature scale과 learning rate가 맞지 않으면 발산할 수 있다.

### 7.5 과적합 그래프

Train Loss는 계속 감소하지만 Validation Loss가 다시 증가하면 과적합을 의심한다.

### 7.6 정규화 비교

Ridge는 계수를 안정적으로 줄이고, Lasso는 일부 계수를 0으로 만들 수 있다.

ElasticNet은 두 성질을 함께 사용한다.

---

## 8. 실습에서 배운 점

- 선형회귀는 연속값을 예측하는 가장 기본적인 모델이다.
- `nn.Linear`는 선형회귀와 딥러닝의 기본 블록이다.
- `nn.Linear`의 입력/출력 차원을 정확히 맞춰야 한다.
- `view(-1, 1)`은 1차원 데이터를 Linear 입력 형태로 맞출 때 자주 쓴다.
- PyTorch 모델은 `nn.Module`을 상속해서 정의한다.
- `__init__`은 Layer를 선언하고, `forward`는 데이터 흐름을 정의한다.
- `net(inputs)`는 내부적으로 `forward(inputs)`를 실행한다.
- 회귀에서는 `MSELoss`를 표준 손실로 많이 사용한다.
- 학습 루프는 `zero_grad → forward → loss → backward → step` 순서다.
- 단순회귀는 입력 변수가 하나다.
- 중회귀는 입력 변수가 둘 이상이다.
- 학습률이 너무 크면 loss가 발산할 수 있다.
- 입력 표준화는 학습 안정성에 도움을 준다.
- R²는 모델의 설명력을 보는 평가 지표다.
- Ridge, Lasso, ElasticNet은 과적합을 줄이기 위한 정규화 기법이다.

---

## 9. 시험용 요약

```text
선형회귀 = y = Wx + b
```

- 회귀는 연속적인 실수값을 예측한다.
- 분류는 class label을 예측한다.
- 선형회귀는 입력과 출력의 선형 관계를 가정한다.
- OLS는 오차 제곱합을 최소화하는 방법이다.
- `nn.Linear`는 PyTorch의 선형 변환 Layer다.
- `nn.Linear(in_features, out_features)`에서 차원을 정확히 맞춰야 한다.
- weight shape은 `[out_features, in_features]`다.
- bias shape은 `[out_features]`다.
- 1입력 1출력은 `y = wx + b`다.
- 2입력 1출력은 `y = w1x1 + w2x2 + b`다.
- 커스텀 모델은 `nn.Module`을 상속한다.
- `__init__`은 Layer 선언, `forward`는 데이터 흐름 정의다.
- `net(inputs)`를 호출하면 `forward`가 실행된다.
- 회귀 손실함수로 `MSELoss`를 많이 쓴다.
- 학습 순서는 `zero_grad → forward → loss → backward → step`이다.
- 입력 변수가 하나면 단순회귀다.
- 입력 변수가 둘 이상이면 중회귀다.
- 학습률이 너무 크면 `inf`, `nan`으로 발산할 수 있다.
- StandardScaler는 feature scale을 맞춰 학습을 안정화한다.
- R²는 모델이 데이터 변동성을 얼마나 설명하는지 나타낸다.
- Ridge는 L2 정규화다.
- Lasso는 L1 정규화다.
- ElasticNet은 L1과 L2를 결합한다.